#!/usr/bin/env python3
"""
Pipeline de teste: poda ESTRUTURAL de YOLOv10 (n/s/m/b/l/x) via torch-pruning.

Objetivo: demonstrar que a poda estrutural funciona para YOLOv10 — reduz
Params/FLOPs/tamanho de verdade — mantendo mAP alto apos fine-tuning, com as
mesmas metricas de desktop (GPU) da tabela de experimentos:
  Params (M), Taxa compressao (%), mAP50, mAP50-95, Precision, Recall, F1,
  Inferencia (ms), FPS, FLOPs (G), Tamanho em disco (MB).

Etapas:
  1. Carrega o checkpoint Ultralytics (best.pt)
  2. Converte C2f/C2fCIB -> split-free e VERIFICA equivalencia numerica
  3. Poda estrutural global L1 (ignora cabeca v10Detect e blocos PSA/Attention)
  4. Sanity: forward pos-poda + Params/FLOPs antes/depois
  5. (--sanity-only para parar aqui) Fine-tuning via DetectionTrainer
  6. Avaliacao no split de validacao + latencia GPU -> linha em results.csv

Exemplos:
  # Sanity rapido (CPU ou GPU, sem dataset): prova que a poda reduz o modelo
  python prune_pipeline.py --model weights/yolov10n.pt --ratio 0.4 --sanity-only

  # Baseline (sem poda, so avalia e registra a linha)
  python prune_pipeline.py --model best.pt --data data.yaml --ratio 0

  # Poda 40% + fine-tuning 30 epocas + avaliacao
  python prune_pipeline.py --model runs/train/yolov10n_oxford_finetune/weights/best.pt \
      --data /datasets/oxford/data.yaml --ratio 0.4 --ft-epochs 30 --out results
"""

import argparse
import csv
import time
from copy import deepcopy
from pathlib import Path

import torch

from c2f_splitfree import convert_c2f_splitfree  # noqa: F401 (necessario p/ unpickle)

CSV_FIELDS = [
    "modelo", "compressao", "params_M", "taxa_compressao_pct",
    "mAP50", "mAP50_95", "precision", "recall", "F1",
    "inferencia_ms", "FPS", "FLOPs_G", "tamanho_MB", "checkpoint",
]


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------

def flatten_tensors(obj):
    """Extrai todos os tensores de uma saida arbitraria (tensor/list/tuple/dict)."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from flatten_tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from flatten_tensors(o)


@torch.no_grad()
def forward_tensors(model, x):
    return [t.detach().float().clone() for t in flatten_tensors(model(x))]


@torch.no_grad()
def forward_head_inputs(det_model, x):
    """Captura os tensores de ENTRADA da cabeca de deteccao (saidas do neck).

    E o ponto certo para o teste de equivalencia da conversao split-free: cobre
    toda a regiao convertida (todos os C2f ficam antes da cabeca) e evita o
    pos-processamento do v10Detect, cujo topk reordena deteccoes com scores
    quase iguais e transforma diferencas de 1e-6 em linhas trocadas."""
    captured = []

    def _hook(module, inputs):
        captured.extend(t.detach().float().clone()
                        for t in flatten_tensors(inputs))

    h = det_model.model[-1].register_forward_pre_hook(_hook)
    try:
        det_model(x)
    finally:
        h.remove()
    return captured


def check_equivalence(ref, new, atol=1e-4):
    assert len(ref) == len(new), f"n. de saidas difere: {len(ref)} vs {len(new)}"
    worst = 0.0
    for a, b in zip(ref, new):
        assert a.shape == b.shape, f"shape difere: {a.shape} vs {b.shape}"
        worst = max(worst, (a - b).abs().max().item())
    assert worst < atol, f"conversao NAO e equivalente (max diff {worst:.3e})"
    return worst


def count_ops_params(model, example):
    import torch_pruning as tp
    macs, params = tp.utils.count_ops_and_params(model, example)
    return macs, params


def build_ignored_layers(det_model):
    """Cabeca de deteccao (canais presos a nc/anchors + pos-processamento
    topk do v10) e blocos de atencao PSA (split interno de canais que o
    torch-pruning nao rastreia). Mesma politica do pipeline original."""
    ignored = [det_model.model[-1]]
    for m in det_model.modules():
        if m.__class__.__name__ in ("PSA", "C2PSA", "PSABlock", "Attention"):
            ignored.append(m)
    return ignored


@torch.no_grad()
def measure_latency(det_model, imgsz, device, n_iter=200, n_warmup=50):
    """Latencia de forward (batch 1, fp32) com sincronizacao CUDA."""
    det_model.eval().to(device)
    x = torch.randn(1, 3, imgsz, imgsz, device=device)
    for _ in range(n_warmup):
        det_model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        det_model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_iter * 1000.0
    return ms, 1000.0 / ms


def append_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"[csv] linha adicionada em {csv_path}")


# ---------------------------------------------------------------------------
# Etapas principais
# ---------------------------------------------------------------------------

def load_model(model_path, device):
    from ultralytics import YOLO
    yolo = YOLO(str(model_path))
    det = yolo.model.to(device).float().eval()
    return yolo, det


def prune_structured(det, ratio, imgsz, device, round_to=None, verbose=True):
    """Converte C2f -> split-free (com checagem de equivalencia) e aplica poda
    estrutural global L1 via torch-pruning. Retorna dict com contagens."""
    import torch_pruning as tp

    example = torch.randn(1, 3, imgsz, imgsz, device=device)

    base_macs, base_params = count_ops_params(det, example)

    # 1) conversao split-free + prova de equivalencia numerica
    ref = forward_head_inputs(det, example)
    n_conv = convert_c2f_splitfree(det)
    new = forward_head_inputs(det, example)
    diff = check_equivalence(ref, new)
    if verbose:
        print(f"[conversao] {n_conv} blocos C2f/C2fCIB convertidos — "
              f"equivalencia OK (max diff {diff:.2e})")

    if ratio <= 0:
        return {"base_macs": base_macs, "base_params": base_params,
                "macs": base_macs, "params": base_params}

    # 2) poda estrutural global L1
    ignored = build_ignored_layers(det)
    # o DependencyGraph rastreia via grad_fn — precisa de requires_grad ligado
    # em tudo durante o traco (mesma observacao do pipeline original)
    req = {n: p.requires_grad for n, p in det.named_parameters()}
    det.requires_grad_(True)

    imp = tp.importance.MagnitudeImportance(p=1)  # L1, igual ao criterio atual
    pruner = tp.pruner.MagnitudePruner(
        det, example,
        importance=imp,
        pruning_ratio=ratio,
        global_pruning=True,
        ignored_layers=ignored,
        round_to=round_to,
    )
    pruner.step()

    for n, p in det.named_parameters():      # restaura congelamentos originais
        if n in req:
            p.requires_grad_(req[n])

    # 3) sanity pos-poda
    with torch.no_grad():
        det(example)
    macs, params = count_ops_params(det, example)
    if verbose:
        print(f"[poda {ratio:.0%}] params: {base_params/1e6:.2f}M -> {params/1e6:.2f}M "
              f"({100*(1-params/base_params):.1f}% removidos) | "
              f"MACs: {base_macs/1e9:.2f}G -> {macs/1e9:.2f}G "
              f"({100*(1-macs/base_macs):.1f}% removidos)")
    return {"base_macs": base_macs, "base_params": base_params,
            "macs": macs, "params": params}


def finetune(det, model_path, data_yaml, out_dir, run_name, epochs, batch,
             imgsz, lr, device_arg, workers, amp=True):
    """Fine-tuning do modelo JA PODADO via DetectionTrainer.

    Truque essencial: atribuir o nn.Module podado a trainer.model ANTES de
    train() — BaseTrainer.setup_model() ve um nn.Module pronto e NAO
    reconstroi a arquitetura a partir do yaml (o que silenciosamente
    descartaria os pesos podados por incompatibilidade de shape; e esse o
    motivo de yolo.train() 'puro' nao funcionar apos poda estrutural)."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    overrides = dict(
        model=str(model_path),      # so para registro de args/config
        data=str(data_yaml),
        epochs=epochs, batch=batch, imgsz=imgsz,
        optimizer="SGD", lr0=lr, warmup_epochs=0.0,   # receita do pipeline original
        amp=amp, device=device_arg, workers=workers,
        project=str(Path(out_dir).resolve()), name=run_name, exist_ok=True,
        patience=0, verbose=True,
    )
    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = det.float().train()
    trainer.train()
    best = Path(trainer.best)
    print(f"[fine-tuning] concluido — best: {best}")
    return best


def evaluate(best_path, data_yaml, imgsz, device, device_arg, split="val"):
    """Metricas de validacao (ultralytics) + latencia/FPS + FLOPs/params."""
    from ultralytics import YOLO
    yolo = YOLO(str(best_path))   # requer c2f_splitfree importavel (unpickle)
    det = yolo.model.to(device).float().eval()

    example = torch.randn(1, 3, imgsz, imgsz, device=device)
    macs, params = count_ops_params(det, example)
    ms, fps = measure_latency(det, imgsz, device)

    m = yolo.val(data=str(data_yaml), imgsz=imgsz, split=split,
                 device=device_arg, verbose=False, plots=False)
    p, r = float(m.box.mp), float(m.box.mr)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "params_M": round(params / 1e6, 3),
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "F1": round(f1, 4),
        "inferencia_ms": round(ms, 2),
        "FPS": round(fps, 1),
        "FLOPs_G": round(2 * macs / 1e9, 2),   # convencao FLOPs = 2*MACs
        "tamanho_MB": round(Path(best_path).stat().st_size / 1e6, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Checkpoint .pt (Ultralytics)")
    p.add_argument("--data", help="data.yaml do dataset (obrigatorio sem --sanity-only)")
    p.add_argument("--ratio", type=float, default=0.4,
                   help="Fracao de canais a remover (0 = baseline, so avalia)")
    p.add_argument("--ft-epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0", help="'0' p/ GPU, 'cpu' p/ CPU")
    p.add_argument("--round-to", type=int, default=None,
                   help="Arredondar canais a multiplos de N (ex.: 8/16, ajuda latencia GPU)")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--out", default="results", help="Diretorio de saida")
    p.add_argument("--label", default=None, help="Rotulo do modelo no CSV")
    p.add_argument("--sanity-only", action="store_true",
                   help="So converte + poda + conta params/FLOPs (sem FT/avaliacao)")
    p.add_argument("--no-finetune", action="store_true",
                   help="Poda e avalia SEM fine-tuning (mostra o dano bruto da poda)")
    return p.parse_args()


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    device = torch.device(f"cuda:{args.device}" if use_cuda else "cpu")
    device_arg = args.device if use_cuda else "cpu"
    out_dir = Path(args.out)
    label = args.label or Path(args.model).stem
    ratio_pct = int(round(args.ratio * 100))
    cond = "baseline" if args.ratio <= 0 else f"pruning_estruturado_{ratio_pct}pct"
    run_name = f"{label}_{cond}"

    print("=" * 70)
    print(f"  PODA ESTRUTURAL YOLOv10 | modelo={args.model} | ratio={args.ratio}")
    print(f"  device={device} | saida={out_dir}")
    print("=" * 70)

    yolo, det = load_model(args.model, device)
    counts = prune_structured(det, args.ratio, args.imgsz, device,
                              round_to=args.round_to)

    if args.sanity_only:
        print("[sanity-only] OK — conversao equivalente, poda aplicada, forward valido.")
        return

    if not args.data:
        raise SystemExit("--data e obrigatorio sem --sanity-only")

    if args.ratio > 0 and not args.no_finetune:
        best = finetune(det, args.model, args.data, out_dir, run_name,
                        args.ft_epochs, args.batch, args.imgsz, args.lr,
                        device_arg, args.workers)
    else:
        # salva o modelo atual (baseline ou podado-sem-FT) em formato ultralytics
        best = out_dir / run_name / "model.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": deepcopy(det).half(), "train_args": {}}, best)

    metrics = evaluate(best, args.data, args.imgsz, device, device_arg,
                       split=args.split)
    row = {
        "modelo": label,
        "compressao": cond + ("_sem_ft" if args.no_finetune and args.ratio > 0 else ""),
        "taxa_compressao_pct": round(100 * (1 - counts["params"] / counts["base_params"]), 2),
        "checkpoint": str(best),
        **metrics,
    }
    append_csv(out_dir / "results.csv", row)
    print("\nResumo:")
    for k in CSV_FIELDS:
        print(f"  {k:22s}: {row[k]}")


if __name__ == "__main__":
    main()
