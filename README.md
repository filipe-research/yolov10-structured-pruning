# Poda estrutural de YOLOv10 — repositório de teste

Prova de conceito de que **poda estrutural funciona para YOLOv10** (variantes
n/s/m/b/l/x), reduzindo Params, FLOPs e tamanho em disco de verdade — ao
contrário da poda L1 não-estrutural (máscaras), que zera pesos mas mantém a
forma dos tensores e por isso não muda Params/FLOPs nem consumo em hardware
denso (GPU/CPU ARM).

## Por que a poda estrutural "não funcionava"

O forward do bloco C2f/C2fCIB faz `self.cv1(x).chunk(2, 1)` — um split de
canais que o DependencyGraph do `torch-pruning` não rastreia. A solução (mesma
ideia do exemplo oficial `yolov8_pruning.py` do torch-pruning) é converter o
`cv1` (saída 2c) em dois convs separados `cv0`/`cv1` (saída c cada), copiando
as metades dos pesos. A conversão é **matematicamente idêntica** e o pipeline
verifica isso numericamente a cada execução (diff máx. medido: **0.0**).

A cabeça de detecção (`v10Detect`, canais presos a nc/anchors + pós-
processamento top-k) e os blocos de atenção PSA (split interno de canais)
ficam fora da poda — por isso a taxa efetiva de remoção difere do ratio
nominal; **reporte sempre a taxa medida**, que o pipeline imprime e grava.

## Validação já executada (CPU, pesos COCO oficiais)

| Modelo   | Ratio | Params        | MACs          | Forward pós-poda |
|----------|-------|---------------|---------------|------------------|
| yolov10n | 40%   | 2.78M → 1.83M (−34.0%) | 4.33G → 2.85G (−34.2%) | OK |
| yolov10n | 80%   | 2.78M → 1.24M (−55.4%) | 4.33G → 2.15G (−50.3%) | OK |
| yolov10m | 40%   | 16.58M → 8.88M (−46.4%) | 32.10G → 15.22G (−52.6%) | OK |
| yolov10x | 40%   | 31.81M → 17.10M (−46.2%) | 85.62G → 41.26G (−51.8%) | OK |
| yolov10x | 60%   | 31.81M → 10.55M (−66.8%) | 85.62G → 22.21G (−74.1%) | OK |

Conversão split-free com equivalência exata (max diff 0.00e+00) em todas.
O ciclo completo poda → fine-tuning (DetectionTrainer) → best.pt → reload →
val → CSV também foi validado (1 época, dataset sintético): o checkpoint salvo
mantém a arquitetura podada (best.pt de 3.98 MB vs 5.86 MB do original).

## Instalação (máquina com RTX 5090)

```bash
pip install torch torchvision          # wheel atual já cobre a 5090 (sm_120)
pip install -r requirements.txt
```

Se aparecer "no kernel image is available" na 5090:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

## Uso

```bash
# 1) Sanity (sem dataset): prova conversão + poda + contagens
python prune_pipeline.py --model weights/yolov10n.pt --ratio 0.4 --sanity-only

# 2) Smoke test do ciclo completo (dataset sintético, ~2 min)
python make_tiny_dataset.py
python prune_pipeline.py --model weights/yolov10n.pt --data tiny_dataset/data.yaml \
    --ratio 0.4 --ft-epochs 1 --batch 4 --imgsz 320 --out results_smoke

# 3) Experimento real: best.pt fine-tunado na base (Oxford/CAVIAR) + data.yaml
python prune_pipeline.py \
    --model runs/train/yolov10n_oxford_finetune/weights/best.pt \
    --data /datasets/oxford/data.yaml \
    --ratio 0.4 --ft-epochs 30 --label yolov10n --out results_oxford

# baseline (ratio 0) para a linha de referência no mesmo CSV
python prune_pipeline.py --model .../best.pt --data .../data.yaml --ratio 0 \
    --label yolov10n --out results_oxford

# 4) Varredura completa (edite os caminhos no topo)
DATA=/datasets/oxford/data.yaml MODELS_DIR=/runs/train bash run_all.sh
```

Cada execução adiciona uma linha em `<out>/results.csv` com as mesmas métricas
da tabela de experimentos: Params (M), taxa de compressão (%), mAP50,
mAP50-95, Precision, Recall, F1, inferência (ms), FPS, FLOPs (G) e tamanho
em disco (MB). P/R/F1 vêm do `val` do ultralytics (limiar no máximo do F1 —
consistente entre modelos). Latência/FPS: forward batch 1 fp32, 200 iterações
com warmup e `cuda.synchronize`.

## Detalhes que importam

- **`c2f_splitfree.py` precisa estar importável ao carregar qualquer
  checkpoint podado** (o pickle guarda a referência da classe). Rode scripts
  a partir da raiz do repo; para usar o modelo em outro projeto (ex.: export
  para o Raspberry), copie o arquivo junto.
- **Por que `yolo.train()` puro não funciona após a poda:** o trainer
  reconstrói a arquitetura a partir do yaml e carrega pesos por interseção de
  shapes — os pesos podados seriam descartados em silêncio. O pipeline
  contorna isso atribuindo o nn.Module já podado a `trainer.model` antes de
  `train()` (o `setup_model()` detecta o módulo pronto e não reconstrói).
- **Critério de importância:** L1 global (`MagnitudeImportance(p=1)`), o mesmo
  do pipeline original — a comparação com a poda não-estrutural fica pareada.
- **Gancho para o Energy-Aware Pruning:** trocar a `importance` em
  `prune_structured()` por uma importância customizada que divida a magnitude
  pelo custo energético estimado do grupo de canais (modelo empírico de
  energia do Raspberry). A API do torch-pruning aceita qualquer
  `tp.importance.Importance`.
- `--round-to 8` (ou 16) arredonda canais a múltiplos — costuma melhorar a
  latência real em GPU/NCNN com perda mínima de compressão.
- `--no-finetune` poda e avalia direto: útil para mostrar o dano bruto da
  poda antes da recuperação por fine-tuning.
- Export para o Raspberry (ONNX/NCNN): `YOLO(best).export(format="onnx")`
  funciona normalmente — o modelo podado é um DetectionModel comum, só menor.
