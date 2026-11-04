#!/usr/bin/env bash
# Varredura: baseline + poda estrutural 20/40/60/80% para as variantes do
# estudo, com fine-tuning e avaliacao. Ajuste MODELS_DIR e DATA para o seu
# ambiente (os best.pt fine-tunados do Andre e o data.yaml da base).
set -e

DATA="${DATA:-/caminho/para/oxford/data.yaml}"
MODELS_DIR="${MODELS_DIR:-/caminho/para/runs/train}"   # yolov10{v}_oxford_finetune/weights/best.pt
FT_EPOCHS="${FT_EPOCHS:-30}"
OUT="${OUT:-results_oxford}"

for v in yolov10n yolov10m yolov10l yolov10x; do
  BEST="$MODELS_DIR/${v}_oxford_finetune/weights/best.pt"
  [ -f "$BEST" ] || { echo "pulado (nao encontrado): $BEST"; continue; }

  # baseline (so avalia, registra a linha de referencia no mesmo CSV)
  python prune_pipeline.py --model "$BEST" --data "$DATA" --ratio 0 \
      --label "$v" --out "$OUT"

  for r in 0.2 0.4 0.6 0.8; do
    python prune_pipeline.py --model "$BEST" --data "$DATA" --ratio "$r" \
        --ft-epochs "$FT_EPOCHS" --label "$v" --out "$OUT"
  done
done

echo "Concluido. Tabela: $OUT/results.csv"
