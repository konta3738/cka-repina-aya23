#!/bin/bash
pip install -U transformers accelerate peft datasets bitsandbytes sacrebleu
export HUGGINGFACE_HUB_TOKEN="<YOUR_TOKEN>"

# Mundari → Hindi (both fine at 256)
layer=15
echo "Running with align_layer=${layer}"
python aya_qlora_cka_repina.py \
  --data_csv ./mmloso2025/mundari-train.csv \
  --src_col Mundari --tgt_col Hindi \
  --lang_a_name "Mundari" --lang_b_name "Hindi" \
  --max_source_len 256 --max_target_len 256 \
  --align_layer $layer \
  --lambda_cka 0 --mu_repina 0.05 \
  --epochs 5 --batch_size 1 --grad_accum 16 \
  --lr 2e-4 --warmup_ratio 0.05 \
  --bf16 \
  --output_dir ./mundari-hindi-mmloso-repina-l${layer} \
  --prefix mundari-hindi-mmloso-repina-l${layer}


