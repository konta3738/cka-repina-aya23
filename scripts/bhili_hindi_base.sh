#!/bin/bash
pip install -U transformers accelerate peft datasets bitsandbytes sacrebleu
export HUGGINGFACE_HUB_TOKEN="<YOUR_TOKEN>"


echo "Running with align_layer=base"
python aya_qlora_cka_repina.py \
  --data_csv ./mmloso2025/bhili-train.csv \
  --src_col Bhili --tgt_col Hindi \
  --lang_a_name "Bhili" --lang_b_name "Hindi" \
  --max_source_len 256 --max_target_len 256 \
  --lambda_cka 0 --mu_repina 0 \
  --epochs 5 --batch_size 1 --grad_accum 16 \
  --lr 2e-4 --warmup_ratio 0.05 \
  --bf16 \
  --output_dir ./bhili-hindi-mmloso-base \
  --prefix bhili-hindi-mmloso-base