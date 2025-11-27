# TRepLiNa: Layer-wise CKA + REPINA Alignment for Low-Resource Machine Translation

This repository contains our models and training scripts for **low-resource machine translation (MT)** using **Aya-23 8B** with QLoRA, CKA, and REPINA alignment.  
We focus on translations for **Bhili→Hindi**,**Gondi→Hindi**, **Santali→English**, and **Mundari→Hindi**.

---

## 📖 Description

<img width="1780" height="803" alt="cka_repina_diagram_3_cropped" src="https://github.com/user-attachments/assets/ed947100-7cbc-463f-857f-8ff5e86c7599" />


Low-resource languages often suffer from poor translation quality due to data scarcity and script mismatches.  
TRepLiNa improves MT by aligning intermediate layers of Aya-23 8B through:

- **CKA (Centered Kernel Alignment):** encourages representational similarity between source and target.
- **REPINA:** stabilizes representations by preventing drift on high-resource languages.
- **Layer-wise intervention:** applying alignment at selective layers improves cross-lingual transfer.

---

## 🚀 Models on Hugging Face

You can load our fine-tuned adapters directly:

- **Santali → English**
  - [TRepLiNa (Layer 15)](https://huggingface.co/tona3738/aya23-8b-qlora-cka-repina-santali-english-mmloso-l15-cka001)  
  - [REPINA only (Layer 15)](https://huggingface.co/tona3738/aya23-8b-qlora-cka-repina-santali-english-mmloso-l15-only-repina)  
  - [Baseline (NoAlign)](https://huggingface.co/tona3738/aya23-8b-qlora-cka-repina-santali-english-mmloso-base)

- **Bhili → Hindi**
  - [TRepLiNa (Layer 15)](https://huggingface.co/tona3738/aya23-8b-qlora-cka-repina-bhili-hindi-mmloso-l15-cka001/settings)  
  - [REPINA only (Layer 15)](https://huggingface.co/tona3738/aya23-8b-qlora-cka-repina-bhili-hindi-mmloso-l15-only-repina)  
  - [Baseline (NoAlign)](https://huggingface.co/tona3738/aya23-8b-qlora-cka-repina-bhili-hindi-mmloso-base)

- **Mundari → Hindi**
  - [TRepLiNa (Layer 15)](https://huggingface.co/tona3738/aya23-8b-qlora-cka-repina-mundari-hindi-mmloso-l15-cka001)  
  - [REPINA only (Layer 15)](https://huggingface.co/tona3738/aya23-8b-qlora-cka-repina-mundari-hindi-mmloso-repina-l15)
  - [Baseline (NoAlign)](https://huggingface.co/tona3738/aya23-8b-qlora-cka-repina-mundari-hindi-mmloso-base)  

---

## 📊 Results

### BLEU and ChrF Scores

| Language Pair        | Method        | BLEU ↑ | ChrF ↑ |
|----------------------|--------------|-------- |----------|
| Santali → English    | NoAlign      | 24.26   | 43.96    |
|                      | REPINA-only  | 24.64   | 43.74    |
|                      | TRepLiNa     | **25.24**| **44.68**  |
|                      | Zero-shot    | 0.04    | 3.40     |
|                      | Few-shot(1)  | 0.04    | 4.39     |
|                      | Few-shot(3)  | 0.03    | 2.60     |
|                      | Few-shot(5)  | 0.05    | 2.83     |
| Bhili → Hindi        | NoAlign      | 40.13   | 59.84    |
|                      | REPINA-only  | **40.26** | **59.65**  |
|                      | TRepLiNa     | 40.15   | 59.67    |
|                      | Zero-shot    | 0.88    | 10.57    |
|                      | Few-shot(1)  | 0.64    | 10.40    |
|                      | Few-shot(3)  | 0.93    | 10.72    |
|                      | Few-shot(5)  | 0.35    | 9.38     |
| Mundari → English    | NoAlign      | 24.93  | 46.00     |
|                      | REPINA-only  | 25.08  | 46.02     |
|                      | TRepLiNa     | *25.94* | *46.68*  |

---

## 🧩 Usage

Example: load a fine-tuned adapter with PEFT:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("CohereLabs/aya-23-8B")
model = PeftModel.from_pretrained(
    base_model,
    "tona3738/aya23-8b-qlora-cka-repina-mundari-hindi-mmloso-repina-l15"
)
```
---

## 🛠️ Training

You can reproduce training with:
```python
layer=15
echo "Running with align_layer=${layer}"
python aya_qlora_cka_repina.py \
  --data_csv ./mmloso2025/mundari-train.csv \
  --src_col Mundari --tgt_col Hindi \
  --lang_a_name "Mundari" --lang_b_name "Hindi" \
  --max_source_len 256 --max_target_len 256 \
  --align_layer $layer \
  --lambda_cka 0.01 --mu_repina 0.05 \
  --epochs 5 --batch_size 1 --grad_accum 16 \
  --lr 2e-4 --warmup_ratio 0.05 \
  --bf16 \
  --output_dir ./mundari-hindi-mmloso-l${layer}-cka001 \
  --prefix mundari-hindi-mmloso-l${layer}-cka001
```
---
## 🚀 Train your model using pip

You can fine-tune **any Hugging Face model** with **treplina**, using **any dataset** and **any task**.  
It is *not* limited to machine translation or Aya-23; it works for classification, dialogue, summarization, alignment, etc.

📌 **Colab Example:**  
https://colab.research.google.com/drive/1mgQ8ewAuoVZJWL_z_-1BfUR0sOkxUjAw?usp=sharing

📦 **Install treplina:**

```bash
pip install treplina==0.1.0
```
**Example Training Script:**
```python
# ====== CONFIG ======
layer=15
prefix="mundari-hindi-mmloso-l${layer}-cka001"

echo "Running treplina with prefix=${prefix}  layer=${layer}"

# ====== TRAIN ======
treplina-train \
    --data_csv /content/cka-repina-aya23/mundari-toy.csv \
    --src_col Mundari \
    --tgt_col Hindi \
    --lang_a_name Mundari \
    --lang_b_name Hindi \
    \
    --model_id facebook/xglm-564M \
    --prefix ${prefix} \
    --output_dir ./output-${prefix} \
    \
    --align_layer ${layer} \
    --lambda_cka 0.01 \
    --mu_repina 0.05 \
    \
    --epochs 1 \
    --batch_size 1 \
    --grad_accum 16 \
    \
    --bf16
```
---
## 📄 Citation

If you use TRepLiNa in your research, please cite our paper (to appear):
```
@inproceedings{nakai2025treplina,
  title={Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B},
  author={Nakai, Toshiki and Chikkala, Ravi and Oberkircher, Lena Sophie and Jennings, Nicholas and ...},
  booktitle={Proceedings of the ACL 2025},
  year={2025}
}
```
---
## ✨ Acknowledgments

We thank Saarland University and DFKI for their support.

Also special thanks to Mina Abarico for designing our beautiful architecture diagram!


---

## Questions
For any questions, contact one of our authors:

toshiki3738@gmail.com

rach00004@teams.uni-saarland.de

lenaoberkircher@gmail.com

s8nijenn@stud.uni-saarland.de

natalia.skachkova@dfki.de

tatiana.anikina@dfki.de

jalabi@cs.uni-saarland.de
