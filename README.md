# Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B

[Paper Link (ArXiv/ACL Anthology - TBD)](https://arxiv.org/)  
Official codebase for our paper:

> **Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B**

## Abstract
We study whether enforcing cross-lingual similarity in specific internal layers of a decoder-only multilingual LLM can improve translation quality for low-resource languages (LRLs).  
Our method combines **Centered Kernel Alignment (CKA)** with **REPINA regularization**, applied layer-wise during QLoRA fine-tuning of Aya-23~8B.  
Experiments on the **MMLoSo 2025 Shared Task** (Mundari, Santali, Bhili, Gondari ↔ Hindi/English) show that aligning mid-level layers (≈ layer 15) improves BLEU/chrF in data-scarce regimes.

---

## 📂 Repository Structure
cka-repina-aya23/
│── data/ # Preprocessed MMLoSo splits (not included - see below)
│── scripts/ # Training / evaluation scripts
│── models/ # LoRA adapters or checkpoints (uploaded to HF Hub)
│── results/ # Logs, figures, and tables
│── main.py # Single-script trainer (QLoRA + CKA + REPINA)
│── utils.py # Logging, alignment, tokenization helpers
│── requirements.txt
│── README.md

yaml
Code kopieren

---

## 🚀 Setup

### 1. Environment
```bash
git clone https://github.com/<your-username>/cka-repina-aya23.git
cd cka-repina-aya23
conda create -n cka-repina python=3.10
conda activate cka-repina
pip install -r requirements.txt
Main dependencies:

transformers >= 4.40

peft

datasets

bitsandbytes

sacrebleu

pandas

torch >= 2.2

2. Data
We use the MMLoSo 2025 shared task dataset.
Please download from MMLoSo Shared Task Page.

Expected CSV format:

Code kopieren
src_col, tgt_col
हिंदी वाक्य, भीली अनुवाद
🧪 Training & Evaluation
Train with CKA+REPINA (example: Mundari→Hindi, layer 15)
bash
Code kopieren
python main.py \
  --dataset data/mundari_hindi.csv \
  --src_lang mundari --tgt_lang hindi \
  --layer 15 \
  --lambda_cka 0.05 --mu_repina 0.05 \
  --output_dir results/mundari_hindi
Zero-shot evaluation
bash
Code kopieren
python evaluate.py \
  --dataset data/mundari_hindi.csv \
  --checkpoint models/aya23_zero
Few-shot prompts
We include 1-, 3-, 5-shot prompt templates under prompts/.

📊 Results (Dev set)
Language Pair	Zeroshot	Few-shot (5)	CKA+REPINA (ours)	REPINA-only
Mundari→Hindi	3.54	3.24	34.24	33.45
Santali→English	1.38	1.16	see paper	see paper

See full tables in Section 5 of the paper.

📈 Figures
<p align="center"> <img src="results/mundari_english_exp1.png" width="400"> </p> *Layer sweep: CKA peaks at layer 10; CKA+REPINA peaks at layer 15.*
🤝 Citation
If you use this code, please cite:

bibtex
Code kopieren
@inproceedings{nakai2025cka-repina,
  title={Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B},
  author={First Author and Second Author},
  booktitle={Proceedings of the MMLoSo 2025 Workshop @ IJCNLP-AACL},
  year={2025}
}
⚖️ License
MIT License. See LICENSE file.

🔗 Related Repos
CohereForAI/aya-23

razdaibiedina/REPINA

kornblith/CKA

yaml




