
import os
import math
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from huggingface_hub import HfApi, login
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from peft import prepare_model_for_kbit_training
import torch.nn.functional as F
import random
from torch.utils.data import Subset
from sacrebleu.metrics import BLEU, CHRF
import pandas as pd
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)

from peft import LoraConfig, get_peft_model

import pandas as pd
from datasets import load_dataset

from datetime import datetime

def write_eval_log(path: str, epoch: int, scores: dict, mmloso_score: float):
    """
    Append one eval line to `path` (created if missing).
    Format:
    YYYY-mm-dd HH:MM:SS  epoch=E  BLEU=..  chrF++=..  mmloso=..  n=..
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{ts}\tepoch={epoch}\t"
        f"BLEU={scores['bleu']:.2f}\tchrF++={scores['chrf']:.2f}\t"
        f"mmloso={mmloso_score:.2f}\tn={scores['n']}\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)

def write_train_log(path: str, msg: str):
    """
    Append one training-progress line to `path` (created if missing).
    Format:
    YYYY-mm-dd HH:MM:SS  <msg>
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{ts}\t{msg}\n")

# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def exists(x):
    return x is not None

def print_once(s: str):
    if int(os.environ.get("RANK", "0")) == 0:
        print(s, flush=True)

# ----------------------------
# CKA (linear) - PyTorch
# ----------------------------

def center_rows(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Center features along the sample dimension (rows).

    x: [N, D]
    mask: [N] boolean or float mask (1 for keep). If provided, compute mean over masked rows.
    """
    if mask is None:
        mean = x.mean(dim=0, keepdim=True)
        return x - mean
    else:
        if mask.dtype != torch.float32:
            mask = mask.float()
        # avoid division by zero
        denom = mask.sum().clamp(min=1.0)
        mean = (x * mask.unsqueeze(1)).sum(dim=0, keepdim=True) / denom
        return x - mean

def linear_cka(x: torch.Tensor, y: torch.Tensor, x_mask: Optional[torch.Tensor] = None, y_mask: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute linear CKA between two representation matrices.

    x: [N, D]
    y: [M, D]  (we'll downsample/upsample to min(N,M) by truncation to align tokens if not equal)
    masks: optional [N] and [M] masks
    """
    # If different number of rows (tokens), align by truncation to min length.
    n = x.shape[0]
    m = y.shape[0]
    L = min(n, m)
    x = x[:L]
    y = y[:L]
    if x_mask is not None:
        x_mask = x_mask[:L]
    if y_mask is not None:
        y_mask = y_mask[:L]

    # Center rows
    x = center_rows(x, x_mask)
    y = center_rows(y, y_mask)

    # Following Kornblith et al. (2019): CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    # where X, Y are row-centered.
    xty = x.T @ y
    num = (xty * xty).sum()

    xtx = x.T @ x
    yty = y.T @ y
    denom = torch.sqrt((xtx * xtx).sum().clamp(min=eps)) * torch.sqrt((yty * yty).sum().clamp(min=eps))
    return (num / denom).clamp(min=0.0, max=1.0)

# ----------------------------
# Data
# ----------------------------

class ParallelCSV(Dataset):
    """
    A simple dataset for a single parallel pair (A<->B) stored in a CSV.
    The CSV must contain two columns: src, tgt (override with args).
    We assume each row is a parallel pair.

    We'll produce three tokenizations per sample:
      - lm_inputs: prompt (A->B) + target (B), with labels on target only
      - a_align_inputs: "<src>{A}</src>"  (source-only for alignment)
      - b_align_inputs: "<src>{B}</src>"  (source-only for alignment)
    """
    def __init__(self, csv_path: str, src_col: str, tgt_col: str, tokenizer, max_source_len: int = 256, max_target_len: int = 256, lang_a_name: str = "LRL", lang_b_name: str = "Pivot"):
        super().__init__()
        if csv_path.endswith(".csv"):
            df = pd.read_csv(csv_path)
        else:
            # allow arrow/parquet too via HF datasets
            ds = load_dataset("csv", data_files=csv_path)["train"]
            df = pd.DataFrame(ds)
        if src_col not in df.columns or tgt_col not in df.columns:
            raise ValueError(f"Expected columns '{src_col}' and '{tgt_col}' in {csv_path}. Found: {list(df.columns)}")
        self.src_texts = df[src_col].astype(str).tolist()
        self.tgt_texts = df[tgt_col].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.lang_a_name = lang_a_name
        self.lang_b_name = lang_b_name

    def __len__(self):
        return len(self.src_texts)

    def build_prompt(self, src, lang_b):
        return f"Translate to {lang_b}:\n{src}\n"

    def __getitem__(self, idx):
        src = self.src_texts[idx]
        tgt = self.tgt_texts[idx]

        prompt = self.build_prompt(src, self.lang_b_name)
        # Tokenize prompt alone to compute where target starts
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)
        # Tokenize target with EOS
        tgt_ids = self.tokenizer(tgt + self.tokenizer.eos_token, add_special_tokens=False)

        # Truncate
        prompt_ids_input_ids = prompt_ids["input_ids"][-self.max_source_len:]
        tgt_ids_input_ids = tgt_ids["input_ids"][:self.max_target_len]

        # Build full LM input
        input_ids = prompt_ids_input_ids + tgt_ids_input_ids
        labels = [-100] * len(prompt_ids_input_ids) + tgt_ids_input_ids

        # Alignment inputs
        a_align_text = f"{src}"
        b_align_text = f"{tgt}"  # using parallel target text to align source regions across languages
        a_tok = self.tokenizer(a_align_text, add_special_tokens=False, truncation=True, max_length=self.max_source_len)
        b_tok = self.tokenizer(b_align_text, add_special_tokens=False, truncation=True, max_length=self.max_target_len)

        # Build source token masks for alignment (tokens between <src> and </src>).
        # We detect by finding the token ids for the special markers.
        src_start_id = self.tokenizer.convert_tokens_to_ids("<src>")
        src_end_id   = self.tokenizer.convert_tokens_to_ids("</src>")
        def build_src_mask(ids: List[int]):
            mask = torch.zeros(len(ids), dtype=torch.bool)
            # find first <src> and </src>
            try:
                i0 = ids.index(src_start_id)
                i1 = ids.index(src_end_id)
                if i1 > i0 + 1:
                    mask[i0+1:i1] = True
            except ValueError:
                # if markers not found, fall back to all tokens
                mask[:] = True
            return mask

        a_ids = a_tok["input_ids"]
        b_ids = b_tok["input_ids"]
        a_mask = build_src_mask(a_ids)
        b_mask = build_src_mask(b_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "a_align_ids": torch.tensor(a_ids, dtype=torch.long),
            "b_align_ids": torch.tensor(b_ids, dtype=torch.long),
            "a_align_mask": a_mask,
            "b_align_mask": b_mask
        }

@dataclass
class Collator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, batch: List[Dict]):
        # pad input_ids and labels (left padding preferred for causal)
        input_ids = [item["input_ids"] for item in batch]
        labels    = [item["labels"]    for item in batch]

        def pad_1d(seqs, pad_id):
            maxlen = max([len(s) for s in seqs])
            if self.pad_to_multiple_of:
                # ceil to multiple
                m = self.pad_to_multiple_of
                maxlen = ( (maxlen + m - 1) // m ) * m
            out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, -len(s):] = s  # left pad
            return out

        pad_token_id = self.tokenizer.pad_token_id
        input_ids = pad_1d(input_ids, pad_token_id)
        labels    = pad_1d(labels,   -100)

        # Alignment ids/masks: pad right (no need for left-pad here)
        def pad_right_1d(seqs, pad_id):
            maxlen = max([len(s) for s in seqs])
            out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, :len(s)] = s
            return out

        a_ids = pad_right_1d([b["a_align_ids"] for b in batch], self.tokenizer.pad_token_id)
        b_ids = pad_right_1d([b["b_align_ids"] for b in batch], self.tokenizer.pad_token_id)

        # masks: pad right with 0
        def pad_right_mask(msks):
            maxlen = max([len(m) for m in msks])
            out = torch.zeros((len(msks), maxlen), dtype=torch.bool)
            for i, m in enumerate(msks):
                out[i, :len(m)] = m
            return out

        a_mask = pad_right_mask([b["a_align_mask"] for b in batch])
        b_mask = pad_right_mask([b["b_align_mask"] for b in batch])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "a_align_ids": a_ids,
            "b_align_ids": b_ids,
            "a_align_mask": a_mask,
            "b_align_mask": b_mask
        }

# ----------------------------
# Training
# ----------------------------

def get_model_and_tokenizer(model_id: str, r: int, alpha: float, dropout: float, target_modules: Optional[List[str]], bf16: bool):
    print_once(f"Loading model: {model_id}")

    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # the code below is for <src> tags
    # add special markers for source span detection
    #special_tokens = {"additional_special_tokens": ["<src>", "</src>"]}
    #num_added = tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    #the code below is for <src> tags
    #if num_added > 0:
    #    model.resize_token_embeddings(len(tokenizer))

    # gradient checkpointing for memory
    #if hasattr(model, "gradient_checkpointing_enable"):
    #    model.gradient_checkpointing_enable()

    # enable hidden states output by default
    model.config.output_hidden_states = True

    # Prepare QLoRA
    if target_modules is None:
        # defaults for Llama-like blocks
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer

def gather_layer_hidden(hidden_states: Tuple[torch.Tensor, ...], layer_index_1based: int, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    hidden_states is a tuple length (n_layers + 1) with [emb_out, layer1, layer2, ...]
    layer_index_1based = 1 means layer1.
    Return: (B*N, D) flattened reps and (B*N,) mask for non-pad tokens.
    """
    # Select layer
    reps = hidden_states[layer_index_1based]  # [B, T, D]
    # Build token mask from attention_mask (1=keep)
    # left padding so mask is already aligned.
    tok_mask = attention_mask.bool()  # [B, T]
    B, T, D = reps.shape
    reps = reps.reshape(B*T, D)
    tok_mask = tok_mask.reshape(B*T)
    return reps, tok_mask

def forward_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True
    )

def label_smoothed_nll_loss(shift_logits, shift_labels, epsilon=0.1, ignore_index=-100):
    """
    shift_logits: [B, T-1, V]; shift_labels: [B, T-1]
    """
    # mask out positions with ignore_index
    mask = shift_labels.ne(ignore_index)
    if not mask.any():
        return torch.tensor(0.0, device=shift_logits.device)

    # select only valid positions
    logits = shift_logits[mask]            # [N, V]
    targets = shift_labels[mask]           # [N]

    log_probs = F.log_softmax(logits, dim=-1)     # [N, V]
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).mean()
    smooth = -log_probs.mean(dim=-1).mean()       # uniform target
    return (1.0 - epsilon) * nll + epsilon * smooth

def compute_task_loss_with_label_smoothing(outputs, labels, epsilon=0.1, ignore_index=-100):
    # For causal LM we shift so that logits[t] predicts labels[t]
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return label_smoothed_nll_loss(shift_logits, shift_labels, epsilon, ignore_index)

def compute_losses(
    model,
    batch: Dict[str, torch.Tensor],
    tokenizer,
    layer_index: int,
    lambda_cka: float,
    mu_repina: float,
    bf16: bool,
    global_step
):
    """
    Returns (total_loss, dict of components)
    """
    device = next(model.parameters()).device
    repina_loss = torch.tensor(0.0, device=device)

    # 1) Task loss: LM on prompt+target (labels already mask prompt with -100)
    input_ids = batch["input_ids"].to(device)
    labels    = batch["labels"].to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
        output_hidden_states=True
    )
    #task_loss = outputs.loss
    task_loss = compute_task_loss_with_label_smoothing(outputs, labels, epsilon=0.1, ignore_index=-100)

    # 2) Alignment loss (CKA) on source regions of A and B
    a_ids = batch["a_align_ids"].to(device)
    b_ids = batch["b_align_ids"].to(device)
    a_mask = batch["a_align_mask"].to(device)
    b_mask = batch["b_align_mask"].to(device)

    a_attn = (a_ids != tokenizer.pad_token_id).to(device)
    b_attn = (b_ids != tokenizer.pad_token_id).to(device)

    with torch.no_grad():
        # lengths only used for mask sanity
        pass

    a_out = forward_hidden(model, a_ids, a_attn)
    b_out = forward_hidden(model, b_ids, b_attn)

    a_reps_all, a_tokmask_all = gather_layer_hidden(a_out.hidden_states, layer_index, a_attn)
    b_reps_all, b_tokmask_all = gather_layer_hidden(b_out.hidden_states, layer_index, b_attn)

    # Build per-sample masked selection of the <src> region by intersecting with attention_mask and the provided src-span mask
    # We need to expand the src-span masks to flattened (B*T) same as tokmask_all
    '''
    #this code is for <src></src> tags
    def expand_mask(span_mask_2d: torch.Tensor, attn_mask_2d: torch.Tensor):
        # both [B, T]
        keep = (span_mask_2d & attn_mask_2d.bool())  # [B, T]
        return keep.reshape(-1)
    '''
    a_src_keep = a_attn.reshape(-1).bool()
    b_src_keep = b_attn.reshape(-1).bool()
    #a_src_keep = expand_mask(a_mask, a_attn) #for <src> tags
    #b_src_keep = expand_mask(b_mask, b_attn)

    a_src = a_reps_all[a_src_keep]
    b_src = b_reps_all[b_src_keep]

    # Edge case: if no tokens (shouldn't happen), fall back to all non-pad tokens
    if a_src.numel() == 0:
        a_src = a_reps_all[a_tokmask_all]
    if b_src.numel() == 0:
        b_src = b_reps_all[b_tokmask_all]

    # Now compute CKA on the (N,D) matrices.
    cka_val = linear_cka(a_src, b_src, None, None)
    cka_loss = 1.0 - cka_val

    # 3) REPINA^I (identity projection): L2 between base (adapters disabled) and current reps, same inputs as a_align
    do_repina = ((global_step + 1) % 2 == 0)  # every 2 steps
    if do_repina:
        with torch.no_grad():
            with model.disable_adapter():
                base_out = forward_hidden(model, a_ids, a_attn)
        base_reps_all, _ = gather_layer_hidden(base_out.hidden_states, layer_index, a_attn)
        rep_cur = a_reps_all[a_src_keep]
        rep_pre = base_reps_all[a_src_keep]
        if rep_cur.shape[0] != rep_pre.shape[0]:
            L = min(rep_cur.shape[0], rep_pre.shape[0])
            rep_cur = rep_cur[:L]
            rep_pre = rep_pre[:L]

        repina_loss = torch.mean((rep_cur - rep_pre) ** 2)

        total = task_loss + lambda_cka * cka_loss + mu_repina * repina_loss
    else:
        total = task_loss + lambda_cka * cka_loss

    return total, {
        "task_loss": task_loss.detach().float().item(),
        "cka": cka_val.detach().float().item(),
        "cka_loss": cka_loss.detach().float().item(),
        "repina": repina_loss.detach().float().item(),
        "total": total.detach().float().item()
    }

@torch.no_grad()
def evaluate_bleu_chrf_dump(
    model, tokenizer, pairs, lang_b_name, device,
    max_new_tokens=128, num_beams=4, limit=None,
    # chrF params (chrF++ if word_order=2; plain chrF if 0)
    chrf_char_order=6, chrf_word_order=2, chrf_beta=2, chrf_lowercase=False,
    # sentence BLEU smoothing & behavior
    bleu_smooth_method="exp", bleu_smooth_value=None, bleu_effective_order=True,
    # saving
    save_path=None,  # e.g., "./eval_preds_epoch1.csv" or ".jsonl"
    include_ids=False  # set True if your 'pairs' are (idx, src, ref)
):
    """
    pairs: list of (src, ref) or (id, src, ref) if include_ids=True
    Returns:
      {
        "bleu": float, "chrf": float, "n": int,
        "rows": list_of_dicts  # only returned, not saved, if you want quick access
      }
    """
    model.eval()

    bleu_metric  = BLEU(smooth_method=bleu_smooth_method,
                        smooth_value=bleu_smooth_value,
                        effective_order=bleu_effective_order)
    chrf_metric  = CHRF(char_order=chrf_char_order,
                        word_order=chrf_word_order,
                        beta=chrf_beta,
                        lowercase=chrf_lowercase)

    preds, refs = [], []
    rows = []

    for i, item in enumerate(pairs):
        if limit is not None and i >= limit:
            break

        if include_ids:
            ex_id, src, ref = item
        else:
            src, ref = item
            ex_id = i

        prompt = f"Translate to {lang_b_name}:\n{src}\n"
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            length_penalty=1.0,
            early_stopping=True,
        )
        # decode only newly generated tokens
        gen_ids = gen[0][inputs["input_ids"].size(1):]
        hyp = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # collect for corpus scores
        preds.append(hyp)
        refs.append(ref.strip())

        # sentence-level scores
        sent_bleu = bleu_metric.sentence_score(hyp, [ref]).score
        sent_chrf = chrf_metric.sentence_score(hyp, [ref]).score

        rows.append({
            "id": ex_id,
            "src": src,
            "pred": hyp,
            "ref": ref,
            "bleu_sent": sent_bleu,
            "chrf_sent": sent_chrf,
            "len_src": len(src.split()),
            "len_ref": len(ref.split()),
            "len_pred": len(hyp.split())
        })

    # corpus-level
    bleu  = bleu_metric.corpus_score(preds, [refs]).score
    chrf  = chrf_metric.corpus_score(preds, [refs]).score
    n     = len(preds)

    # optional save
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        if save_path.lower().endswith(".csv"):
            pd.DataFrame(rows).to_csv(save_path, index=False)
        else:
            # JSONL by default
            with open(save_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    model.train()
    return {"bleu": bleu, "chrf": chrf, "n": n, "rows": rows}

def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0

def train(args):
    set_seed(args.seed)
    train_log_path = os.path.join(f"./{args.prefix}_preds", f"{args.prefix}_train.log")
    if is_main_process():
        os.makedirs(os.path.dirname(train_log_path) or ".", exist_ok=True)
    # ----------------------------
    # Load model & tokenizer
    # ----------------------------
    model, tokenizer = get_model_and_tokenizer(
        model_id=args.model_id,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules.split(",") if args.target_modules else None,
        bf16=args.bf16
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    """
    This makes the inputs “require grad” so PyTorch checkpointing is happy and actually saves memory. (You’re already passing use_cache=False in forwards, which is correct with checkpointing.)
    """

    # ----------------------------
    # Data
    # ----------------------------
    dataset = ParallelCSV(
        csv_path=args.data_csv,
        src_col=args.src_col,
        tgt_col=args.tgt_col,
        tokenizer=tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        lang_a_name=args.lang_a_name,
        lang_b_name=args.lang_b_name
    )
    collator = Collator(tokenizer=tokenizer, pad_to_multiple_of=8)

    #added for early stopping (eval for each epoch)
    n_total = len(dataset)
    if n_total < 2000:
        n_val = int(0.1 * n_total)
        #n_val = 1 #delete later
    else:
        n_val = min(2000, max(1000, int(0.05 * n_total)))  # ~10% (cap between 1k and 2k)
    indices = list(range(n_total))
    random.Random(args.seed).shuffle(indices)
    val_idx = set(indices[:n_val])
    train_idx = indices[n_val:]

    train_dataset = Subset(dataset, train_idx)
    dev_pairs = [(dataset.src_texts[i], dataset.tgt_texts[i]) for i in val_idx]

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collator
    )
    # ----------------------------
    # Optimizer & Scheduler
    # ----------------------------
    # Use 8-bit Adam if available
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay
        )
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    total_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    device = next(model.parameters()).device

    # ----------------------------
    # Train loop (fixed)
    # ----------------------------
    model.train()

    use_autocast = args.fp16 or args.bf16
    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16

    # New AMP API (silences the deprecation warnings)
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    best_bleu = -1.0
    patience = 2          # stop if no improvement for 2 epochs
    no_improve = 0
    save_best_dir = os.path.join(args.output_dir, "best")

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_autocast):
                loss, metrics = compute_losses(
                    model=model,
                    batch=batch,
                    tokenizer=tokenizer,
                    layer_index=args.align_layer,
                    lambda_cka=args.lambda_cka,
                    mu_repina=args.mu_repina,
                    bf16=args.bf16,
                    global_step = global_step
                )
                loss = loss / args.grad_accum

            # backward (scaled for fp16; normal for bf16/full)
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # do an optimizer step every grad_accum micro-steps
            if (step + 1) % args.grad_accum == 0:
                if args.fp16:
                    scaler.step(optimizer)   # unscales internally if needed
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1
                if global_step % args.log_every == 0:
                    msg = (
                        f"epoch {epoch+1} step {global_step}/{total_steps}: "
                        f"task {metrics['task_loss']:.3f} | cka {metrics['cka']:.3f} "
                        f"(loss {metrics['cka_loss']:.3f}) | repina {metrics['repina']:.3f} "
                        f"| total {metrics['total']:.3f}"
                    )
                    print_once(msg)
                    if int(os.environ.get("RANK", "0")) == 0:
                        write_train_log(train_log_path, msg)

        # --- eval at end of epoch ---
        #bleu = evaluate_bleu(model, tokenizer, dev_pairs, args.lang_b_name, device, max_new_tokens=128, num_beams=4, limit=500)  # limit for speed
        #print_once(f"[eval] epoch {epoch+1} SacreBLEU = {bleu:.2f}")

        scores = evaluate_bleu_chrf_dump(
            model, tokenizer, dev_pairs, args.lang_b_name, device,
            limit=500,
            save_path=os.path.join(f"./{args.prefix}_preds", f"dev_preds_epoch{epoch+1}.csv"),
            include_ids=False
        )
        bleu = scores['bleu']
        mmloso_score = 0.6*scores['bleu']+0.4*scores['chrf']
        print_once(f"[eval] epoch {epoch+1} BLEU={scores['bleu']:.2f} | chrF++={scores['chrf']:.2f} | mmloso_score={mmloso_score:.2f} (n={scores['n']})")
        
        eval_log_path = os.path.join(f"./{args.prefix}_preds", f"{args.prefix}_eval.log")
        if int(os.environ.get("RANK", "0")) == 0:
            write_eval_log(eval_log_path, epoch+1, scores, mmloso_score)
        
        # save-when-best
        if bleu > best_bleu:
            best_bleu = bleu
            no_improve = 0
            if args.output_dir and is_main_process():
                os.makedirs(save_best_dir, exist_ok=True)
                model.save_pretrained(save_best_dir)
                tokenizer.save_pretrained(save_best_dir)
                print_once(f"New best (BLEU {best_bleu:.2f}). Saved to {save_best_dir}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print_once(f"No BLEU improvement for {patience} epoch(s). Early stopping.")
                break
        
        # Save LoRA adapter each epoch
        if args.output_dir and is_main_process():
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch{epoch+1}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print_once(f"Saved LoRA adapter to {save_path}")

    # Final save
    if args.output_dir and is_main_process():
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print_once(f"Saved final LoRA adapter to {args.output_dir}")

        repo_id = f"tona3738/aya23-8b-qlora-cka-repina-{args.prefix}"
        local_dir = args.output_dir

        api = HfApi()
        api.create_repo(repo_id, private=True, exist_ok=True)   # set private=False if you want it public
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            commit_message="Upload LoRA adapter (QLoRA + CKA + REPINA^I)."
        )
        print(f"Pushed to https://huggingface.co/{repo_id}")

def build_argparser():
    p = argparse.ArgumentParser(description="QLoRA fine-tuning of Aya-23 with CKA alignment + REPINA^I drift regularization.")
    # Data
    p.add_argument("--data_csv", type=str, required=True, help="Path to CSV with parallel sentences.")
    p.add_argument("--src_col", type=str, default="src", help="Column name for source language (LRL).")
    p.add_argument("--tgt_col", type=str, default="tgt", help="Column name for pivot language (e.g., Hindi/English).")
    p.add_argument("--lang_a_name", type=str, default="LRL")
    p.add_argument("--lang_b_name", type=str, default="Pivot")
    p.add_argument("--max_source_len", type=int, default=256)
    p.add_argument("--max_target_len", type=int, default=256)

    # Model / LoRA
    p.add_argument("--model_id", type=str, default="CohereLabs/aya-23-8B")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Training
    p.add_argument("--output_dir", type=str, default="./aya23_qlora_cka_repina")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", type=str, required = True)

    # Loss knobs
    p.add_argument("--align_layer", type=int, default=5, help="1-based layer index from input (embedding output is 0).")
    p.add_argument("--lambda_cka", type=float, default=0.05)
    p.add_argument("--mu_repina", type=float, default=0.05)

    # Precision
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 compute for 4-bit.")
    p.add_argument("--fp16", action="store_true", help="Use fp16 autocast for speed.")

    return p

if __name__ == "__main__":
    hf_token = "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"
    if is_main_process():
        login(hf_token)
    args = build_argparser().parse_args()
    train(args)
