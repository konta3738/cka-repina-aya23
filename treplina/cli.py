# treplina/cli.py
import os
from huggingface_hub import login
from .trainer import build_argparser, train, is_main_process

def main():
    """
    Console entry point for TRepLiNa training.
    HF_TOKEN がある場合のみ login() を行う。
    """
    hf_token = os.environ.get("HF_TOKEN", None)

    # main process のみ HF login
    if is_main_process() and hf_token:
        print("[treplina] Using HF_TOKEN from environment.")
        login(token=hf_token)
    else:
        # 何もしない：HF login は完全に任意
        pass

    parser = build_argparser()
    args = parser.parse_args()
    train(args)
