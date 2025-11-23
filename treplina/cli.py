import os
from huggingface_hub import login
from .trainer import build_argparser, train, is_main_process

def main():
    """
    Console entry point for training Aya-23 with CKA + REPINA.
    """
    hf_token = os.environ.get("HF_TOKEN", None)

    if is_main_process():
        if hf_token:
            login(hf_token)
        else:
            # will prompt user to paste token or use HF_HOME/token cache
            login()

    parser = build_argparser()
    args = parser.parse_args()
    train(args)