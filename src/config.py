import pathlib

import torch
from transformers import BitsAndBytesConfig

BNB_CONFIG = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.float16)
DATA_PATH = pathlib.Path(__file__).parent.parent/"data"
MODEL_TRAINED = pathlib.Path(__file__).parent.parent/"model_trained"
MODEL_GGUF = pathlib.Path(__file__).parent.parent/"model_gguf"
MODEL_HUGGINGFACE = pathlib.Path(__file__).parent.parent/"model_huggingface"
