"""
Quick test: verify text2gesturetype handles multi-sentence text via chunking.
Run from the gesture_generation/ directory:
  conda run -n gesture python test_multisent.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.makedirs(".tmp", exist_ok=True)

import torch
from transformers import BertTokenizer
from gesture_type_prediction.model import GestureTypePredictor
from gesture_type_prediction.text2gesturetype import load_checkpoint, text2gesturetype

MODEL_PATH = "./gesture_type_prediction/model/model_valid_f1_class_augmented_finetune_Linear.pt"
MAX_SEQ_LEN = 256

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cpu")
model = GestureTypePredictor(num_class=3).to(device)
load_checkpoint(MODEL_PATH, model, device)

SHORT = "Please make yourself feel at home. It is great to see you."

LONG = (
    "Hello, it is wonderful to meet you today. "
    "I have been looking forward to our conversation for a long time. "
    "There are many interesting topics we could discuss together. "
    "For example, we could talk about the weather, science, or technology. "
    "I hope you enjoy our time together and feel comfortable asking anything."
)

def check(label, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"\n[{label}]  token count (no special): {len(ids)}, needs chunking: {len(ids)+2 > MAX_SEQ_LEN}")
    tokens, types = text2gesturetype(text, model, tokenizer, device)
    assert len(tokens) == len(types), "token/type length mismatch!"
    # [CLS] at 0, [SEP] at -1
    assert tokens[0] == "[CLS]" and tokens[-1] == "[SEP]", "missing CLS/SEP!"
    content_len = len(tokens) - 2
    print(f"  content tokens: {content_len}  gesture types: {set(int(t) for t in types[1:-1])}")
    print("  PASS")

OVER_LIMIT = (
    "The history of artificial intelligence is a fascinating journey that spans several decades. "
    "In the early days, researchers believed that human-level intelligence could be achieved within a generation. "
    "However, they soon encountered many unexpected challenges that proved far more difficult than originally anticipated. "
    "The development of expert systems in the nineteen eighties was considered a major breakthrough at the time. "
    "These systems were designed to replicate the decision-making ability of a human expert in a specific domain. "
    "Despite their initial success, expert systems eventually became too expensive and difficult to maintain in practice. "
    "The rise of machine learning in the nineteen nineties opened up a completely new set of possibilities for researchers. "
    "Neural networks, inspired by the structure of the human brain, began to show remarkable results on many problems. "
    "Today, deep learning models are capable of performing tasks that were once thought to be exclusively within human reach. "
    "Looking forward, the future of artificial intelligence holds both tremendous promise and significant ethical challenges."
)

check("SHORT (2 sentences)", SHORT)
check("LONG  (5 sentences)", LONG)
check("OVER  (10 sentences, >256 tokens)", OVER_LIMIT)
print("\nAll tests passed.")
