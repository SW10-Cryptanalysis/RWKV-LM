import json
import os
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from config import cfg
from model import get_model

# Setup logging to match your style
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rwkv_eval")

def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Locate and Load RWKV Model
    # Note: Using the .pth naming convention from your training script
    model_path = "rwkv7_cipher_final.pth" 
    if not os.path.exists(model_path):
        # Fallback to check step-based checkpoints if final isn't there
        logger.warning(f"Final model not found at {model_path}, check directory.")
        return

    model = get_model() # This already moves to CUDA and float32
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Tokenization Setup (Matches Mistral logic)
    sep_token = cfg.sep_token_id
    char_offset = cfg.char_offset
    chars = "abcdefghijklmnopqrstuvwxyz "
    id_to_char = {i + char_offset: char for i, char in enumerate(chars)}

    # 3. Load Test Files
    # Using the pathing logic from your RWKV training setup
    test_dir = cfg.tokenized_test_dir 
    test_files = list(test_dir.glob("*.json"))[:10]

    if not test_files:
        logger.warning(f"No test files found in: {test_dir}")
        return

    for file_path in test_files:
        with open(file_path) as f:
            data = json.load(f)

        cipher_ids = [int(x) for x in data["ciphertext"].split()]
        true_plain = data["plaintext"]

        # Enforce context limits
        if len(cipher_ids) > (cfg.sequence_length // 2):
            cipher_ids = cipher_ids[:(cfg.sequence_length // 2)]
            true_plain = true_plain[:(cfg.sequence_length // 2)]

        # Prepare initial input: [Cipher] + [SEP]
        input_ids = cipher_ids + [sep_token]
        current_tokens = torch.tensor([input_ids], dtype=torch.long).to(device)

        # 4. Autoregressive Generation Loop
        # RWKV-7 predicts one token at a time
        generated_ids = []
        
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for _ in range(len(cipher_ids)):
                # Forward pass
                logits = model(current_tokens)
                
                # Get the last token's logits
                next_token_logits = logits[:, -1, :]
                
                # Greedy Decoding (equivalent to do_sample=False)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_ids.append(next_token.item())
                
                # Append to input for next step
                current_tokens = torch.cat([current_tokens, next_token], dim=1)

        # 5. Decode
        pred_plain = "".join([id_to_char.get(idx, "?") for idx in generated_ids])

        # 6. Calculate SER (Symbol Error Rate)
        errors = sum(t != p for t, p in zip(true_plain, pred_plain))
        # Add penalty for length mismatch (though here they are forced equal)
        errors += abs(len(true_plain) - len(pred_plain))
        ser = errors / max(len(true_plain), 1)

        logger.info(f"--- File: {file_path.name} ---")
        logger.info(f"True Plain: {true_plain[:50]}...")
        logger.info(f"Pred Plain: {pred_plain[:50]}...")
        logger.info(f"SER: {ser:.4f}")
        logger.info("-" * 30)

if __name__ == "__main__":
    evaluate()