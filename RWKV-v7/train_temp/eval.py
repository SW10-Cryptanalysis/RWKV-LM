import torch
import torch.nn.functional as F
from config import cfg
from model import get_model
from rwkv7_train_cipher import CipherDataset

# --- INITIALIZATION ---
device = 'cuda'
model = get_model()
model.load_state_dict(torch.load("rwkv7_cipher_final.pth"))
model.eval()

# --- LOAD TEST DATA ---
dataset = CipherDataset(cfg.tokenized_training_dir)
# Pick a random sample from the end of the dataset (usually unseen data)
input_ids = dataset[len(dataset) - 1].unsqueeze(0).to(device)

# --- GENERATION ---
# We take the first 100 tokens as a "prompt" and let the model finish the rest
prompt_length = 100
generated = input_ids[:, :prompt_length]

print(f"--- STARTING EVALUATION ---")
with torch.no_grad():
    print("--- GENERATING ---")
    for _ in range(50):
        T_current = generated.size(1)
        
        # Calculate how much padding we need to reach the next multiple of chunk_len
        pad_needed = (cfg.chunk_len - (T_current % cfg.chunk_len)) % cfg.chunk_len
        
        if pad_needed > 0:
            # Pad with zeros at the end
            model_input = F.pad(generated, (0, pad_needed), "constant", cfg.pad_token_id)
        else:
            model_input = generated
        
        logits = model(model_input)
        
        # IMPORTANT: We want the logit for the ACTUAL last token (T_current - 1)
        # not the padded tokens.
        next_token = torch.argmax(logits[:, T_current - 1, :], dim=-1).unsqueeze(0)
        
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == cfg.pad_token_id:
            break

# --- DECODE OUTPUT ---
def sota_decode(tokens):
    char_offset = 2499
    chars = "abcdefghijklmnopqrstuvwxyz "
    # Map indices to characters
    id_to_char = {i + char_offset: char for i, char in enumerate(chars)}
    id_to_char[cfg.space_token_id] = " "
    id_to_char[cfg.sep_token_id] = " | " # Visual separator
    
    res = []
    for t in tokens:
        if t in id_to_char:
            res.append(id_to_char[t])
        elif t == cfg.pad_token_id:
            continue
        else:
            res.append(f"[{t}]")
    return "".join(res)

print(f"\nPROMPT (Cipher):")
# Use the new decoder!
print(sota_decode(generated[0, :prompt_length].tolist()))

print(f"\nMODEL OUTPUT (Decryption Attempt):")
print(sota_decode(generated[0, prompt_length:].tolist()))