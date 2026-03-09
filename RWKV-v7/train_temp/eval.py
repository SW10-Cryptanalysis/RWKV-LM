import torch, torch.nn.functional as F
from rwkv7_train_cipher import MODEL, CipherDataset, TOKENIZED_TRAINING_DIR, V, C, T

# 1. Load Model
device = 'cuda'
model = MODEL().to(device).float()
model.load_state_dict(torch.load("rwkv7_cipher_final.pth"))
model.eval()

# 2. Grab a test sample
dataset = CipherDataset(TOKENIZED_TRAINING_DIR)
# Pick a random sample from the end of the dataset (usually unseen data)
input_ids = dataset[len(dataset)-1].unsqueeze(0).to(device)

# 3. Generate
# We take the first 100 tokens as a "prompt" and let the model finish the rest
prompt_length = 100
generated = input_ids[:, :prompt_length]

print(f"--- STARTING EVALUATION ---")
with torch.no_grad():
    print("--- GENERATING ---")
    for _ in range(50):
        T_current = generated.size(1)
        
        # Calculate how much padding we need to reach the next multiple of 16
        pad_needed = (16 - (T_current % 16)) % 16
        
        if pad_needed > 0:
            # Pad with zeros at the end
            model_input = F.pad(generated, (0, pad_needed), "constant", 0)
        else:
            model_input = generated
        
        logits = model(model_input)
        
        # IMPORTANT: We want the logit for the ACTUAL last token (T_current - 1)
        # not the padded tokens.
        next_token = torch.argmax(logits[:, T_current - 1, :], dim=-1).unsqueeze(0)
        
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == 0: break

# 4. "Human Readable" Decode
# Since your cipher is likely mapped to specific token IDs:
def simple_decode(tokens):
    # This is a placeholder; replace with your actual tokenizer.decode if available
    return "".join([chr(t % 256) if 32 <= (t % 256) <= 126 else f"[{t}]" for t in tokens])

print(f"\nPROMPT (Cipher):")
print(simple_decode(generated[0, :prompt_length].tolist()))

print(f"\nMODEL OUTPUT (Decryption Attempt):")
print(simple_decode(generated[0, prompt_length:].tolist()))