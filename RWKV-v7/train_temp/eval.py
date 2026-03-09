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
    for i in range(200): # Predict next 200 tokens
        logits = model(generated)
        # Focus on the very last token's predictions
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)
        
        if next_token.item() == 0: # Stop if it hits a PAD/EOS token
            break

# 4. "Human Readable" Decode
# Since your cipher is likely mapped to specific token IDs:
def simple_decode(tokens):
    # This is a placeholder; replace with your actual tokenizer.decode if available
    return "".join([chr(t % 256) if 32 <= (t % 256) <= 126 else f"[{t}]" for t in tokens])

print(f"\nPROMPT (Cipher):")
print(simple_decode(generated[0, :prompt_length].tolist()))

print(f"\nMODEL OUTPUT (Decryption Attempt):")
print(simple_decode(generated[0, prompt_length:].tolist()))