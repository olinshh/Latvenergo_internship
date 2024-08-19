import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import time

path = Path("./archive_ok.json")
#note to self: if result of this is gibberish maybe i should use words rather than characters to encode everything?

# jāpārveido dati tādā formātā, kā karpathy
# pārējajam kodam vajadzētu būt tādam pašam
# kad ir gatavs var pamēģināt ar 1 epochu trenēt modeli

with open(path, encoding = 'utf-8') as file:
    data = json.load(file)
    
# print(data[0:5])
# data = data[:50000] # limitēju datus, lai ātrāk trenētos
# remember to revert this ^^^^^ if everything works but only run on external graphics card then because my computer might explode if i try to run it from here

torch.manual_seed(1337)
batch_size = 1000
block_size = 16
max_iters = 606387
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cuda' if torch.cuda().is_available() else 'cpu'
eval_iters = 200
# n_emd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2

print(type(data))

unique_chars = set()

for item in data:
    if 'raksts' in item:
        for char in item['raksts']:
            unique_chars.add(char)


chars = sorted(list(unique_chars))
vocab_size = len(chars)

print(''.join(chars))
print(vocab_size)

encoder = tiktoken.get_encoding("cl100k_base")
chars_text = ''.join(chars)

print("chars text first 10:", chars_text[:10])

stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}
encoded_chars = [stoi[char] for char in chars]
decoded_chars = ''.join(itos[i] for i in encoded_chars)

print("encoded chars: ", encoded_chars)

print("decoded chars: ", decoded_chars)
i = 0
for char in decoded_chars:
    print(f"{i}: decoded character is {char}")
    i += 1

j=0
for char in chars_text:
    print(f"{j}: character is {char}")
    j += 1
    
print(chars[:10])

test_list = ['Ž', 'b', 'c', 'ž', 'z', 'Z', 'a']
test_string = ''.join(test_list)
encoded_test_string = encoder.encode(test_string)
decoded_test_string = encoder.decode(encoded_test_string)

print("encoded test string: ", encoded_test_string)
print("decoded test string: ", decoded_test_string)



raksts_values = [item['raksts'] for item in data if 'raksts' in item]
raksti_string = ''.join(raksts_values) # visus datus parveidoju par skaitliem

char_mapping = {}
if(len(encoded_chars) == len(decoded_chars)):
    for encoded_char, decoded_char in zip(encoded_chars, decoded_chars):
        char_mapping[decoded_char] = encoded_char
else:
    print("the two lists have differen lengths")
    
encoded_to_decoded = {v: k for k, v in char_mapping.items()}

encoded_data = [char_mapping[char] for char in raksti_string if char in char_mapping]


encoded_sentence = [char_mapping[char] for char in "your encoded sentence here"] # test
decoded_sentence = ''.join([encoded_to_decoded[encoded_char] for encoded_char in encoded_sentence])

print("Decoded Sentence: ", decoded_sentence)

encoded_data = torch.tensor(encoded_data, dtype = torch.long).to(device) # converting my list of integers of data to a tensor


n = int(0.9*len(encoded_data))
train_data = encoded_data[:n]
val_data = encoded_data[n:] # train test split


def get_batch(split):
    encoded_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(encoded_data) - block_size, (batch_size,), device=device)
    x = torch.stack([encoded_data[i:i+block_size] for i in ix])
    y = torch.stack([encoded_data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')

print('inputs:')
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

print('--------')

# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()} the target is: {target}")


        
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
    
    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:    
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = 1)
            idx_next = torch.multinomial(probs, num_samples = 1).to(device)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)
logits, loss = m(xb, yb)
print("logits shape: ", logits.shape)
print(loss)

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
idx = torch.zeros((1,1), dtype = torch.long, device=device)

m_generate = m.generate(idx = torch.zeros((1,1), dtype = torch.long, device=device), max_new_tokens = 100)[0].tolist()
print("m_generate: ")
print(''.join([itos[i] for i in m_generate]))

optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

print("--- START TRAINING ---")
t_start = time.time()

prev_val_loss = 1e9
for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        train_loss = losses['train']
        val_loss = losses['val']
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        m_generate = m.generate(idx = torch.zeros((1,1), dtype = torch.long, device = device), max_new_tokens = 50)[0].tolist()
        print(''.join([itos[i] for i in m_generate]))
        
        if val_loss < prev_val_loss:
            torch.save(model.state_dict(), "state_dict.pt")
        prev_val_loss = val_loss
        
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    

# for steps in range(10000):
#     xb, yb = get_batch('train')
#     logits, loss = m(xb, yb)
#     optimizer.zero_grad(set_to_none = True)
#     loss.backward()
#     optimizer.step()
#     if steps % 200 == 0:
#         print(f"Step: {steps}")
#         m_generate = m.generate(idx = torch.zeros((1,1), dtype = torch.long, device = device), max_new_tokens = 500)[0].tolist()
#         print(''.join([itos[i] for i in m_generate]))
#     torch.save(model.state_dict(), "state_dict.pt")

t_end = time.time()
print("--- END TRAINING ---")
print(f"Training time: {(t_end - t_start):.2f}")
    
print(loss.item())

m_generate = m.generate(idx = torch.zeros((1,1), dtype = torch.long, device = device), max_new_tokens = 500)[0].tolist() ## attempt #2 at generating something cohesive
print("m_generate: ")
print(''.join([itos[i] for i in m_generate]))