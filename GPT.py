import tiktoken
import os
import urllib.request
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

file_path="Verdict.txt"
url="https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open (file_path, 'w', encoding='utf-8') as file:
        file.write(text_data)
else:
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.read()

# print(text_data[:99])
# print("\n",text_data[-99:])# ...existing code...
tokenizer = tiktoken.get_encoding("gpt2")
total_charaters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print(f"\nTotal characters: {total_charaters}")
print(f"Total tokens: {total_tokens}")

# add data loader  and split tokens for taining and validation

class GPTDatasetV1(Dataset):
    def __init__(self, token_ids, tokenizer, max_length, stride):
        self.input_ids=[]
        self.target_ids=[]

        #tokenize the entire text

        # token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids)-max_length,stride):
            input_chunk = token_ids[i:i+max_length]        
            target_chunk = token_ids[i+1: i+max_length+1]
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return (
        torch.tensor(self.input_ids[idx], dtype=torch.long),
        torch.tensor(self.target_ids[idx], dtype=torch.long)
    )
    

def create_dataloader_v1(token_ids, batch_size=4, max_length=128, stride=64, shuffle=True,drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(token_ids, tokenizer, max_length, stride)

    dataloader= DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

GPT_CONFIG_124M={
    "vocab_size":50257,
    "context_length":32,
    "emb_dim":768,
    "n_layer":12,
    "n_head":12,
    "dropout":0.1,
    "qkv_bias":False
}

#train/validation split

train_ratio = 0.9
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text_data, allowed_special={"<|endoftext|>"})

split_idx= int(train_ratio* len(token_ids))
train_data = token_ids[:split_idx]

val_data= token_ids[split_idx:]
torch.manual_seed(123)

train_loader= create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader=create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

#sanity check
if total_tokens *(train_ratio)  < GPT_CONFIG_124M["context_length"]:
   print("Warning: Training data may be too small for the given context length.")
if total_tokens *(1-train_ratio) < GPT_CONFIG_124M["context_length"]:
   print("Warning: Validation data may be too small for the given context length.")     

#check data loader
print("Train dataloader")
for x, y in train_loader:
    print(x.shape, y.shape)
   
print("\nValidation dataloader")
for x, y in val_loader:
    print(x.shape, y.shape)


#set tokens
train_tokens =0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens =0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print(f"\nTotal training tokens per epoch: {train_tokens}")
print(f"Total validation tokens per epoch: {val_tokens}\n")

#GPT model implementation will go here

#layer norm class

class Layer_Norm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps=1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x =(x-mean) / torch.sqrt(var+self.eps)

        return self.scale * norm_x + self.shift
    
#GELU activation function

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5* x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x,3))))
    
class Feed_Forward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4* cfg["emb_dim"]),
            GELU(),
            nn.Linear(4* cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)

class Multi_Head_Attention(nn.Module):
    def __init__(self, d_in,d_out, context_length, dropout, n_head, qkv_bias=False):
        super().__init__()
        assert (d_out % n_head ==0), "d_out must be divisible by n_head"

        self.d_out = d_out
        self.n_head = n_head
        self.head_dim = d_out // n_head
         #reduce the projection dim to match desired output dim

        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_out_proj = nn.Linear(d_out, d_out)

        #Liner layer combined output of all heads

        self.dropout = nn.Dropout(dropout)
        mask = torch.ones((context_length, context_length))  # create tensor of ones
        mask = torch.triu(mask, diagonal=1)                 # get upper triangular part with diagonal offset
        self.register_buffer("mask", mask)


    def forward(self, x):

        b, n_tokens, d_in = x.shape

        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)

        keys = keys.view(b, n_tokens, self.n_head, self.head_dim)
        queries = queries.view (b, n_tokens, self.n_head, self.head_dim)   
        values = values.view (b, n_tokens, self.n_head, self.head_dim)

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)

        mask_bool = self.mask.bool()[:n_tokens, :n_tokens]

        attn_scores.masked_fill(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(b, n_tokens, self.d_out)

        context_vec = self.w_out_proj(context_vec)

        return context_vec
    

class TransformerBlock (nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = Multi_Head_Attention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            n_head = cfg["n_head"],
            dropout = cfg["dropout"],
            qkv_bias = cfg["qkv_bias"]
        )

        self.ff = Feed_Forward(cfg)
        self.norm1 = Layer_Norm(cfg["emb_dim"])
        self.norm2 = Layer_Norm(cfg["emb_dim"])
        self.dropout_shortcut = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        shortcut = x
        x= self.norm1(x)
        x= self.att(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut

        #2nd set
        shortcut = x
        x = self.norm2(x)
        x= self.ff(x)
        x= self.dropout_shortcut(x)
        x= x+shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["dropout"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range (cfg["n_layer"])]
        )

        self.final_norm = Layer_Norm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False) 

    def forward(self, input_ids):
        
        batch_size, seq_len = input_ids.shape

        tok_embed = self.tok_emb(input_ids)
        pos_embed = self.pos_emb(torch.arange(seq_len, device=input_ids.device))

        x = tok_embed + pos_embed

        x = self.drop_emb(x)

        x =self.trf_blocks(x)

        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits
    

def generate_text_simple(model, idx, max_new_tokens, context_size):
    
    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        with torch.nograd():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)

        idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

        return idx
    

def calc_loss_batch (input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    logits = model (input_batch)

    loss = nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())

    return loss

def calc_loss_loader (data_loader, model, device, num_batches=None):
    total_loss =0

    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    avg_loss = total_loss / num_batches

    return avg_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(GPT_CONFIG_124M).to(device)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print(f"Initial training loss: {train_loss:.4f}")
print(f"Initial validation loss: {val_loss:.4f}")