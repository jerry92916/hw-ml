import os
import urllib.request
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 1. 處理資料集與字元 Tokenizer 
if not os.path.exists('input.txt'):
    print("下載資料集中...")
    # 卡帕西（Karpathy）經典的 names.txt 英文名字資料集
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

with open('input.txt', 'r', encoding='utf-8') as f:
    docs = [l.strip() for l in f.read().strip().split('\n') if l.strip()]

random.shuffle(docs)
uchars = sorted(set(''.join(docs))) # 撈出所有不重複的字元
BOS = len(uchars)                  # 用最後一個 index 當作起始與結束符號 <BOS>
vocab_size = len(uchars) + 1        # 加上 <BOS> 後的總詞彙量

stoi = {ch: i for i, ch in enumerate(uchars)}
itos = {i: ch for i, ch in enumerate(uchars)}
itos[BOS] = '<BOS>'

# 2. 模型超參數（可以通通在這邊調爐溫）
n_embd = 32          # 每個 Token 映射出來的向量維度
n_head = 4           # 多頭注意力機制的頭數
n_layer = 2          # 疊兩層 Transformer Block
block_size = 16      # 上下文窗口長度，模型一次最多看 16 個字元
dropout = 0.1        # 隨機丟棄 10% 的權重，省得模型死背資料
head_dim = n_embd // n_head
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# 3. 把名字包裝成 PyTorch Dataset
class NameDataset(Dataset):
    def __init__(self, docs):
        self.docs = docs
    
    def __len__(self):
        return len(self.docs)
        
    def __getitem__(self, idx):
        doc = self.docs[idx]
        # 前後都塞 <BOS>，格式變成：<BOS> + 'a' + 'l' + 'e' + 'x' + <BOS>
        tokens = [BOS] + [stoi[c] for c in doc] + [BOS]
        
        # 萬一名字太長超出 block_size，就切掉尾巴，並確保最後一個字是 <BOS>
        if len(tokens) > block_size + 1:
            tokens = tokens[:block_size+1]
            tokens[-1] = BOS 
            
        # x 是輸入序列（去掉最後一個），y 是預報答案（往後移一格，也就是 x 的下一個字）
        x_seq = tokens[:-1]
        y_seq = tokens[1:]
        
        # 長度不夠的話就用 <BOS> 補滿 x，y 的部分用 -1 補，之後算 Loss 會直接忽略 -1
        if len(x_seq) < block_size:
            pad_len = block_size - len(x_seq)
            x_seq = x_seq + [BOS] * pad_len
            y_seq = y_seq + [-1] * pad_len 
            
        return torch.tensor(x_seq, dtype=torch.long), torch.tensor(y_seq, dtype=torch.long)

# 4. 模型主體架構 (微型 GPT Transformer)
class RMSNorm(nn.Module):
    # 比 LayerNorm 更輕量好用的 RMSNorm，Llama 也是用這個
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.scale * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5))

class CausalSelfAttention(nn.Module):
    # 因果自注意力機制（因果指的是：只能看過去，不能偷看未來的答案）
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        # 切成多頭 (Multi-Head) 模式，方便平行運算
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)
        
        # 用 PyTorch 內建的高效能快取計算，is_causal=True 會自動幫我們幫未來資料打上 Mask
        drop_p = dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_p)
        # 把多頭的結果重新接回原本的維度尺寸
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)

class MLP(nn.Module):
    # Transformer 裡面的前饋網路，通常中間會放大 4 倍
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))

class Block(nn.Module):
    # 標準的 Transformer 解碼層，採用 Pre-Norm 結構加上殘差連線（Residual）
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = RMSNorm(n_embd)
        self.mlp = MLP()
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)  # Token 嵌入層
        self.wpe = nn.Embedding(block_size, n_embd)  # 位置編碼（Position Embedding）
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.apply(self._init_weights) # 依據 GPT 慣例初始化權重
        self.lm_head.weight = self.wte.weight # 經典的權重共享（Weight Tying）省記憶體又好練
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # 字元向量 + 位置向量 = 餵給模型的特徵
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x) 
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # 算 CrossEntropy，自動跳過剛才那些補 -1 的 Padding 欄位
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

# 5. 訓練流程：要跑 20,000 步
model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# 【修改】總共跑 20000 step，搭配餘弦退火（Cosine Annealing）動態調降學習率
max_steps = 20000
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-4)

print(f"訓練設備: {device}")
print(f"模型參數總量: {sum(p.numel() for p in model.parameters())}")

dataset = NameDataset(docs)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

# 寫個產生器讓 Dataloader 可以無限循環拋出 Batch
def get_infinite_batches(dl):
    while True:
        for x, y in dl:
            yield x, y
train_iter = iter(get_infinite_batches(dataloader))

model.train()
print(f"開始訓練，預計執行 {max_steps} 步...")

for step in range(max_steps):
    xb, yb = next(train_iter)
    xb, yb = xb.to(device), yb.to(device)
    
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True) # 效能小技巧：改 None 比歸 0 快
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪，防止手滑大爆炸
    optimizer.step()
    scheduler.step()
    
    if step % 1000 == 0 or step == max_steps - 1:
        print(f"步驟 {step:5d} / {max_steps} | 損失值 Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")
        
    # 【新增】每 10000 步就存一次權重存檔點，買個保險
    if step > 0 and step % 10000 == 0:
        checkpoint_name = f'microgpt_step_{step}.pth'
        torch.save(model.state_dict(), checkpoint_name)
        print(f"💾 進度已備份至 {checkpoint_name}")

torch.save(model.state_dict(), 'microgpt_names_weights_final.pth')
print("\n🎉 訓練完全結束！最終模型權重已儲存至 'microgpt_names_weights_final.pth'")

# 6. 讓訓練好的模型出來秀一下（Inference 生成名字）
model.eval()
print("\n--- 模型生成的英文名字 ---")
temperature = 0.8 # 溫度設 0.8 稍微來點隨機與創造力

with torch.no_grad():
    for i in range(20):
        # 每次都用一個 <BOS> 作為開頭讓模型通靈
        idx = torch.tensor([[BOS]], dtype=torch.long).to(device)
        generated_name = ""
        max_gen_length = 30 
        
        for _ in range(max_gen_length):
            # 超過 block_size 就把舊的字切掉，只維持最後 16 個字的 context window
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            
            logits, _ = model(idx_cond)
            # 取最後一個時間點的輸出機率，除以溫度調整軟硬度
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # 依機率進行多項式抽樣（Multinomial），決定下一個字
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 如果模型吐出 <BOS>，代表它覺得這名字結束了，收工
            if idx_next.item() == BOS:
                break 
                
            generated_name += itos[idx_next.item()]
            idx = torch.cat((idx, idx_next), dim=1) # 把新字貼在後面，下一輪繼續預測
                
        if generated_name:
            print(f"sample {i+1:>2d}: {generated_name.capitalize()}")