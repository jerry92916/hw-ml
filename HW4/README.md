# MicroGPT-Names: 基於字符級 Transformer 的英文名字生成器

這是我親手打造的一個輕量級、純 PyTorch 實現的 **Causal Transformer (GPT) 語言模型**。本專案的目標是透過一個「字元級 (Character-level) 語言模型」，讓機器學習英文字母的排列組合規律，並自主生成符合英文語感的全新名字。

本專案從底層的數據預處理、資料載入器 (DataLoader)、自注意力機制 (Self-Attention) 到整體的 GPT 解碼器架構（Decoder-only），全部手工實作，不依賴任何高階的 Hugging Face 庫。

---

## 🚀 專案特點

* **純粹的 GPT 架構**：採用了當前主流的大語言模型（如 LLaMA、GPT-4）的核心設計思維，包括 Causal Self-Attention、RMSNorm 與權重共享（Weight Tying）。
* **字元級 Tokenizer**：直接將 26 個英文字母與特殊標記（如 `<BOS>`）對應到 Token ID，模型不需要處理龐大的詞彙表，專注於學習字母間的「拼音概率」。
* **高效訓練流程**：整合了 `CosineAnnealingLR` 學習率排程器、`Gradient Clipping` 防梯度爆炸，並支援動態進度備份（Checkpoint）。

---

## 🏗️ 模型架構與技術細節

本模型參考了現代 LLM 的最佳實踐，以下是核心組件與參數的詳細配置：

### 1. 超參數設定 (Hyperparameters)
* **上下文視窗 (Context Window)**：16 字元 (`block_size = 16`)
* **嵌入維度 (Embedding Dimension)**：32 (`n_embd = 32`)
* **注意力頭數 (Attention Heads)**：4 頭 (`n_head = 4`)，每個頭的維度為 8
* **網路層數 (Transformer Layers)**：2 層 (`n_layer = 2`)
* **正規化與防止過擬合**：Dropout 率設定為 0.1

### 2. 核心模組設計
* **Token 與位置嵌入 (WTE & WPE)**：結合了字元嵌入與絕對位置嵌入，並加入 Dropout 層防止模型死記硬背。
* **預正規化架構 (Pre-RMSNorm)**：在自注意力與 Feed-Forward 網路之前使用 `RMSNorm` 進行正規化，這比傳統的 LayerNorm 計算更流暢，有助於穩定深層網路。
* **因果自注意力 (Causal Self-Attention)**：使用 PyTorch 內建的高效 `F.scaled_dot_product_attention`，並開啟 `is_causal=True` 施加下三角遮罩（Mask），確保模型在預測下一個字母時，只能看見「過去」的字元。
* **權重共享 (Weight Tying)**：將輸入層的 `Embedding` 權重與最後輸出層的 `Linear` 投射層權重綁定，大幅減少模型參數總量，同時提升泛化能力。

---

## 📊 資料集與預處理 (Dataset)

本專案使用著名的 `names.txt` 資料集，包含數萬個真實的英文名字。

1.  **首尾標記**：在每個名字的前後加上 `<BOS>` (Beginning of Sequence) 標記，例如名字 `"anna"` 在處理後會變成 `<BOS> a n n a <BOS>`。這能讓模型學會何時該開始一個名字，以及何時該收尾。
2.  **定長填充 (Padding)**：為了實現 Batch 化的高效並行訓練，短於 16 個字元的序列會使用 `<BOS>` 填充，並在 Cross-Entropy 損失函數中設定 `ignore_index=-1`，確保這些填充位置不會干擾梯度的計算。

---

## 📉 訓練細節與流程

模型預設會自動偵測環境，若有 `CUDA` (NVIDIA GPU) 或 `MPS` (Apple M-series Silicon) 則會開啟硬體加速。

* **優化器**：使用 `AdamW`，搭配權重衰減（Weight Decay = 0.01）抑制過擬合。
* **訓練步數**：總共執行 **20,000 步 (Steps)**，每 1,000 步輸出一次 Loss 值。
* **動態存檔**：每 10,000 步會自動導出一份 `.pth` 權重檔案，防止訓練中斷。
* **排程器**：採用餘弦退火（Cosine Annealing）動態調整學習率，從 `1e-3` 流暢降至 `1e-4`。

---

## 🎲 模型生成推論 (Inference)

訓練完成後，模型會進入評估模式（`model.eval()`），並在 `temperature = 0.8` 的隨機採樣（Multinomial Sampling）下，一次性生成 20 個全新的英文名字。

透過調整 **Temperature（溫度）**，我們可以自由控制生成結果：
* **較低溫度 (< 0.5)**：生成的名字會非常保守，高度接近資料集裡常見的名字。
* **較高溫度 (> 1.0)**：生成的字母組合會更具創意，但過高可能會導致拼不出合理的英文單字。

---

## 🛠️ 如何運行這份作業

1.  確保你的環境中安裝了 `torch`。
2.  直接執行 Python 腳本：
    ```bash
    python microgpt_names.py
    ```
3.  程式會自動下載 `names.txt`，顯示模型參數總量，開始訓練並在最後打印出 20 個全新生成的英文名字。