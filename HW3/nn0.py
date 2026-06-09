import torch
import torch.nn as nn
import warnings

# 忽略 LazyLinear 的內部形狀初始化警告，讓輸出畫面更乾淨
warnings.filterwarnings("ignore", category=UserWarning)

# 範例一：單層線性回歸 (Linear Regression)
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # 定義一個線性層：y = xW^T + b
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # 定義資料如何流過網路
        return self.linear(x)

# 範例二：多層感知機 (MLP / 二分類任務)
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        # 第一層：輸入層到隱藏層
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 激活函數：引入非線性能力
        self.relu = nn.ReLU()
        # 第二層：隱藏層到輸出層
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # 輸出層激活函數：將結果壓縮到 0~1 之間（代表機率）
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# 範例三：多分類神經網路 (Multi-class Classification)
class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiClassClassifier, self).__init__()
        # 使用 nn.Sequential 可以把結構包在一起，讓 forward 更簡潔
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes) # 注意：PyTorch 的 CrossEntropyLoss 會內建 LogSoftmax
        )
        
    def forward(self, x):
        return self.network(x)

# 範例四：卷積神經網路 (CNN - 影像分類基礎)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 卷積層與池化層
        self.conv_block = nn.Sequential(
            # 輸入 1 個通道(黑白圖片), 輸出 16 個通道, 卷積核 3x3
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 圖片大小減半
        )
        # 【修正！】改用 nn.LazyLinear
        # 這樣一來，不論前一層拉平成多少維度（例如 3136 或 12544），它都會自動對齊，再也不會噴 RuntimeError！
        self.fc = nn.LazyLinear(num_classes)
        
    def forward(self, x):
        x = self.conv_block(x)
        # 將特徵圖拉平成一維向量 (除了 Batch 維度以外)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

# 範例五：循環神經網路 (RNN - 序列資料處理)
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        # 定義 RNN 層，batch_first=True 代表輸入形狀為 [Batch, Seq_len, Feature]
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # 解碼器：將 RNN 的隱藏狀態轉為最終輸出
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # rnn_out 的形狀: [Batch, Seq_len, Hidden_size]
        rnn_out, hn = self.rnn(x)
        
        # 取出序列中「最後一個時間步」的輸出
        last_time_step = rnn_out[:, -1, :]
        
        out = self.fc(last_time_step)
        return out

# 執行測試
if __name__ == "__main__":
    print("==================================================")
    print(" PyTorch nn.Module 5大基礎範例測試開始（已修正形狀問題）")
    print("==================================================\n")

    # --- 範例一測試 ---
    print("--- [範例一] 線性回歸 ---")
    model_1 = LinearRegressionModel(input_dim=3, output_dim=1)
    input_1 = torch.randn(2, 3) # Batch=2, Features=3
    output_1 = model_1(input_1)
    print(f"輸入形狀: {input_1.shape}")
    print(f"輸出形狀: {output_1.shape}")
    print(f"輸出結果:\n{output_1}\n")

    # --- 範例二測試 ---
    print("--- [範例二] 多層感知機 (MLP 二分類) ---")
    model_2 = MultiLayerPerceptron(input_dim=10, hidden_dim=5, output_dim=1)
    input_2 = torch.randn(4, 10) # Batch=4, Features=10
    output_2 = model_2(input_2)
    print(f"輸入形狀: {input_2.shape}")
    print(f"輸出形狀: {output_2.shape}")
    print(f"輸出機率:\n{output_2}\n")

    # --- 範例三測試 ---
    print("--- [範例三] 多分類網路 ---")
    model_3 = MultiClassClassifier(input_dim=8, num_classes=3)
    input_3 = torch.randn(3, 8) # Batch=3, Features=8
    output_3 = model_3(input_3)
    print(f"輸入形狀: {input_3.shape}")
    print(f"輸出形狀: {output_3.shape}")
    print(f"輸出結果:\n{output_3}\n")

    # --- 範例四測試 ---
    print("--- [範例四] 卷積神經網路 (CNN) ---")
    model_4 = SimpleCNN(num_classes=10)
    # 這裡我們模擬圖片輸入，LazyLinear 會自動去適應這張圖片產生的特徵數
    input_4 = torch.randn(2, 1, 28, 28) # Batch=2, C=1, H=28, W=28
    output_4 = model_4(input_4)
    print(f"輸入影像形狀: {input_4.shape}")
    print(f"輸出分類形狀: {output_4.shape}\n")

    # --- 範例五測試 ---
    print("--- [範例五] 循環神經網路 (RNN) ---")
    model_5 = SimpleRNN(input_size=4, hidden_size=8, output_size=1)
    input_5 = torch.randn(2, 5, 4) # Batch=2, Seq_len=5, Features=4
    output_5 = model_5(input_5)
    print(f"輸入序列形狀: {input_5.shape}")
    print(f"輸出預測形狀: {output_5.shape}")
    print(f"輸出結果:\n{output_5}\n")

    print("==================================================")
    print(" 測試全數順利完成！")
    print("==================================================")