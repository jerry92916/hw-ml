import requests
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# 1. 從網路上抓測試資料來預處理
github_url = "https://github.com/ccc114b/cccocw/blob/main/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92/07-%E8%AA%9E%E8%A8%80%E6%A8%A1%E5%9E%8B/01-%E5%82%B3%E7%B5%B1%E6%96%B9%E6%B3%95/lm/tw.txt"
# 把 GitHub 網址換成 raw 的格式，這樣 requests 才抓得到純文字
raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

print("正在從網路下載測試文本...")
try:
    response = requests.get(raw_url)
    response.encoding = 'utf-8'
    # 把所有的空白、換行、空格通通濾掉，黏成一大條字串
    text = "".join(response.text.split())
    print(f"成功讀取資料！文本總字數：{len(text)} 字")
except Exception as e:
    # 萬一網路上不去或斷線，就用這段備用的小短文頂著用
    print(f"下載失敗 ({e})，改用內建測試短文本。")
    text = "我們都是好朋友，今天天氣很好，我們一起去公園玩，公園裡面有很多人。好朋友一起玩真開心。"

# 2. 準備馬可夫模型的訓練集（拿前面 2 個字，當作預測下 1 個字的線索）
X = []
y = []
for i in range(len(text) - 2):
    X.append(text[i:i+2])  # 題目：目前的兩個字（狀態）
    y.append(text[i+2])    # 答案：正後方的下一個字

# 3. 用 sklearn Pipeline 把特徵轉換和模型綁在一起
pipeline = Pipeline([
    # 用客製化的 lambda 把字串包成 list，讓 CountVectorizer 乖乖做字元級的 One-Hot 編碼
    ('vectorizer', CountVectorizer(analyzer=lambda x: [x])),
    # 用決策樹來學「看到這兩個字、後面該接什麼字」的機率分佈
    ('classifier', DecisionTreeClassifier(criterion='entropy', random_state=42))
])

print("正在訓練二階馬可夫模型...")
pipeline.fit(X, y)
classes = pipeline.classes_ # 撈出模型認得的所有中文字清單
print("模型訓練完成！\n")

# 4. 互動式預測與接龍模式
print("=" * 60)
print(" 互動預測模式已啟動！")
print(" 提示：請輸入任意中文（模型會自動抓最後兩個字當作馬可夫狀態）")
print("=" * 60)

while True:
    # 撈使用者輸入的文字
    user_input = input("\n請輸入文字（或輸入 q 離開）：").strip()
    
    # 檢查要不要登出
    if user_input.lower() in ['q', 'quit', 'exit']:
        print("退出互動模式，謝謝使用！")
        break
        
    # 字數不夠就沒辦法看前兩個字了，提示一下使用者
    if len(user_input) < 2:
        print("⚠️ 錯誤：請至少輸入兩個字！模型才能幫你預測下一個字喔。")
        continue
        
    # 直接切最後兩個字出來，當作馬可夫鏈目前所在的節點
    context = user_input[-2:]
    print(f"👉 偵測到的預測脈絡為：【{context}】")
    
    try:
        # 1. 預測下一個字出現的機率分佈
        probs = pipeline.predict_proba([context])[0]
        
        # 排序一下，把機率最高的前 3 名撈出來
        top_indices = np.argsort(probs)[::-1][:3]
        
        print("\n【🤖 模型預測最可能的下一個字：】")
        has_prediction = False
        for rank, idx in enumerate(top_indices, 1):
            if probs[idx] > 0:  # 有機率的字才印出來
                print(f"  Top {rank}: '{classes[idx]}' ── 機率: {probs[idx]*100:.2f}%")
                has_prediction = True
                
        if not has_prediction:
            print("  （這個詞彙後面在文本中沒有接續任何字）")
            
        # 2. 自動延伸接龍（依據機率隨機往下通靈 20 個字，比較好玩）
        generated = user_input
        for _ in range(20):
            curr_ctx = generated[-2:] # 每次都看最後剛吐出來的兩個字
            p = pipeline.predict_proba([curr_ctx])[0]
            if np.sum(p) == 0:  # 萬一走到死胡同（資料庫沒這詞），就提早收工
                break
            p = p / np.sum(p)  # 重新校正機率，讓總和等於 1 
            next_char = np.random.choice(classes, p=p) # 依照機率權重抽一個字出來
            generated += next_char
            
        print(f"\n【🔮 AI 自動幫你接龍後續：】\n{generated}")
        print("-" * 40)
        
    except Exception as e:
        # 萬一輸入的這兩個字在訓練文本中從來沒一起出現過，決策樹就會找不到路
        print("❌ 抱歉！這兩個字的組合在訓練資料（tw.txt）中從未出現過，馬可夫模型無法預測。")
        print("💡 提示：可以試試看文本中常出現的字（例如：台灣、我們、發展、經濟...）")
        print("-" * 40)