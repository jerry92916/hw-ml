# 基於深度學習之羽球賽事即時多目標追蹤與運動表現量化分析系統
### Real-Time Multi-Object Tracking and Performance Analytics System in Badminton Using Deep Learning

本專題成功開發出一套結合深度學習電腦視覺、空間幾何變換與運動科學的端到端（End-to-End）自動化分析系統。本系統專為高強度、快速節奏的羽球賽事設計，克服了球員高速位移、肢體劇烈形變及球場視野透視形變等技術挑戰。系統能自動化擷取線上賽事串流，並利用前沿的物件偵測與多目標追蹤演算法精準鎖定並辨識場上球員。透過自主設計的雙足接地定位演算法與幾何特徵工程，系統將傳統非結構化的賽事影像，解構並提煉為具備高戰術與學術價值的結構化運動學數據，為現代運動科技（Sports Tech）應用落地提供科學化的解決方案。

---

## 🛠️ 系統核心架構與技術實現

本系統的運作流程由底層的非結構化影像特徵提取到高階的幾何物理轉換，主要由以下三大核心模組串聯而成：

### 一、 自動化資料串流與深度學習推論管道 (Inference Pipeline)
1. **資料擷取與異步串流 (Data Ingestion)**
   系統建構了自動化賽事資料串流管線，使用者僅需輸入線上賽事影音連結（如 YouTube），系統便會啟用串流下載引擎，自動擷取高格率（High Frame Rate, 如 60 FPS）影片。時序連續性對於運動學分析至關重要，高格率能有效減少選手高速移動產生的殘影（Motion Blur），確保後續特徵提取的精確度。
2. **深度神經網路特徵推論 (Neural Network Inference)**
   核心偵測引擎部署 **YOLOv11** 輕量化卷積神經網路模型，鎖定人類物件特徵（Class 0）進行即時邊界框（Bounding Box）預測。隨後，系統將物件特徵圖動態輸入 **BoT-SORT** 追蹤演算法。BoT-SORT 通過融合卡爾曼濾波（Kalman Filter）的運動預測與匈牙利演算法（Hungarian Algorithm）的全局關聯度匹配，即使在球員交叉換位、被球網或自身肢體嚴重遮擋（Occlusion）的複雜場景下，仍能有效維持特定運動員的唯一 ID 生命週期。系統全管道啟用 GPU CUDA 運算加速，以最大化推論吞吐量。

### 二、 雙足接地定位與幾何空間變換 (Spatial Calibration)
1. **接地點精確動態動態定位 (Foot-point Grounding Localization)**
   傳統視覺分析常採用邊界框中心點（Centroid）作為球員坐標，然而羽球選手在擊球時常有躍起、跨步等大幅度垂直位移，中心點會受到肢體擺動嚴重干擾。為了解決此痛點，本系統自主設計了雙足接地點定位演算法，其幾何邏輯如下：
   * 提取 YOLOv11 輸出的預測矩陣：左上角座標 $(x_1, y_1)$ 與右下角座標 $(x_2, y_2)$。
   * 通過幾何變換，將球員在 3D 空間中與球場地面接觸的動態投影點鎖定為：
     $$Feet_{x} = \frac{x_1 + x_2}{2}, \quad Feet_{y} = y_2$$
2. **像素與真實世界物理尺度映射 (Calibration Factor)**
   由於廣角攝影機存在遠近透視誤差（Perspective Distortion），畫面上各區域的像素間距並不對等。本系統配置了畫面比例尺校正因子（Calibration Factor, $\text{PIXELS\_PER\_METER} = 45.0$），將雙足接地點在影像上的二維像素位移（Pixels），動態映射變換為真實世界的標準物理單位（公尺, Meters），為後續運動學指標奠定精確的數據基準。

### 三、 高階運動學指標特徵工程 (Feature Engineering)
數據提煉模組接收到時序坐標後，會進行一系列的特徵工程計算，將原始坐標轉化為高階運動表現指標：
1. **時序瞬時速度微分 (Velocity Derivative)**
   系統透過計算連續幀之間的歐幾里得距離（Euclidean Distance）來獲取像素位移量（$\Delta d_{px}$），再結合影片的幀率（$FPS$）進行時間微分，最終轉換為真實世界物理速度（$m/s$）：
   $$\Delta d_{px} = \sqrt{(Feet_{x, t} - Feet_{x, t-1})^2 + (Feet_{y, t} - Feet_{y, t-1})^2}$$
   $$Speed_{m/s} = \frac{\Delta d_{px}}{\text{PIXELS\_PER\_METER}} \times FPS$$
2. **防守覆蓋面積拓撲計算 (Court Coverage Area)**
   為了量化選手的跑位範圍，系統引入了幾何拓撲學中的**凸包演算法（Convex Hull）**。此演算法會將選手在整場比賽（或特定回合）中所有雙足接地點的二維空間分佈視為點雲（Point Cloud），並尋找一個能夠包覆所有點點點的最小凸多邊形。通過計算該凸多邊形的內部面積（幾何體積 $V$），再除以比例尺的平方，即可精確產出選手的實際球場防守覆蓋面積（$m^2$）：
   $$\text{Coverage Area } (m^2) = \frac{V_{hull}}{\text{PIXELS\_PER\_METER}^2}$$
3. **量化體能消耗指數 (Stamina Depletion Index)**
   本專題自主研發了一套經驗加權能耗評估模型。考量到羽球運動中「長距離位移」與「瞬間高爆發衝刺」對體能的交叉影響，模型對結構化數據進行特徵加權融合，定義能耗指數（$SDI$）為：
   $$SDI = (\text{Total Distance} \times 0.6) + (\text{Average Speed} \times 0.4)$$
   此指標能有效協助教練團定量評估運動員的體能消耗與疲勞臨界點。

---

## 📊 核心量化成果與視覺化矩陣

系統在推論與特徵工程運算結束後，會自動在本地端儲存高解析度（300 DPI）的統計視覺化圖表與結構化 CSV 數據庫。這些產出可直接作為戰術報告與學術論文的佐證資料：

* **空間活動強度熱力圖 (Player Movement Heatmap)**
  系統利用**二維核密度估計（Kernel Density Estimation, KDE）**技術，動態計算選手在空間中的站位機率密度分佈，並將其色彩矩陣融合疊加於低採樣的場地背景圖上。這項成果能消除多拍時雜亂的線條，直觀呈現球員的「核心控制區域」與站位偏好（如：傾向後場重殺或前場放網），有助於快速洞察對手的戰術盲區與防守破綻。
* **時序動態瞬時速度曲線 (Instantaneous Speed Analytics)**
  以影格時間軸（Frame）為橫導向，將兩位選手的物理移速（$m/s$）時序變動以折線圖呈現。透過此曲線，研究人員能精確抓取選手在接發球時的瞬間爆發啟動點、多拍來回（Rally）時的移動節奏快慢，以及比賽前半段與後半段因體能下降導致的最高衝刺速度下滑趨勢。
* **多指標綜合表現對比長條圖 (Player Performance Comparison)**
  在相同實驗基準下，系統會將特徵工程提煉出的核心數據進行橫向橫向直方圖對比。包含：總奔跑物理距離（Total Distance）、最高衝刺瞬時速度（Maximum Speed）、平均移動速度（Average Speed）、實際防守覆蓋面積（Court Coverage Area）、體能消耗能耗指數（Stamina Index）以及用以評估 AI 追蹤穩定度基準的模型平均置信度（AI Confidence），提供一目了然的運動員表現量化雷達基準。
* **系統推論基準測試報告 (System Inference Benchmark)**
  內建效能評估模組，自動列印目前部署之計算核心平台（如 CUDA (GPU) 或 CPU）與即時處理吞吐量（FPS），充分證實演算法的即時運算實力。同時，系統內建 F1-Score 混淆矩陣交叉檢驗框架，預留與基準真相（Ground Truth）標註資料集比對之接口，確保機器學習專題的學術嚴謹性。

---

## 💻 開發環境與技術棧

* **電腦視覺與幾何拓撲演算法：** OpenCV (v4.x), SciPy (Spatial Topology Engine)
* **深度學習與物件追蹤框架：** Ultralytics YOLOv11 (Neural Engine), BoT-SORT (Tracking Framework)
* **資料處理與統計科學：** Pandas, NumPy
* **高階數據視覺化：** Matplotlib, Seaborn
* **串流下載引擎：** yt-dlp