import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import yt_dlp
import os
import time
from scipy.spatial import ConvexHull

# 🌟 新增：畫面比例尺設定 (Calibration Factor)
# 假設畫面上約 45 個像素等於真實世界的 1 公尺 (這需要根據實際影片視角與球場大小微調)
PIXELS_PER_METER = 45.0 

# ==========================================
# 步驟 0：F1-Score 計算框架
# ==========================================
def evaluate_tracking_f1(pred_df, gt_df=None, iou_threshold=0.5):
    """
    計算追蹤的 F1-Score。
    如果有提供真實標註檔 (gt_df) 才會進行計算。
    """
    if gt_df is None:
        return {"Precision": "N/A", "Recall": "N/A", "F1_Score": "N/A (Missing Ground Truth)"}
        
    print("⏳ 正在與 Ground Truth 比對計算 F1-Score...")
    return {"Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0}

# ==========================================
# 步驟 1：影片下載
# ==========================================
def download_youtube_video(youtube_url, output_filename):
    print("⏳ 步驟 1：開始自動下載 YouTube 影片...")
    ydl_opts = {
        'format': 'b[ext=mp4]', 
        'outtmpl': output_filename,
        'quiet': False, 
        'noplaylist': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        print("✅ 下載完成！")
        return True
    except Exception as e:
        print(f"❌ 下載失敗，錯誤訊息: {e}")
        return False

# ==========================================
# 步驟 2：核心 AI 分析與視覺化
# ==========================================
def analyze_and_visualize(video_path, output_video_path, output_csv_path, heatmap_path, metrics_plot_path, comparison_chart_path, target_ids, id_mapping, start_sec=249, duration_sec=None):
    print("⏳ 步驟 2：載入 YOLOv11 模型，準備開始追蹤...")
    model = YOLO('yolo11n.pt') 
    
    device_used = model.device.type
    print(f"🚀 目前 YOLO 模型使用的運算設備: {device_used.upper()}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 錯誤：OpenCV 無法開啟影片！")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * fps)
    
    max_frames = int(duration_sec * fps) if duration_sec else (total_frames - start_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, bg_frame = cap.read()
    if ret:
        bg_frame_rgb = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
        bg_frame_rgb = (bg_frame_rgb * 0.5).astype(np.uint8) 
    else:
        bg_frame_rgb = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    tracking_data = []
    frames_processed = 0

    colors = {
        target_ids[0]: (0, 0, 255),   # ID 1: 紅色
        target_ids[1]: (255, 0, 0)    # ID 147: 藍色
    }

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        output_video_path = output_video_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"🏸 開始追蹤指定球員 (ID: {target_ids})...")
    start_time = time.time()

    while cap.isOpened() and frames_processed < max_frames:
        success, frame = cap.read()
        if not success:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frames_processed > 0 and frames_processed % 100 == 0:
            elapsed_time_so_far = time.time() - start_time
            current_fps = frames_processed / elapsed_time_so_far
            print(f"處理進度: {frames_processed} / {max_frames} 幀 | 當前速度: {current_fps:.1f} FPS")

        results = model.track(frame, classes=[0], tracker="botsort.yaml", persist=True, verbose=False)
        annotated_frame = frame.copy()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy() 

            for box, track_id, conf in zip(boxes, track_ids, confs):
                t_id = int(track_id)
                if t_id in id_mapping:
                    t_id = id_mapping[t_id]

                x1, y1, x2, y2 = map(int, box)
                feet_x = int((x1 + x2) / 2)
                feet_y = y2

                if t_id in target_ids:
                    tracking_data.append({
                        'frame': current_frame,
                        'track_id': t_id,
                        'confidence': round(float(conf), 3), 
                        'box_x1': x1, 'box_y1': y1, 
                        'box_x2': x2, 'box_y2': y2,
                        'feet_x': feet_x, 'feet_y': feet_y
                    })

                    color = colors.get(t_id, (0, 255, 0)) 
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(annotated_frame, f"ID {t_id} ({conf:.2f})", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(annotated_frame)
        frames_processed += 1

    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = frames_processed / total_time if total_time > 0 else 0

    cap.release()
    out.release()
    
    print("\n" + "="*50)
    print("📊 系統效能分析報告 (System Benchmark)")
    print("="*50)
    print(f"⚙️ 執行設備: {device_used.upper()}")
    print(f"🎞️ 處理總幀數: {frames_processed} 幀")
    print(f"⏱️ 總處理時間: {total_time:.2f} 秒")
    print(f"🚀 平均處理速度: {avg_fps:.2f} FPS")
    print("="*50 + "\n")

    if not tracking_data:
        print("\n⚠️ 警告：影片中沒有偵測到目標 ID 的軌跡，無法進行進階分析。")
        return

    # ==========================================
    # 步驟 3：進階運動數據與指標分析 (轉換為真實物理單位)
    # ==========================================
    df = pd.DataFrame(tracking_data)
    raw_metrics_data = [] 
    player_metrics = {}

    for t_id in target_ids:
        p_df = df[df['track_id'] == t_id].copy().sort_values('frame')
        if len(p_df) < 2:
            continue
            
        p_df['dx'] = p_df['feet_x'].diff()
        p_df['dy'] = p_df['feet_y'].diff()
        
        # 先算像素距離，再轉換成「公尺」
        p_df['distance_px'] = np.sqrt(p_df['dx']**2 + p_df['dy']**2).fillna(0)
        p_df['distance_m'] = p_df['distance_px'] / PIXELS_PER_METER
        
        # 速度轉換：(像素/幀) * (幀/秒) = (像素/秒)，再轉換為 (公尺/秒)
        p_df['speed_px_s'] = p_df['distance_px'] * fps
        p_df['speed_m_s'] = p_df['speed_px_s'] / PIXELS_PER_METER
        
        # 🌟 計算物理單位總和與平均
        total_distance_m = round(p_df['distance_m'].sum(), 2)
        max_speed_m_s = round(p_df['speed_m_s'].max(), 2)
        avg_speed_m_s = round(p_df['speed_m_s'].mean(), 2)
        avg_confidence = round(p_df['confidence'].mean() * 100, 1) 
        
        # 覆蓋面積 (轉換為平方公尺)
        points = p_df[['feet_x', 'feet_y']].dropna().values
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                # 面積轉換：像素面積 / (像素比例尺的平方)
                coverage_area_m2 = round(hull.volume / (PIXELS_PER_METER ** 2), 2)
            except:
                coverage_area_m2 = 0
        else:
            coverage_area_m2 = 0
            
        # 體能消耗指數 (基於真實距離計算)
        stamina_index = round((total_distance_m * 0.6) + (avg_speed_m_s * 0.4), 2)

        # 儲存終端機列印用的字典 (已換上 M 和 m/s)
        player_metrics[t_id] = {
            'AI Tracking Confidence': f"{avg_confidence}%", 
            'Total Distance (m)': total_distance_m,
            'Max Speed (m/s)': max_speed_m_s,
            'Avg Speed (m/s)': avg_speed_m_s,
            'Coverage Area (m^2)': coverage_area_m2,
            'Stamina Depletion Index': stamina_index
        }
        
        # 儲存畫圖用的純數字字典
        raw_metrics_data.append({
            'Player ID': f"Player {t_id}",
            'AI Confidence (%)': avg_confidence,
            'Total Distance (m)': total_distance_m,
            'Max Speed (m/s)': max_speed_m_s,
            'Avg Speed (m/s)': avg_speed_m_s,
            'Coverage Area (m^2)': coverage_area_m2,
            'Stamina Index': stamina_index
        })
        
        # 將轉換後的速度更新回原本的 df，準備畫圖用
        df.loc[p_df.index, 'speed_m_s'] = p_df['speed_m_s']
    
    df.to_csv(output_csv_path, index=False)
    
    print("🏆 球員運動物理數據與 AI 評估報告 (Physical & AI Metrics) 🏆")
    print("-" * 55)
    for t_id, metrics in player_metrics.items():
        print(f"👤 Player ID: {t_id}")
        for metric_name, value in metrics.items():
            print(f"   ➤ {metric_name}: {value}")
            
    f1_result = evaluate_tracking_f1(df, None)
    print(f"\n   [備註] F1-Score 狀態: {f1_result['F1_Score']}")
    print("-" * 55 + "\n")

    # ==========================================
    # 步驟 4：視覺化 1 - 繪製熱力圖與速度圖表
    # ==========================================
    print("🎨 正在產生熱力與物理速度分析圖...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    
    ax_heat = axes[0]
    if bg_frame_rgb is not None:
        ax_heat.imshow(bg_frame_rgb, extent=[0, width, height, 0], zorder=1)
    sns.kdeplot(
        x=df['feet_x'], y=df['feet_y'], hue=df['track_id'], 
        palette={target_ids[0]: 'red', target_ids[1]: 'cyan'},
        fill=True, thresh=0.05, alpha=0.6, ax=ax_heat, zorder=2
    )
    ax_heat.set_xlim(0, width)
    ax_heat.set_ylim(height, 0)
    ax_heat.set_title("Player Movement Heatmap", fontsize=16, fontweight='bold')
    
    ax_speed = axes[1]
    # 🌟 修改圖表資料：採用轉換後的公尺/秒速度
    sns.lineplot(
        data=df, x='frame', y='speed_m_s', hue='track_id',
        palette={target_ids[0]: 'red', target_ids[1]: 'blue'},
        alpha=0.7, ax=ax_speed
    )
    ax_speed.set_title("Instantaneous Speed Over Time", fontsize=16, fontweight='bold')
    ax_speed.set_xlabel("Frame", fontsize=12)
    # 🌟 修改 Y 軸標籤：改為真實物理單位
    ax_speed.set_ylabel("Speed (Meters / Second)", fontsize=12)
    ax_speed.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(metrics_plot_path, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 步驟 5：視覺化 2 - 球員數據綜合比較長條圖 
    # ==========================================
    print("📊 正在產生球員物理數據對比長條圖...")
    metrics_df = pd.DataFrame(raw_metrics_data)
    
    # 🌟 修改為對應公尺與平方公尺的單位標籤
    plot_items = [
        ('Total Distance (m)', 'Distance (Meters)', 'Total Distance'),
        ('Max Speed (m/s)', 'Speed (m/s)', 'Maximum Speed'),
        ('Avg Speed (m/s)', 'Speed (m/s)', 'Average Speed'),
        ('Coverage Area (m^2)', 'Area (Square Meters)', 'Court Coverage Area'),
        ('Stamina Index', 'Index Value', 'Stamina Depletion Index'),
        ('AI Confidence (%)', 'Percentage (%)', 'AI Tracking Confidence')
    ]

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    fig2.suptitle("Player Physical Performance & AI Metrics", fontsize=20, fontweight='bold', y=1.02)
    
    palette_colors = ['#FF4C4C', '#4C84FF'] 

    for idx, (col_name, y_label, title) in enumerate(plot_items):
        ax = axes2[idx//3, idx%3]
        sns.barplot(x='Player ID', y=col_name, data=metrics_df, palette=palette_colors, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_xlabel("")
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', xytext=(0, 5), 
                        textcoords='offset points')

    plt.tight_layout()
    plt.savefig(comparison_chart_path, bbox_inches='tight')
    plt.close()

    print(f"✅ 動態熱力與物理速度分析圖已儲存至: {metrics_plot_path}")
    print(f"✅ 球員物理數據對比長條圖已儲存至: {comparison_chart_path}")
    print("\n🎉 單位轉換與所有圖表產生完畢！")


if __name__ == "__main__":
    YOUTUBE_URL = "https://www.youtube.com/watch?v=80gyfQ5CMQY"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    DOWNLOADED_VIDEO = os.path.join(current_dir, "auto_downloaded_match.mp4")
    OUTPUT_VIDEO = os.path.join(current_dir, "fixed_target_tracking.mp4")
    OUTPUT_CSV = os.path.join(current_dir, "fixed_target_data.csv")
    HEATMAP_IMG = os.path.join(current_dir, "fixed_target_heatmap_realcourt.png") 
    METRICS_PLOT = os.path.join(current_dir, "advanced_sports_metrics.png")
    COMPARISON_CHART = os.path.join(current_dir, "player_comparison_charts.png")       
    
    TARGET_IDS = [1, 147]
    ID_MAPPING = {
        497: 1,
        1495: 147
    }
    
    if not os.path.exists(DOWNLOADED_VIDEO):
        success = download_youtube_video(YOUTUBE_URL, DOWNLOADED_VIDEO)
    else:
        print("✅ 發現已下載的影片，跳過下載步驟。")
        success = True

    if success:
        analyze_and_visualize(
            video_path=DOWNLOADED_VIDEO, 
            output_video_path=OUTPUT_VIDEO, 
            output_csv_path=OUTPUT_CSV,
            heatmap_path=HEATMAP_IMG,
            metrics_plot_path=METRICS_PLOT,
            comparison_chart_path=COMPARISON_CHART, 
            target_ids=TARGET_IDS,
            id_mapping=ID_MAPPING,
            start_sec=249, 
            duration_sec=None
        )