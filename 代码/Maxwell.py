import os
import sys
import torch
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# --- 路径配置 ---
PROJECT_ROOT = "/home/s202510010/workspace/test/Q-Align-main"
VIDEO_DIR = "/home/s202510010/workspace/test/Q-Align-main/fuxian/maxwell/test"
EXCEL_PATH = "/home/s202510010/workspace/test/Q-Align-main/fuxian/maxwell/test.xlsx"
MODEL_PATH = "q-future/one-align" 

sys.path.append(PROJECT_ROOT)

# 从你提供的 scorer.py 导入
from q_align.evaluate.scorer import QAlignVideoScorer, load_video

def main():
    # 1. 初始化评分器
    print(f"正在加载视频模型: {MODEL_PATH}...")
    # 源码中初始化参数为 pretrained 和 device
    scorer = QAlignVideoScorer(pretrained=MODEL_PATH, device="cuda:0")

    # 2. 读取 Excel 文件
    df = pd.read_excel(EXCEL_PATH)
    # 根据你的截图，列名分别是 '视频名称' 和 '综合'
    video_col = '视频名称'
    gt_col = '综合'
    
    print(f"Excel 读取成功，共 {len(df)} 条数据。")

    predict_scores = []

    # 3. 开始遍历 909 个视频进行推理
    print(f"开始对 {VIDEO_DIR} 中的视频进行评分...")
    
    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df), desc="测评进度"):
            video_name = str(row[video_col]).strip()
            video_path = os.path.join(VIDEO_DIR, video_name)
            
            if os.path.exists(video_path):
                try:
                    # 使用源码的 load_video (1fps采样)
                    video_frames = load_video(video_path)
                    
                    # 按照 List[List[Image]] 格式输入
                    score_tensor = scorer([video_frames])
                    
                    # 提取分数值
                    score = score_tensor.tolist()[0]
                    predict_scores.append(score)
                except Exception as e:
                    print(f"\n[错误] 视频 {video_name} 处理异常: {e}")
                    predict_scores.append(None)
            else:
                # 如果文件夹里没找到这个视频
                predict_scores.append(None)

    # 4. 将测评分数填入 DataFrame
    df['测评分数'] = predict_scores
    
    # 5. 计算指标 (剔除缺失数据)
    valid_data = df.dropna(subset=['测评分数', gt_col])
    
    print("\n" + "="*40)
    print(f"处理完成！有效样本数: {len(valid_data)}")
    
    if len(valid_data) > 1:
        srcc, _ = spearmanr(valid_data['测评分数'], valid_data[gt_col])
        plcc, _ = pearsonr(valid_data['测评分数'], valid_data[gt_col])
        
        print(f"SRCC (Spearman): {srcc:.4f}")
        print(f"PLCC (Pearson):  {plcc:.4f}")
    else:
        print("有效数据不足，无法计算 SRCC/PLCC")
    print("="*40)

    # 6. 保存新文件 (包含：视频名称、综合评分、测评分数)
    # 只保留这三列
    output_df = df[[video_col, gt_col, '测评分数']]
    output_path = "/home/s202510010/workspace/test/Q-Align-main/fuxian/maxwell/test_vqa_final_results.xlsx"
    output_df.to_excel(output_path, index=False)
    
    print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    main()
