import os
import sys
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# --- 路径配置 ---
# 项目根目录，用于确保能够正确 import q_align
PROJECT_ROOT = "/home/s202510010/workspace/test/Q-Align-main"
# 1000张图片存放文件夹
IMG_DIR = "/home/s202510010/workspace/test/Q-Align-main/fuxian/AGIQA-3K/图片"
# 数据文件 (含图片名和纯 MOS)
EXCEL_PATH = "/home/s202510010/workspace/test/Q-Align-main/fuxian/AGIQA-3K/纯mos.xlsx"
# 模型路径
MODEL_PATH = "q-future/one-align"

# 1. 环境准备：动态添加路径
sys.path.append(PROJECT_ROOT)
from q_align.evaluate.scorer import QAlignScorer

def main():
    # 2. 初始化评分器 (使用默认设置，自动调用 cuda:0)
    print(f"正在加载模型: {MODEL_PATH}...")
    try:
        scorer = QAlignScorer(pretrained=MODEL_PATH)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 3. 读取 Excel 数据
    df = pd.read_excel(EXCEL_PATH)
    # 自动识别列：假设第一列为图片文件名，第二列为 MOS
    img_col = df.columns[0]
    mos_col = df.columns[1]
    print(f"读取到 {len(df)} 条待测数据。")

    predict_scores = []

    # 4. 批量推理评分
    print(f"开始对 {len(df)} 张图片进行 IQA 评分...")
    
    # 使用推理模式以优化性能
    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df), desc="测评进度"):
            img_name = str(row[img_col])
            img_path = os.path.join(IMG_DIR, img_name)
            
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    # 调用 Q-Align 评分逻辑：输入为 List[Image.Image]
                    # 模型根据文本级别进行加权计算
                    score_tensor = scorer([img]) 
                    predict_scores.append(score_tensor.item())
                except Exception as e:
                    print(f"\n[错误] 图片 {img_name} 处理异常: {e}")
                    predict_scores.append(None)
            else:
                # 若 1000 张图片中某些文件不存在，记录为空并跳过
                predict_scores.append(None)

    # 5. 结果统计
    df['predicted_score'] = predict_scores
    
    # 剔除无效数据行（如图片缺失或读取失败）
    valid_data = df.dropna(subset=['predicted_score', mos_col])
    
    if len(valid_data) > 1:
        # 计算相关系数指标
        srcc, _ = spearmanr(valid_data['predicted_score'], valid_data[mos_col])
        plcc, _ = pearsonr(valid_data['predicted_score'], valid_data[mos_col])
        
        print("\n" + "="*40)
        print(f"1000张图片测评完成 (有效样本: {len(valid_data)})")
        print(f"SRCC (Spearman Rank Correlation): {srcc:.4f}")
        print(f"PLCC (Pearson Linear Correlation): {plcc:.4f}")
        print("="*40)
    else:
        print("\n[错误] 未能获取足够的有效评分数据。")

    # 6. 保存最终结果
    output_path = EXCEL_PATH.replace(".xlsx", "_ce1000_results.xlsx")
    df.to_excel(output_path, index=False)
    print(f"测评结果已存至: {output_path}")

if __name__ == "__main__":
    main()
