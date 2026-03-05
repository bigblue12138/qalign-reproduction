import os
import sys
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# --- 路径配置 ---
PROJECT_ROOT = "/home/s202510010/workspace/test/Q-Align-main"
# 2000张图片存放文件夹
IMG_DIR = "/home/s202510010/workspace/test/Q-Align-main/5000sample/AVA测试集均匀抽样2000"
# 数据文件 (AVA测试集)
EXCEL_PATH = "/home/s202510010/workspace/test/Q-Align-main/fuxian/AVA/测试集均匀抽样2000.xlsx"
# 模型路径
MODEL_PATH = "q-future/one-align"

# 1. 环境准备：确保能导入 q_align
sys.path.append(PROJECT_ROOT)
try:
    # 注意：测 AVA 数据集必须使用 QAlignAestheticScorer
    from q_align.evaluate.scorer import QAlignAestheticScorer
except ImportError:
    print("错误：无法从 q_align 导入 QAlignAestheticScorer，请检查项目路径。")

def main():
    # 2. 初始化美学评分器 (Aesthetic Scorer)
    print(f"正在加载美学模型: {MODEL_PATH}...")
    try:
        # 使用专用美学类，这会改变 Prompt 为 "How would you rate the aesthetics..."
        scorer = QAlignAestheticScorer(pretrained=MODEL_PATH, device="cuda:0")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 3. 读取 Excel 数据
    # dtype=str 解决 2766 变成 2766.0 的问题
    df = pd.read_excel(EXCEL_PATH, dtype=str)
    
    img_col = df.columns[0]
    mos_col = df.columns[1]
    
    # 转换 MOS 列为数值用于计算相关系数
    df[mos_col] = pd.to_numeric(df[mos_col], errors='coerce')
    
    print(f"读取到 {len(df)} 条待测数据。")

    predict_scores = []

    # 4. 批量推理评分
    print(f"开始进行 AVA 美学评分测评...")
    
    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df), desc="测评进度"):
            # 清洗 ID：去掉 .0 后缀并拼接 .jpg
            raw_id = str(row[img_col]).split('.')[0]
            img_name = f"{raw_id}.jpg"
            img_path = os.path.join(IMG_DIR, img_name)
            
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    # 模型输入是 List[Image.Image]
                    score_tensor = scorer([img]) 
                    current_score = score_tensor.item()
                    predict_scores.append(current_score)
                    
                    # 打印前五个具体分数，观察预测是否“准”
                    if len([s for s in predict_scores if s is not None]) <= 5:
                        print(f"\n[前5样本检查] 图片: {img_name} | 真实MOS: {row[mos_col]:.4f} | 预测分: {current_score:.4f}")
                        
                except Exception as e:
                    print(f"\n[错误] 图片 {img_name} 处理异常: {e}")
                    predict_scores.append(None)
            else:
                if index < 5:
                    print(f"\n[警告] 找不到图片文件: {img_path}")
                predict_scores.append(None)

    # 5. 结果统计与指标计算
    df['predicted_score'] = predict_scores
    
    # 剔除无效数据行（如图片读取失败的）
    valid_data = df.dropna(subset=['predicted_score', mos_col])
    
    if len(valid_data) > 1:
        srcc, _ = spearmanr(valid_data['predicted_score'], valid_data[mos_col])
        plcc, _ = pearsonr(valid_data['predicted_score'], valid_data[mos_col])
        
        print("\n" + "="*50)
        print(f"AVA 测评完成 (有效样本: {len(valid_data)})")
        print(f"SRCC (斯皮尔曼相关系数): {srcc:.4f}")
        print(f"PLCC (皮尔逊相关系数): {plcc:.4f}")
        print("="*50)
    else:
        print("\n[错误] 有效样本不足，请检查图片路径和 Excel ID 是否匹配。")

    # 6. 保存最终结果到 AVA 文件夹
    output_path = os.path.join(os.path.dirname(EXCEL_PATH), "AVA_Aesthetic_Results.xlsx")
    df.to_excel(output_path, index=False)
    print(f"测评结果已存至: {output_path}")

if __name__ == "__main__":
    main()
