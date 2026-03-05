# Q-ALIGN 论文复现项目 (Image/Video Quality Assessment)

本项目是对论文 [Q-ALIGN: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels](https://arxiv.org/abs/2312.17090) 的个人复现实验。

## 1. 项目简介
Q-ALIGN 提出了一种改进大型多模态模型（LMM）在视觉评分任务（IQA/IAA/VQA）中表现的新方法。其核心在于将连续分值回归任务转化为基于五个离散文本等级（excellent/good/fair/poor/bad）的模拟推理任务。 

## 2. 复现范围
由于资源限制，本项目完成了以下部分的验证：
- **IAA (美学评价)**：在 AVA 数据集中抽样 2000 张进行测评。 
- **IQA (质量评价)**：在 KONIQtest 和 AGIQA-3K 数据集上进行了测评。
- **VQA (视频质量)**：在 Maxwell 数据集上进行了初步测评。

## 3. 复现方法与思路
参考论文提出的 `Q-ALIGN` Syllabus：
- **训练/推理转换**：将 MOS 分数映射为 5 个文本等级词汇。
- **分值计算**：提取 LMM 对各等级词汇的概率分布（Logits），通过加权平均还原为定量分数。 

## 4. 复现结果
<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/516d3974-48fd-40be-bd94-82b33d784be6" />

<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/97991445-5c1e-4b05-9fd9-f002a8bb5d42" />

<img width="450" height="500" alt="image" src="https://github.com/user-attachments/assets/cc29577b-b375-4434-a281-00d5390b8bb0" />

<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/2b27fb31-65b1-45d7-bc61-405fb137d126" />
