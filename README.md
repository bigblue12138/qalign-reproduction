# Q-ALIGN 论文复现项目 (Image/Video Quality Assessment)

本项目是对论文 [Q-ALIGN: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels](https://arxiv.org/abs/2312.17090) 的个人复现实验。

## 1. 项目简介
[cite_start]Q-ALIGN 提出了一种改进大型多模态模型（LMM）在视觉评分任务（IQA/IAA/VQA）中表现的新方法。其核心在于将连续分值回归任务转化为基于五个离散文本等级（excellent/good/fair/poor/bad）的模拟推理任务。 [cite: 772, 773]

## 2. 复现范围
由于资源限制，本项目完成了以下部分的验证：
- [cite_start]**IAA (美学评价)**：在 AVA 数据集中抽样 2000 张进行测评。 [cite: 788]
- [cite_start]**IQA (质量评价)**：在 KONIQtest 和 AGIQA-3K 数据集上进行了测评。 [cite: 788]
- [cite_start]**VQA (视频质量)**：在 Maxwell 数据集上进行了初步测评。 [cite: 788]

## 3. 复现方法与思路
参考论文提出的 `Q-ALIGN` Syllabus：
- [cite_start]**训练/推理转换**：将 MOS 分数映射为 5 个文本等级词汇。 [cite: 785]
- [cite_start]**分值计算**：提取 LMM 对各等级词汇的概率分布（Logits），通过加权平均还原为定量分数。 [cite: 380, 785]

## 4. 复现结果
<img width="960" height="626" alt="image" src="https://github.com/user-attachments/assets/9aba3b6a-ef13-4edb-a4ae-ce330c93fe13" />
<img width="783" height="419" alt="image" src="https://github.com/user-attachments/assets/a3aefcbc-771d-4987-b739-706a0d58b293" />
<img width="622" height="572" alt="image" src="https://github.com/user-attachments/assets/8eff332d-5511-480c-831d-ffa2243a361a" />
<img width="866" height="476" alt="image" src="https://github.com/user-attachments/assets/40f9cfa2-7075-4cd7-95e3-96110772f6ca" />
