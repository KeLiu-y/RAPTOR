<div align="center">
  <!-- <img src="RDMFNet.png" width="800"/>  -->
  <div>&nbsp;</div>
  <img src="GR-DC.png" width="800"/>
  <div>&nbsp;</div>

[![license](https://img.shields.io/github/license/open-mmlab/mmrotate.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-blue.svg)](https://pytorch.org/)
[![MMRotate](https://img.shields.io/badge/MMRotate-0.3.4-orange.svg)](https://github.com/open-mmlab/mmrotate)


</div>

## 📝 简介 (Introduction)

**RDMFN** (Rotation-Aware Dual-Branch Mamba Fusion Network) 是专门为军事遥感图像中有向目标检测（Oriented Object Detection）设计的高效骨干网络架构。

针对军事遥感中极端的几何多变性、上下文模糊以及特征退化挑战，RDMFN 提出了一种解耦局部几何建模与全局上下文聚合的双分支架构。它在保持线性计算复杂度的同时，显著提升了对复杂环境下旋转目标的检测精度。

### 🌟 核心亮点 (Key Highlights)
* **并行局部-全局融合块 (PLGFB)**: 协同两个专业分支进行特征提取。**GR-DC** (组内旋转可变形卷积) 动态预测旋转角度以适应目标轮廓；**RVMB** 分支利用视觉 Mamba 的线性复杂度捕获全局上下文。
* **LoGGS 引导 Stem**: 采用拉普拉斯-高斯 (LoG) 滤波器在输入阶段增强边缘先验并抑制噪声。
* **ADRFD 降采样模块**: 引入自适应动态路由融合降采样，智能保留细粒度细节，极大减少空间信息丢失。
* **Military-RSOD 数据集**: 构建了一个包含 **53 个精细类别**、18,195 张图像的大规模军事遥感数据集，提供精确的有向旋转框 (OBB) 标注。

---

## 🚀 性能概览 (Performance at a Glance)

在 **Military-RSOD** 数据集上，RDMFN 与当前主流 SOTA 方法的性能对比：

| 阶段 (Stage) | 检测方法 (Method) | 骨干网络 (Backbone) | FLOPs (G) | mAP (%) |
| :--- | :--- | :--- | :---: | :---: |
| **One-Stage** | R³Det | ResNet-50 | 346.8 | 81.19 |
| | SASM | ResNet-50 | - | 82.08 |
| | O-RepPoints | ResNet-50 | 194.4 | 82.42 |
| | R³Det-GWD | ResNet-50 | 336.2 | 82.64 |
| | R³Det-KLD | ResNet-50 | 336.2 | 83.26 |
| | S²ANet | ResNet-50 | 199.8 | 81.02 |
| | S²ANet | LEGNet-S | 175.3 | 83.46 |
| | S²ANet | LSKNet-S | 164.3 | 83.80 |
| | S²ANet | PKINet-S | 502.6 | 83.86 |
| | **S²ANet** | **RDMFN (Ours)** | **161.53** | **84.16** |
| **Two-Stage** | CenterMap | ResNet-50 | 198.4 | 80.56 |
| | SCRDet | ResNet-50 | - | 81.50 |
| | Roi Trans. | ResNet-50 | 225.4 | 84.26 |
| | Strip R-CNN | StripNet | 218.3 | 85.39 |
| | O-RCNN | ResNet-50 | 211.4 | 83.89 |
| | O-RCNN | LSKNet-S | 173.6 | 84.84 |
| | O-RCNN | DecoupleNet | 142.4 | 84.47 |
| | O-RCNN | LEGNet-S | 184.6 | 85.42 |
| | **O-RCNN** | **RDMFN (Ours)** | **172.55** | **86.39** |

---

## 📂 数据集 (Military-RSOD Dataset)

数据集涵盖了海、陆、空全方位的军事目标，能够评估模型在复杂军事场景下的泛化能力：
* **空中目标**: 战略轰炸机 (B-1B, TU-160)、运输机 (C-17)、五代战斗机 (F-35, SU-35) 等。
* **海上目标**: 尼米兹级核动力航空母舰 (NAA)、阿利·伯克级驱逐舰 (ABD)、潜艇、辅助舰船。
* **地面设施**: 装甲车 (AFV)、军事工程车 (MCV)、桥梁、机场设施。

### 📥 数据集下载链接
- **Baidu Netdisk**: [整理中稍后开源](#) (提取码: `xxxx`)

---

## 🛠️ 安装 (Installation)

本项目基于 [MMRotate 0.3.4](https://github.com/open-mmlab/mmrotate) 构建。


1. **环境准备**:
```shell
conda create -n rdmfn python=3.8 -y
conda activate rdmfn
# 推荐安装版本
conda create -n openmmlab python=3.7 pytorch==1.7.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate openmmlab
pip install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .

