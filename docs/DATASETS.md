# 数据集、论文与参考实现

资料核对日期：2026-10-12。此清单记录来源，不表示数据已下载、代码已验证或实验已完成。

## MVTec AD：第一版数据

[官方介绍与下载入口](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)。页面提供 Download dataset 表单和 Evaluation Code 入口。数据包含正常训练图、正常／异常测试图和异常像素标注。

第一版使用 `metal_nut`、`cable`、`carpet`，先处理 `metal_nut`。解压后通常包含 `train/good`、`test/good`、各异常类型目录与 `ground_truth`。

备用下载信息：[anomalib 数据模块中的 DOWNLOAD_INFO](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/data/datamodules/image/mvtecad.py)。源码同时记录完整压缩包 URL 和校验值；获取后按对应版本验证，不混用重新打包文件的校验值。

[数据集论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html)。

许可为 **CC BY-NC-SA 4.0**，用途与展示遵循数据集原许可。完整数据保留在本地，本仓库只记录下载与准备方式。

## VisA：后续第二数据集

[作者仓库](https://github.com/amazon-science/spot-diff) · [下载入口](https://github.com/amazon-science/spot-diff#data-download) · [划分与整理](https://github.com/amazon-science/spot-diff#data-preparation)。

第一版完成后可选择一个 PCB 子集，使用其正常训练图重新建立模型，并按作者的 1-class 划分评测。目标域重新建模与无需目标数据的零样本迁移应分开表述。

## 后续扩展

- [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)：独立数据集，采用自身评测规则。
- 真实工业采集数据：按产品／端面 ID、批次、设备等分组划分；公开内容需有明确的数据使用权限。

## 方法与代码

| 资源 | 入口 | 使用方式 |
|---|---|---|
| PatchCore 论文 | [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265) | 学习局部特征、代表性特征库与评分 |
| PatchCore 作者实现 | [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection) | 核对参考方法及配置 |
| EfficientAD 论文 | [EfficientAD](https://arxiv.org/abs/2303.14535) | 学习教师—学生、自动编码器与异常图 |
| 通用参考框架 | [anomalib](https://github.com/open-edge-platform/anomalib) | 建立基线，记录所选版本 |
| EfficientAD 框架文档 | [Efficient AD](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/efficient_ad.html) | 核对辅助数据、权重及训练约定 |
| 辅助自然图片 | [ImageNette](https://github.com/fastai/imagenette) | 按所选 EfficientAD 实现准备并记录 |
| PyTorch 学习 | [官方基础教程](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | 张量、数据与训练基础 |
| 分块参考 | [anomalib Tiling](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/data/utils/tiling.html) | 对照分块与拼接过程 |
| 部署参考 | [ONNX Runtime Python](https://onnxruntime.ai/docs/get-started/with-python.html) | 后续导出与推理验证 |

额外阅读：[nelson1425/EfficientAD](https://github.com/nelson1425/EfficientAD) 明确标注为非官方实现。使用或借鉴外部实现时，保留来源、版本与许可，不能把参考代码当成本项目原创实现。
