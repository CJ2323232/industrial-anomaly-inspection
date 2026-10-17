# 固定清单与独立正常校准的参考基线

入口准备日期：2026-10-16，同日已由使用者手动完成首个单类 GPU 实验，包括 176 张正常图重建库、44 张独立正常图校准、45 张开发图评价和模型保存加载核验。见[实际完成记录](results/2026-10-16-trusted-baseline.md)。70 张保留测试未打开，后续配置仍需各自实际运行验证。

## 运行

准备[首次试跑环境](FIRST_RUN.md)、数据和[固定清单](../configs/splits/metal_nut-v1.json)后，在仓库根目录执行：

```console
python scripts/run_trusted_baseline.py --data-root datasets/MVTecAD --check-only
python scripts/run_trusted_baseline.py --data-root datasets/MVTecAD --output artifacts/trusted_baseline_001
```

输出目录必须尚不存在，以保留已有实验。脚本要求 anomalib 2.2.0 和 CUDA；启动前须在当前 Hugging Face 缓存中准备 `timm/resnet18.a1_in1k` 的 `model.safetensors`。默认缓存也可通过环境变量 `HF_HOME` 指向已有缓存。本地权重检查使用 `local_files_only=True`。首次实验的已下载权重可复用，不复用旧模型中的正常特征库。

运行器直接调用 anomalib 的 `PatchcoreModel`，不用 Lightning Engine 的验证与自适应后处理，避免这些组件隐式接触开发标签来选择阈值。这是参考算法和项目自定义运行／评价流程，尚非个人独立实现 PatchCore。

## 冻结的起始配置

ResNet18 的 `a1_in1k` 权重，`layer2/layer3`，256×256 全图缩放、无中心裁剪、ImageNet 归一化，1% coreset，拟合 batch=4，评价 batch=1，seed=0，9 个支持邻居的加权图像评分。骨干参数及 BatchNorm 统计在建库前后计算身份，要求保持不变。

读取固定清单，校验身份、角色、跨集合路径和重复图像哈希，并核对本次使用的 176 拟合、44 正常校准、45 开发图像及 37 张开发异常掩码的文件 SHA-256。保留测试只检查清单的分配关系，不打开其图片／掩码，不生成预测。

1. 只用 `fit` 建正常特征库。
2. 只用 `normal_calibration` 预测并校准，图像阈值取原始图像分数 95% 分位数，像素阈值取这些正常图的原图尺寸像素分数池的 99.5% 分位数；均采用 NumPy `linear` 方法，严格 `>` 判异常。
3. 将校准结果写入文件并保存身份后，才开始 `development` 评价；评价后要求文件身份不变。

原始输入不裁剪；参考实现内部的异常图按其 v2.2.0 默认方式上采样和平滑，再用 bilinear／align_corners=False 映射到原图尺寸。原始 GT 掩码不缩放。图像 AUROC/AP 和 pooled pixel AUROC/AP 用连续原始分数精确计算，阈值仅用于混淆计数、F1、IoU。

## 结果文件

`preflight.json`、`config.json`、`split_manifest.json`、`calibration.json`、两份逐图预测 CSV、`report.json`、`error_cases.json`、`model.pt`，以及校准／开发原图尺寸连续异常图和开发可视化。成功时打印 `BASELINE_COMPLETED`，并以 0 退出。

`model.pt` 是模型状态、配置、清单身份和阈值字典，采用本脚本的 `load_checkpoint` 重新构建无预训练下载的模型再载入状态；与旧 Lightning `.ckpt` 格式不同。结束前核对同一开发样本保存加载前后的图像分数与异常图，未通过则不写成功报告。

图像组内只有单一真值类别时，AUROC/AP 返回空值，报告对应检测数量与召回／误报。正常样本既无真实异常像素又无预测异常像素时，单图 IoU/F1 不定义，不填成 1 来抬高平均数。小面积分组界限为 `<0.1%`、`0.1%–<1%`、`≥1%`。

精确像素排序指标可能占用较多 CPU 内存和时间，不能把整个流程耗时当成单图推理延迟。保存的峰值显存是 PyTorch 分配量，未覆盖所有进程和 CUDA 保留内存，也不是正式性能测量。

## 已检查的边界与待验证范围

回归测试检查连续分数评价、阈值相等处理、单类指标不定义、拒绝异常校准、分位数与非有限数、面积组边界、清单篡改／角色错置、源文件改变、保留测试不被打开、原图坐标插值和随机权重模型保存加载。这些测试不训练工业模型，也不下载预训练权重。

首个完整 GPU 运行及实际拟合模型的保存加载已通过。开发集只有 8 张正常与 37 张异常图；44 张正常图及其像素也不等于独立像素试验，因此经验分位数不保证未来误报水平。新流程使用独立校准，但尚待多种子、其他类别与最终保留测试。此自定义划分与缩减配置不直接与原论文数字比较。

代码出处：[PatchCore 原论文](https://arxiv.org/abs/2106.08265)、[anomalib 2.2.0 PatchCore](https://github.com/open-edge-platform/anomalib/tree/v2.2.0/src/anomalib/models/image/patchcore)。
