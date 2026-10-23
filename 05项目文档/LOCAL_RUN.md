> 目录已于 2026-10-23 整理。启动文件统一放在 `01运行入口`；路径均以项目根目录为起点。历史报告中的旧路径保留用于追溯，当前位置见根目录“00先看这里.html”。

# 本机首次试跑记录

2026-10-13，Windows 游戏本，已完成 `metal_nut` 的 PatchCore 开发集流程验证。

## 已完成

- 从 [CJ2323232/industrial-anomaly-inspection](https://github.com/CJ2323232/industrial-anomaly-inspection) 取回全部 11 个源文件，并在修改前逐个校验 GitHub blob 哈希。基础提交：`8c1888507ad099b2d20381ac56c576d5681d78f9`。
- 建立独立 Python 环境，安装 CUDA 版 PyTorch 和 anomalib，GPU 矩阵计算与依赖检查通过。
- 从 anomalib 维护方当前记录的数据源下载 MVTec AD，校验完整压缩包 SHA-256 后仅解压 `metal_nut`。
- 检查 335 张图像与 93 张异常掩码：均可读取、掩码尺寸正确，未发现图像文件完全重复。
- 完成正常特征提取、2252×384 正常特征库构建、开发集评价、模型保存和 45 张可视化输出。
- 重新加载保存的模型，对开发集图片再次进行 GPU 推理，输出有限数值及 256×256 异常图。

## 实测环境与配置

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU，6GB |
| 驱动 | 566.07 |
| Python | 3.10.10，独立虚拟环境 |
| PyTorch / torchvision | 2.6.0+cu118 / 0.21.0+cu118 |
| CUDA 运行时 | 11.8 |
| anomalib | 2.2.0 |
| 骨干与预训练权重 | ResNet18，timm/resnet18.a1_in1k |
| 输入 / batch / 随机种子 | 256×256 / 4 / 0 |
| coreset 采样率 | 1% |
| 正常训练 / 开发 / 保留测试 | 220 / 45 / 70 张 |

本机 Python 3.10 满足 [anomalib 2.2.0 的 Python >=3.10 要求](https://pypi.org/project/anomalib/2.2.0/)。PyTorch 版本组合与 CUDA 安装源依据 [PyTorch 官方历史版本说明](https://docs.pytorch.org/get-started/previous-versions/)。完整依赖见 [environment-lock.txt](../04实验结果/03早期运行与原理核验/preflight/environment-lock.txt)。

## 本次结果

| 指标 | 45 张开发图上的结果 |
|---|---:|
| Image AUROC | 1.000000 |
| Pixel AUROC | 0.984898 |
| Image F1 | 0.986301 |
| Pixel F1 | 0.830946 |

此次沿用仓库的首次试跑协议：开发集用于校准和流程评价，70 张保留测试图未参与评价。上述数字不是最终测试成绩，不用于宣称论文复现或正式工业检测性能。正式实验仍需实现项目协议中的独立正常校准、固定分层划分及原图坐标评价。

成功运行约 49.9 秒，包含脚本启动、模型加载、特征库建立、开发集评价和输出保存；不包含环境安装、数据及预训练权重下载，不能作为单图推理速度。

## Windows 兼容处理

以下三处兼容修改已同步至 GitHub main 分支，提交为 [1081c3b](https://github.com/CJ2323232/industrial-anomaly-inspection/commit/1081c3b17e501c7c29f225f1186bbbaf1cfc1c3a)，同时发布了试跑报告、划分清单和环境记录：

1. `02代码与配置/scripts/run_first_experiment.py`：Windows 使用普通 `latest` 输出目录，解决 anomalib 2.2.0 创建符号链接触发的 WinError 1314。
2. `02代码与配置/requirements-first-run.txt`：显式补充 `requests`，解决模型入口导入辅助模块时的缺失依赖。
3. `05项目文档/FIRST_RUN.md`：同步安装说明，并说明每次实验可指定新输出目录。

兼容处理使用 anomalib 2.2.0 的内部工作区接口，因此保留该版本锁定。算法、数据划分、输入尺寸与采样率沿用原脚本。可审查 [本机兼容补丁](../04实验结果/03早期运行与原理核验/preflight/windows_compatibility.patch)；实际运行元数据记录了基础提交和修改文件哈希。

取回的是可运行的源文件快照，当前目录没有 Git 历史。额外提供的 `02代码与配置/run_on_this_laptop.py` 是使用本机已验证环境的启动入口。

## 在这台游戏本上继续运行

在本文件所在的项目目录中执行：

```powershell
python run_on_this_laptop.py
```

入口会使用已建立的独立环境、已有数据和权重缓存，并按北京时间创建新的实验输出目录。可传入原脚本参数，例如：

```powershell
python run_on_this_laptop.py --batch-size 1
python run_on_this_laptop.py --output artifacts/my_next_run
```

2026-10-19 起，入口使用本项目目录内的 `.venv` 与 `.cache`，数据和输出同样保存在本项目文件夹。向 Mac 同步时同步源代码、配置与结果，并重新建立对应设备的环境。

## 结果入口

- [指标与模型路径](../04实验结果/03早期运行与原理核验/first_run_verified/development_results.json)
- [数据划分清单](../04实验结果/03早期运行与原理核验/first_run_verified/split_manifest.json)
- [成功运行日志](../04实验结果/03早期运行与原理核验/first_run_verified/console.log)
- [运行版本、时间与修改哈希](../04实验结果/03早期运行与原理核验/first_run_verified/run_metadata.json)
- [模型重新加载及划分核验](../04实验结果/03早期运行与原理核验/first_run_verified/verification.json)
- [模型 checkpoint](../04实验结果/03早期运行与原理核验/first_run_verified/Patchcore/MVTecAD/metal_nut/latest/weights/lightning/model.ckpt)
- [异常定位示例](../04实验结果/03早期运行与原理核验/first_run_verified/Patchcore/MVTecAD/metal_nut/latest/images/test/bent/000.png)
- [环境检查](../04实验结果/03早期运行与原理核验/preflight/environment.json)、[数据来源与官方校验值](../04实验结果/03早期运行与原理核验/preflight/dataset.json)、[图片与掩码检查](../04实验结果/03早期运行与原理核验/preflight/dataset_validation.json)

MVTec AD 来源：Bergmann 等，CVPR 2019 / IJCV 2021；许可 CC BY-NC-SA 4.0。官方入口：[MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)。

![开发集 bent/000：原图、真实掩码、异常热力图、预测掩码](../04实验结果/03早期运行与原理核验/first_run_verified/Patchcore/MVTecAD/metal_nut/latest/images/test/bent/000.png)
