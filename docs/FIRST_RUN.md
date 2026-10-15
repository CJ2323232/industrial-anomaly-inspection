# Windows + RTX 3060：第一次试跑

目标：先跑通 `metal_nut` 的 PatchCore 参考实现，得到模型、开发集指标和可视化输出。

2026-10-13 已在 Windows + RTX 3060 Laptop 6GB、Python 3.10.10、PyTorch 2.6.0+cu118、anomalib 2.2.0 上完成首次试跑及模型重新加载验证。配置、开发集结果和适用范围见[试跑报告](results/2026-10-13-first-run.md)。遇到报错仍需保留完整输出。

## 1. 新建环境

在 **Anaconda Prompt** 逐行运行：

```bat
conda create -n industrial-ad python=3.10 -y
conda activate industrial-ad
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install anomalib==2.2.0 "requests>=2.32,<3"
```

CUDA 11.8 是这里选用的起始安装方案，不需要先单独安装 CUDA Toolkit。GPU 驱动需要兼容该版本；启动脚本会检查 CUDA 是否可用。安装命令来源：[PyTorch 官方历史版本](https://docs.pytorch.org/get-started/previous-versions/)、[anomalib 2.2.0](https://pypi.org/project/anomalib/2.2.0/)。

显式安装 `requests`，用于 anomalib 2.2.0 模型入口导入的辅助模块。Windows 启动脚本使用普通输出目录，避免创建符号链接所需的额外权限。每次试验可用 `--output` 指定新的目录，以保留此前结果。

## 2. 获取项目

有 Git 时，在自己选好的项目父目录中运行：

```bat
git clone https://github.com/CJ2323232/industrial-anomaly-inspection.git
cd industrial-anomaly-inspection
```

已有项目则在项目目录运行 `git pull`。没有 Git 时，在[仓库首页](https://github.com/CJ2323232/industrial-anomaly-inspection)点击 **Code → Download ZIP**，解压后在 Anaconda Prompt 中进入包含 `README.md` 的目录。

## 3. 放入数据

从 [MVTec AD 官方页面](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)获取数据。初次只需 `metal_nut` 类别；若拿到完整压缩包，可只提取这一类。

解压后整理为：

```text
industrial-anomaly-inspection/
  scripts/run_first_experiment.py
  datasets/MVTecAD/metal_nut/
    train/good/*.png
    test/good/*.png
    test/bent/*.png
    test/color/*.png
    test/flip/*.png
    test/scratch/*.png
    ground_truth/bent/*_mask.png
    ground_truth/color/*_mask.png
    ground_truth/flip/*_mask.png
    ground_truth/scratch/*_mask.png
```

数据放其他磁盘也可以，启动时指定 `--data-root "D:\data\MVTecAD"`；这个目录下面应直接包含 `metal_nut`。

## 4. 启动

```bat
python scripts/run_first_experiment.py
```

第一次会下载预训练骨干权重，需要网络。看到提取特征、coreset sampling 或进度条属于正常过程；PatchCore 建立正常特征库，不需要像 YOLO 一样训练很多轮。这里使用 ResNet18、256×256 输入、1% coreset 和 batch=4 来降低启动规模，不能把结果当成原论文默认配置的复现。

Windows 使用普通 `latest` 输出目录。再次运行时建议给 `--output` 指定新的目录，例如 `--output artifacts/first_run_20261013`，以保留此前的模型和记录。

结束后查看：

- `artifacts/first_run/development_results.json`：开发集指标与模型位置。
- `artifacts/first_run/split_manifest.json`：本次数据划分路径。
- `artifacts/first_run/` 下 anomalib 生成的模型和可视化文件。

跑通后记录环境：

```bat
python -m pip freeze > artifacts/first_run/environment.txt
```

## 本次结果的范围

脚本使用官方正常训练图建库，从官方 test 中拆出约 40% 作为开发集；另外约 60% 保留，脚本不评价该部分。开发集同时用于 anomalib 的校准和试跑评价，所以 F1、阈值相关指标只能作为流程检查结果。该划分由 anomalib 生成，尚未实现项目协议中的按缺陷类型分层及文件哈希检查。

这一步不替代 [正式评测协议](PROTOCOL.md)。正式对照实验前会建立固定划分、独立正常校准和原图坐标评测；不要把本次试跑数字作为最终简历指标。第一次试跑生成的路径清单只用于记录，后续完整协议需要独立固化清单。

## 常见卡点

- `CUDA out of memory`：先改为 `python scripts/run_first_experiment.py --batch-size 1`。
- `CUDA is unavailable`：确认激活的是 `industrial-ad` 环境，再核对上述 CUDA 版 PyTorch 安装；仍失败时保留输出。
- 权重下载超时：保留报错中的下载网址，按该网址处理网络或本地权重缓存，不反复重装所有依赖。
- 提示缺少目录：按上面的结构检查是否多套了一层 `metal_nut` 文件夹。

完整实测依赖见[环境锁定记录](results/2026-10-13-environment.txt)。该记录对应 Python 3.10 和 Windows；仍需先从上述 CUDA 11.8 安装源获取匹配的 torch 与 torchvision。

核心参考：[anomalib PatchCore 2.2.0 源码](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/src/anomalib/models/image/patchcore/lightning_model.py)。
