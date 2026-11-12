# 第三方与数据许可说明

本项目有权授权的项目代码和自有说明文档采用根目录[MIT许可证](LICENSE)，版权署名为CJ2323232。MIT不覆盖下列第三方软件、数据、模型权重或外部材料；原有上游声明继续有效。

|项目|用途|已核查许可/来源|
|---|---|---|
|anomalib 2.2.0|PatchCore参考模型、特征提取和加载|Apache-2.0；[v2.2.0原始LICENSE](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/LICENSE)|
|PyTorch 2.6.0|张量、推理和权重加载|BSD-3-Clause；[原仓库LICENSE](https://github.com/pytorch/pytorch/blob/v2.6.0/LICENSE)|
|torchvision 0.21.0|图像预处理|BSD；[原仓库LICENSE](https://github.com/pytorch/vision/blob/v0.21.0/LICENSE)|
|timm 1.0.30|预训练骨干接口|Apache-2.0；依赖包自身许可|
|Lightning 2.6.6|anomalib相关运行依赖|Apache-2.0；依赖包自身许可|
|NumPy / SciPy / pandas / scikit-learn|数值计算、评价与结果处理|已安装包包含BSD类许可及各自第三方声明|
|Pillow 12.3.0|图片读写|MIT-CMU|
|Matplotlib 3.10.9|图表和热力图色表|Matplotlib自身许可；不将其改为本项目MIT|

当前公开预览不打包以上依赖库本体。安装依赖时保留其wheel中已有的许可文件；anomalib原始Apache文本另附于05项目文档/最终交付/第三方许可文本中，作为归属记录。完整已安装包的许可元数据记录在本次本地交付补全目录。

## 数据与权重

- MVTec AD来源：[原发布方](https://www.mvtec.com/company/research/datasets/mvtec-ad)。项目既有核查记录标注CC BY-NC-SA 4.0；本项目MIT不能改变数据的非商业等条件。
- Roboflow Nuts用于5880张跨域记录，不作为自建数据集，不重新授权或包含在公开预览中。
- ImageNet预训练骨干及项目保存的模型、正常Memory Bank不在源码MIT声明的分发范围内；公开预览不包含这些artifact。后续分发需按原来源核对许可。
- 原始图片、GT、叠加了原图的结果图、虚拟环境、缓存、大型NPY/NPZ、私人绝对路径报告均不在发布允许清单。

项目贡献是基于开源参考实现的数据协议、实验分析、诊断与工程封装。不得移除第三方版权或将其实现冒称为自己的原创代码。
