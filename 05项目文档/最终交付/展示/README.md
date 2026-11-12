# 展示图片来源与许可

代码和自有文档采用MIT；本目录的MVTec图片及叠加改编单独采用 **CC BY-NC-SA 4.0**，不得因仓库根LICENSE为MIT而移除其许可。

|文件|来源|修改及许可|
|---|---|---|
|gui.png|本项目实际Tkinter窗口，2026-10-28截图|未选择图片、未伪造结果；项目自有界面，系统窗口装饰仍属于相应权利方|
|scratch021-original.png|MVTec AD / metal_nut/test/scratch/021.png|项目既有PNG导出，可能重新编码，保留原始像素内容；CC BY-NC-SA 4.0|
|scratch021-overlay.png|同一图片及已有WRN50 C检测输出|在超过像素阈值的区域叠加红色；本项目叠加改编贡献同样以CC BY-NC-SA 4.0分享|

归属：**MVTec AD，MVTec Software GmbH**。原始发布及数据获取：[MVTec AD官方页面](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)。许可：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，[完整条款](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)。

许可要求署名、非商业用途、分享改编时采用相同许可，并说明修改；不表示MVTec认可或背书本项目。保留上游免责声明，素材不提供保证。代码MIT不授予图片或模型商业使用权；如用途不确定，应向权利方核实。

这两张图片是同一个已使用开发样例的原图与预测展示，不是两个新增独立样本。冻结WRN50 C：K=11，图像分数2.3899730552，图像阈值2.2410881395，像素阈值2.1031780243；来自2026-10-28已保存输出，本次未重新推理。候选区域不能当作真实缺陷标注。

作者于2026-10-29明确授权公开这两份示例，发布范围仅此两图，不含完整数据集、GT或连续分数图。逐文件来源与SHA见[media_manifest.json](media_manifest.json)。GUI、核心指标图和可运行的公开摘要提供不依赖数据集图片的其他展示入口。
