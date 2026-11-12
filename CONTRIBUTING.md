# 贡献指南


## 验证

基础测试只需要Python 3.10或更新版本，在项目根目录执行。

```sh
python -B -S "02代码与配置/scripts/run_tests.py" --suite basic
```

安装推理依赖后可运行 `--suite unit`。真实模型集成检查使用 `--suite integration`，模型应从正式发布获取并通过SHA校验。测试条件与跳过原因见现有测试说明。

核心代码格式化范围在 pyproject.toml 中登记。使用 `black==26.10.1`，执行 `python -m black --check .` 检查，通过同一命令去掉 `--check` 可格式化登记文件。

算法变更必须另行讨论。格式整理需保留原有运算顺序、dtype、阈值、K、随机种子及数据划分。提交时说明测试和数值一致性检查范围。保留真实失败记录与第三方署名。
