# Scenario analysis

H0 × I1/I2/I3 × 四事件（201903、201908、201911、201912）的 GitHub 源码发布包。H1 是 H0 的旧标签，所有执行路径统一为 H0；没有延长灾害窗口，没有 2H0/4H0 独立代码。

已收齐用户指定的11份脚本，另纳入原项目的 model.py、库存/计算初始行业库存.py、库存/统一MRIO顺序.py 三份必要支持源码。公式与数值参数保持原样；修改仅限路径适配、H0标签与一处明确的旧库存目录衔接。原文件均未修改。

## 安装与配置

Python 3.10+：

```console
python -m pip install -r requirements.txt
```

复制 config.example.json 为本地 config.json，设置 data_root 为输入项目根目录。默认相对布局沿用原项目，包括 data_root/第五阶段情景分析new/I_H0/。各阶段路径键为 null 时使用默认布局，可按 INPUT_DATA.md 显式覆盖路径；候选路径键填写字符串数组，其余填写字符串。相对路径以命令运行目录为准，输出目录请设置为自己的结果目录。

入口在执行前只替换源码顶层路径赋值，不重新实现算法。禁止通过配置覆盖BPR、ETA、库存上下限等科学参数。`--check` 只编译配置后的代码，不验证数据存在或计算结果。

```console
python run.py S1 --config config.json --no-plots
python run.py S2 --config config.json
python run.py S3 --config config.json
python run.py S4 --config config.json
python run.py S5 --config config.json
python run.py S6 --config config.json
python run.py S7 --config config.json
python run.py S8 --config config.json
python run.py inventory_level1 --config config.json
python run.py order --config config.json
python run.py S9 --config config.json --no-plots
```

S1/S9 默认保留 TkAgg 绘图行为；服务器上使用 `--no-plots`，只禁用绘图及相关导入，不改变模拟参数。不绘图时可安装 requirements-no-plots.txt。预测仍需要 PyTorch，S4 仍需要 NetworkX。

## 必须保留的中间步骤

S8 → **order**（原统一MRIO顺序.py）→ S9。本次真实 MRIO 的全部285个列标签及对应行标签已核对，确认为 region-major。order 将 Z 的双轴、F 的活动轴、VA/X 的活动轴重排为 sector-major，并输出 MRIO模型_sector_major_checked。VA_TT 本身为19×15×365，不重排。不能把旧 MRIO模型 目录直接改名代替该步骤。

S9默认库存来自 **inventory_level1**，即原库存/计算初始行业库存.py：平均期初/期末库存÷经营投入×365、H19优先及H17+H18回退、1–90天裁剪。它与另外两份旧库存脚本采用不同口径。

可选旧分支：`inventory` → `inventory_variants`。前者按 Closing stocks/MRIO 中间投入计算，裁剪1–30天；后者low/middle/high为×0.5/×1/×2。已将后者默认基线目录显式接到前者的 sector_stock_days_AES2019_30 输出，不改变任何公式/上限。该分支不自动替换S9的Level1库存。

## 数据覆盖与验证

四个官方代码窗口合计22天；现有S1预测及S7中间结果实际覆盖21天，缺2019-12-01。S1原 build_base_times 从每月1日开始构建，再跳过 Tin=24 小时，因此十二月首次预测为12月2日。本包保留窗口、Tin和时间轴公式；没有伪造12月1日或重定义官方持续时间。内部历史变量 EVENT_WINDOWS_H1、period_type=h1_official 保留为兼容字段，含义均为H0。

已实际运行S2（三强度×四事件）、S3、S5、S6、S7、S8、两套库存分支、order及S9（三强度×365期）。S4输入使用原项目126份OD反推结果。972个CSV的数值列与45个NPY历史文件比较全部通过（rtol=atol=1e-10，最大绝对差4.55e-13）。S9输出365×285，均为有限非负值；这是运行/内部一致性检查，不是独立历史结果验证。

未重新执行S1神经网络推断、S4用户均衡/反推或GUI绘图；测试环境无torch/networkx/matplotlib。NumPy/Pandas路径及原有中间结果用于可行测试。完整记录见 validation.json、docs/historical_comparisons.json、docs/input_audit.json。复测：

```console
python -m unittest discover -s tests -v
python audit_inputs.py --mrio your_mrio.csv --s7-root your_step7_root --stock your_stock.csv
```

## 与已有仓库模块关系

作为 scenario_analysis/ 与 traffic_prediction/、od_estimation/ 并列发布。

- S1所需 model.py 与已有 traffic_prediction 中的模型逐字节相同。入口优先查找并复用相邻模块中的相同文件；SHA-256不同则使用随包审计副本。也可通过 shared_model_path 显式指定，内容不一致时会拒绝。权重、归一化参数、样本应复用交通预测模块的配套资产。S1本身增加三强度raw-only干预，不能用普通反事实命令代替。
- S4可复用 od_estimation 的接入节点、种子OD和边模板数据，通过配置映射。保留S4原反推实现：已有event.py的 update_region_od 在零权重分组时与S4存在处理差异，不能仅因默认参数相同就强行替换。六个共同函数AST完全相同，其余差异已记录在 docs/MODULE_RELATIONSHIP.md。
- 已有od_estimation十二月默认窗口12-02至12-08，不能用它覆盖本情景源码12-01至12-08的官方窗口。

## 发布内容

源码、配置示例、README、INPUT_DATA、provenance、requirements、tests、validation及审计记录均在本目录。原始数据、模型权重和本次运行的大批中间结果不随包分发；输入指纹见 data_manifest.json。未操作远端GitHub。源码许可证尚未由权利人指定，未擅自添加许可证。

未解决事项见 UNRESOLVED.md；实际完整目录见 TREE.txt。
