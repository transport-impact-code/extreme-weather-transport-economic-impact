# traffic_prediction

基于六个上传原始脚本整理的 24 小时历史 → 12 小时交通流预测模块。代码已逐个检查；未提供研究数据与已训练资产，因此本包不是完整实验复现。没有修改远端 GitHub。

## 安装和运行

Python 3.10+，在本 README 所在目录执行：

```console
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m traffic_prediction --help
python -m traffic_prediction label --config configs/default.json
python -m traffic_prediction samples --config configs/default.json
python -m traffic_prediction train --config configs/default.json
python -m traffic_prediction analyze --config configs/default.json
```

先编辑 `configs/default.json`。所有路径相对于**运行命令时的工作目录**；配置文件可放在其他位置。可安装为包：`python -m pip install -e .`。Windows 批处理：`powershell -File scripts/run_pipeline.ps1`；Linux/macOS：`sh scripts/run_pipeline.sh`。这些脚本执行真实四阶段流程，需要全部数据。

每个阶段支持 `--set KEY=JSON` 覆盖原脚本大写常量，例如 `python -m traffic_prediction train --set EPOCHS=1 --set NUM_WORKERS=0`。字符串值须使用合法 JSON 引号，优先在配置文件中编辑。`--show-config` 仅显示显式覆盖值，无需科学依赖。未知字段报错。

分析其他月份时同时设置 `MONTH` 与 `MONTH_NPZ`。默认配置固定为 201903；仅修改 MONTH 不会覆盖配置中显式指定的 NPZ。训练 SAVE_DIR 改变后，四个训练产物路径会自动更新；也允许逐一指定 CKPT_PATH、NORM_PATH、LOSS_PATH、LOSS_PNG。模型输出固定 12 步，不能通过配置改变 Tin/Tout。

## 原始文件映射

| 原始文件 | 包内模块 | 命令 |
|---|---|---|
| S1构建含标签的数据.py | traffic_prediction/label_data.py | label |
| S2构建样本.py | traffic_prediction/build_samples.py | samples |
| dataset_train.py | traffic_prediction/dataset_train.py | 被训练模块使用 |
| model.py | traffic_prediction/model.py | 被训练及分析模块使用 |
| train.py | traffic_prediction/train.py | train |
| plot_all_multi.py | traffic_prediction/plot_all_multi.py | analyze |

部署改动限定为相对路径默认值、配置/CLI、包内导入、把 S2 顶层数据处理封装为 main，以及把建目录和分析窗口初始化移到 CLI。请通过 CLI 调用流程；直接调用分析模块 main 之前必须初始化事件时间。

`provenance.json` 记录六个原始文件的 SHA-256；`docs/source_equivalence.json` 记录 29 个原始函数/类与 S2 执行体共 30 项 AST 指纹。原文件从引用对话附件读取；本机不存在用户提到的 /mnt/data 路径，因此没有假定该目录可用。

## 保留的计算行为

- S1 在全部月份气象记录上计算分位数阈值；没有改成仅训练集阈值。标签为气象极端 OR 时间事件，不按区域。
- S2 按节点表顺序构建 tensor，FLOW 求和、天气平均、标签 max；流量缺失填 0，天气按节点前后填充再用均值兜底。全部缺失的天气列可能仍为 NaN，Dataset 按原逻辑 nan_to_num。
- 训练月划分、训练历史输入的均值/总体标准差、标准差下限、加权 MAE、方向损失、Adam、梯度截断和批次均值方式均沿用原始实现。测试月份只被分离，原训练脚本不计算 test 指标。
- 训练反事实只在 EXTREME_FLAG==1 替换 raw 天气为 [训练 TEMP 均值, 0, 训练 WDSP 均值, 70]；两次预测的 x_future 始终不变。分析以 EXTREME_FLAG 与分析事件时间窗的交集替换 x_future_raw，x_future 仅 clone，无数值修改。
- raw 天气改变 penalty，也会通过 prev_flow 影响后续解码步骤，不能把输出差异理解为只有当步的直接惩罚差。
- 模型输出没有截断：原“防负数”注释不符合实际代码。分析反归一化后分别截断 real/cf/true 到非负；CF_SCALE 默认 1，仅作用事件窗口 h=0。没有私加截断或单调保证。
- 分析时间轴由月份第一日加 Tin 推导，**不读取 NPZ times**。S2 则以输入实际最小时间起始。月首缺小时会使分析时间错位；本次仅记录，未修正。
- 邻接矩阵默认已归一化且不添加自环。原代码未验证矩阵标签顺序与节点表一致；运行者必须预先核对。num_nodes 参数不改变模型权重结构。
- KMP_DUPLICATE_LIB_OK、旧版 torch.cuda.amp 调用及未使用的 y_true 参数保留。未来 PyTorch 兼容性未验证，requirements 是依赖范围，不是完整训练环境锁文件。

四个月份的两套窗口全部不同，见 [审计记录](docs/CODE_AUDIT.md)。不统一窗口，也不使用 NPZ EVENT_FLAG 替代分析窗口。

## 输入、输出与 OD

完整输入清单见 [INPUT_DATA.md](docs/INPUT_DATA.md)。缺少 checkpoint、norm JSON、样本 NPZ、月度原始交通/天气 CSV、节点顺序表及邻接矩阵，均未伪造。

S1 输出月度标签 CSV 与汇总；S2 输出 YYYYMM_samples.npz；训练输出最优 state_dict、归一化 JSON、loss NPZ/PNG；分析输出 national 结果 NPZ、影响汇总 CSV 和四张 PNG。输出文件名和字段保持原脚本格式。

与现有 od_estimation 的对接契约、缺失映射和方向问题见 [OD_INTEGRATION.md](docs/OD_INTEGRATION.md)。当前预测输出不能直接冒充 OD 的边输入。

## 验证与上传

本次 **5 项通过，1 项因缺 PyTorch/Matplotlib 跳过**；执行环境与范围见 [TEST_REPORT.md](docs/TEST_REPORT.md)。合成数据通过标签→NPZ 流程；没有执行真实训练、真实 checkpoint 推理、历史结果比较或完整复现。

把本目录内容作为仓库或已有仓库子目录提交即可。`.gitignore` 排除 data、artifacts、缓存与环境；数据和模型如需分享请另行管理。没有添加未经确认的许可证。
