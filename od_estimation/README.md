# 新西兰极端天气研究：OD 交通分配与反推

本仓库提供区域 OD 到节点 OD 的展开、MSA 用户均衡交通分配、区域 OD 迭代反推，以及 2019 年 12 月事件窗口的 real/cf 批量处理。公式、常数和输出行为沿用原研究脚本。

初始 OD 是预计算输入 `data/region15_od_seed_from_gdp.csv`。仓库不包含生成该初始 OD 的代码。

## 快速运行

需要 Python 3.10 或以上版本。先在解压后的仓库目录打开终端，安装依赖：

```console
python -m pip install -r requirements.txt
```

先复现已核对的单日案例（真实、反事实各一次）：

```console
python -m nz_od --dates 2019-12-02 --scenarios real cf
python verify.py
```

结果写入 `results/2019-12-02/real` 和 `results/2019-12-02/cf`。验证成功时显示 54 个历史 CSV 比较通过。运行可能需要几分钟；保留了原脚本逐条最短路的计算方式。

运行完整 7 天窗口（2019-12-02 至 2019-12-08，共 14 个案例）：

```console
python -m nz_od --output results_full
```

运行第二阶段 S9 基线反推：

```console
python -m nz_od --mode baseline --output results_baseline
```

每次使用空输出目录，避免覆盖已有结果。输入数据默认从仓库内 data/ 读取，也可用 `--data-dir` 指定其他目录。不需要原电脑的 D: 或 E: 盘路径。

## 文件结构

- `nz_od/`：分配与反推代码，以及命令行入口。
- `data/`：3 个公共输入 CSV，以及 201912/ 下的 14 个每日边输入 CSV。
- `reference_results/`：2019-12-02 real/cf 的 54 个原研究结果 CSV，用于独立比较。
- `verify.py`：检查随附文件的 SHA-256，并将运行结果逐文件与历史结果比较。
- `requirements.txt`：本次验证使用的依赖版本。
- `provenance.json`：原脚本相对路径与 SHA-256。
- `data_manifest.json`：输入数据、参考结果的文件清单与 SHA-256。
- `validation.json`：整理初版时完成的函数一致性与历史结果检查记录。

## 算法与行为

保留 BPR alpha=0.15、beta=4.0，容量下限 1e-6；MSA 步长 1/it，UE 最多 50 次，相对间隙阈值 1e-4；OD 外层最多 5 次，更新指数 0.20，因子范围 0.50–1.50，EPS=1e-6，每轮总量缩放，两个外层停止阈值均为 0.01。

保留原 DiGraph 的平行边覆盖行为、不可达 OD 跳过方式、缺失值处理、对角线置零以及运算顺序。默认日期、每个日期 real 然后 cf 的顺序均与原 Step3 一致。

逐轮输出包括节点 OD、边流量、UE 日志、区域因子、区域 OD，以及最终 OD、迭代汇总、批处理汇总；文件名、列和 UTF-8 BOM 格式沿用原脚本。批处理汇总中的路径取决于运行位置。最后一轮边流量对应该轮 OD 更新之前的分配，保持原行为。

## 验证范围

2019-12-02 real/cf 的全部 54 个 CSV 已与历史结果比较通过（rtol=atol=1e-10）；event、baseline 模块提取的函数体已与原脚本核对一致。其余 6 天与 baseline 尚未做完整历史结果重跑。完整窗口输入已随附，但不据此声称全部案例验证通过。

本仓库覆盖 OD 分配与反推，不包括后续对角线补全、MRIO 或经济动态模拟。S9 在这里特指第二阶段New/S9迭代.py，S6UE.py 为交通分配参考来源。

## 上传 GitHub

将本 README 所在目录中的全部文件和子目录上传到仓库根目录，包括 data/ 与 reference_results/。无需上传压缩包本身、运行生成的 results/、Python 缓存或整个原始研究项目。
