# 输入数据清单

本次只收到六个 .py 文件；下列研究资产均未收到，ZIP 不包含这些数据。

| 阶段 | 所需输入 | 必需结构 |
|---|---|---|
| label | data/weather_flow_month/YYYYMM*.csv | DATETIME；后续需要 SITEREF、FLOW；气象 TEMP/PRCP/WDSP/RH，MXSPD 可选。文件名前六位为月份，分位数在所有匹配文件上计算 |
| samples | 月度 *_extreme_multihazard_event_v1.csv | SITEREF, DATETIME, FLOW,TEMP,PRCP,WDSP,RH,EXTREME_FLAG；六个子标签缺失按原逻辑填零 |
| samples | data/site_locations_130.csv | SITEREF，一行一节点，转字符串、strip、zfill(8) 后定义节点顺序，不能含重复节点 |
| train/analyze | data/adjacency_distance_130x130_exp.csv | 带索引列的 N×N 数值邻接矩阵；行列必须与节点顺序一致。默认 N=130，ADJ_ALREADY_NORMALIZED=true |
| train | artifacts/samples/YYYYMM_samples.npz | X_hist:(S,24,N,5), X_future:(S,12,N,4), Y:(S,12,N,1), EXTREME_FLAG:(S,12,N,1) |
| analyze | 同上，指定单月 NPZ | 同上；RAIN/WIND/COLD/SNOW_ICE/WEATHER_EXTREME/EVENT_FLAG 可选；S2 同时存 nodes:(N,), times:(T,) |
| analyze | stgnn_weather_multihazard_event_v1_24to12.pth | 与 model.py 对应的 state_dict；原脚本 torch.load，仅使用可信文件 |
| analyze | norm_stats_multihazard_event_v1.json | mean 和 std 各 5 个值，顺序 FLOW,TEMP,PRCP,WDSP,RH，必须与 checkpoint 配套 |

历史特征顺序 FLOW,TEMP,PRCP,WDSP,RH；未来特征 TEMP,PRCP,WDSP,RH；全部 tensor 最后维度顺序不可调换。原始气象单位未在脚本中给出转换依据，必须沿用原始数据单位，不可自行假定或换算。

S=T-24-12+1。默认训练集为其余月份，验证集为 201304/201305/201306/201309，测试集为 201903/201908/201911/201912；三部分都必须非空。节点顺序与邻接矩阵标签、训练和推理节点数必须一致。

训练可从齐全 NPZ 重新生成 checkpoint 和 norm，但这不等于拿到了历史配套资产，也不代表复现了原实验。
