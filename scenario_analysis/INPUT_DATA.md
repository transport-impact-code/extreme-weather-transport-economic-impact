# 输入数据与接口

默认 BASE_DIR=data_root，情景根为 data_root/第五阶段情景分析new/I_H0。配置 paths_by_stage 支持按阶段覆盖路径；null不覆盖，路径均相对命令运行目录。不要把D/E盘机器路径写入公开配置。

|阶段/入口|必要输入|输出/下一步|
|---|---|---|
|S1|SAMPLE_ROOT/{month}_samples.npz；ADJ_PATH；CKPT_PATH；NORM_PATH；相同model.py|OUT_ROOT/H0/I*/{month}_national_multihazard_event_results.npz|
|S2|S1_ROOT/H0/I*/同名NPZ：pred_real/pred_cf/true，future_time_axis或time_axis及horizon|每日/小时节点流量CSV及汇总|
|S3|S2每日CSV、NODE_EDGE_MAP_CSV、EDGE_OBS_CSV、EDGE_TEMPLATE_CSV|各日real/cf边输入|
|S4|S3边输入、ACCESS_CSV、INITIAL_REGION_OD_CSV|各日real/cf/region_od_final.csv及UE迭代诊断|
|S5|S4区域OD、REGION_ORDER_CSV|补对角线区域OD，225行/矩阵|
|S6|S5 real/cf OD|区域与OD容量因子|
|S7|S6日矩阵/因子|日期/区域顺序、regional_transport_factor_TT.npy、OD数组|
|S8|S7数组/索引；MRIO_PATH_CANDIDATES|I*/MRIO模型：285×285 Z、285×15 F、1×285 VA/X、19×15×365 VA_TT|
|order|S8上述目录、同一原始MRIO|I*/MRIO模型_sector_major_checked及排列映射|
|inventory_level1|AES_PATH、S8的sector_names.csv候选|库存/sector_stock_days_AES2019_middle/sector_stock_days_for_model.csv|
|S9|order输出、上述Level1库存CSV|365期动态经济结果、行业/区域损失、诊断|
|inventory（旧分支）|AES2019 Closing stocks、同一MRIO|sector_stock_days_AES2019_30/sector_stock_days_for_model.csv|
|inventory_variants（旧分支）|前一行输出|独立low/middle/high目录，未自动接入默认S9|

详细路径键、函数列表与原始字面量参数见 docs/source_audit.json，原项目机器根路径已脱敏。

S1原模型输入：Tin=24，Tout=12，130节点；历史交通/天气、未来天气、Y、EXTREME_FLAG。天气raw顺序TEMP、PRCP、WDSP、RH。CF在事件且极端标记位置采用正常raw；I1/I2/I3采用 normal+k*(observed-normal)，k=1/2/3，标准化未来天气输入不变。保持降水/风速非负及RH 0–100约束。

官方窗口（端点包含）：2019-03-23 00:00至03-27 23:00；08-10至08-14；11-19至11-22；12-01至12-08。现有预测样本不覆盖12-01，实际21天。新输入若覆盖22天，不能使用注释中的21天断言截断数据。

重排前核对MRIO全部地区/行业标签，不能只看形状。本次MRIO行/列活动标签一致，region-major全网格成立；重排后收支最大绝对误差约2.18e-11。每个强度共用同一MRIO，VA_TT维度为行业×地区×时间。

AES Level1库存中Owner-Occupied Property Operation无匹配项，由原程序用中位数填补；合并行业与AES Level1之间采用原关键词/模糊映射，并非新增精细行业拆分。mapping_review.csv必须结合研究口径复核。

包中不含原始数据、权重或完整历史输出。data_manifest.json记录本地验证所用输入的项目相对路径及SHA-256，供持有数据者核对，不代表已获得公开再分发许可。旧H1目录数据若复用，应复制到本地测试目录并将目录标签映射H0；本次验证采用此方式，未修改原D盘项目。
