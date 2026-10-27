# 与现有 od_estimation 对接

已读取本地现有 od_estimation/README.md 及 nz_od/event.py 的输入校验和批处理命名。它要求 data/201912/edge_input_{real|cf}_{YYYY-MM-DD}.csv，每日每场景一份；字段为 EdgeID, FromNode, ToNode, AB_FF_Time, BA_FF_Time, AB_Capacity, BA_Capacity, AB_Count, BA_Count。日期默认 2019-12-02 至 12-08。其结果是迭代 OD/边分配；本预测包提供的是对接到它**输入端**的候选交通量，不直接替代其反推输出。

预测分析 NPZ 中 pred_real/pred_cf/true 为非负、反归一化后的 (S,12,N,1)；time_axis 为每个样本 h=0 的时间。原 NPZ 没有另存 nodes，必须从配套样本 NPZ 取 nodes，并核验其与模型及邻接的对应关系。h=k 对应 time_axis[s]+k 小时。

建议经业务确认的独立转换阶段按以下契约实现：

1. 只选 h=0，得到 time_axis×SITEREF 的 real/cf 小时预测。不要把全部 12 步叠加，同一目标小时会被重叠预测多次。
2. 提供经过核验的 SITEREF→EdgeID→AB/BA 方向映射及多站点同边处理规则。当前 FLOW 没有方向维，不能擅自均分到 AB/BA 或同时复制给两个方向。
3. 确认 FLOW 单位、交通量统计口径和时间区，按本地自然日聚合；检查每天完整 24 小时、站点覆盖和月首历史窗口造成的缺小时。不要悄悄补齐或乘 24。
4. 使用既有道路拓扑/自由流时间/容量表，保持 EdgeID、FromNode、ToNode、AB/BA_FF_Time、AB/BA_Capacity；仅按已批准映射写入对应 AB_Count/BA_Count。缺方向与缺站点使用既有 OD 的缺失口径，必须留记录。
5. 输出上述日文件，与 od_estimation 既有日期目录对应，再运行 python -m nz_od。分别保留 real/cf 与转换清单，核查计数总量和可达性。

本次缺少站点到路段/方向映射、原始单位说明和中间转换脚本。因此只交付可审查的契约，未编造可执行转换器，也未声称与既有 OD 日输入数值一致。

特别注意：预测 201912 分析窗是 12-01 至 12-08，OD 默认 12-02 至 12-08；下游应显式选择日期。全国汇总 CSV 不能用来恢复边方向 Count。SITEREF 是观测站编号，不可直接当作 OD NodeID。
