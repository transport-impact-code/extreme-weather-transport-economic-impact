# 与仓库已有模块的关系

本次检查的是本机既有traffic_prediction与od_estimation交付源码，不是远端GitHub实时内容。未进行网络上传或修改。

model.py与traffic_prediction/traffic_prediction/model.py字节完全相同，故入口可按SHA-256相等条件优先复用相邻共享模型，缺失或不同则使用审计副本（显式指定不同文件时拒绝）。S1强度构造仍使用原情景脚本，不调用普通反事实命令冒充。

S4与od_estimation/nz_od/event.py中以下六个同名函数AST完全相同：ensure_required_columns、build_graph、aon_load、calc_tstt、expand_region_to_node_od、global_scale_region_od。这里不以此声称完整等价。

bpr_time默认值表达形式不同（变量与0.15/4常数）；run_ue存在关键字/位置参数、显式BPR参数与标量输出类型差异。update_region_od存在实质边界差异：S4在分组OD权重总和非正时使用普通均值回退，而既有event.py直接np.average。批处理与输出配置也不同。保留S4，避免在整理时改变边界算法。

优先复用既有接入节点、种子OD、边模板及模型资产，通过config映射。数据接口、地区顺序及日期范围须核对；已有OD十二月默认12-02起，不改写情景官方12-01起的窗口。
