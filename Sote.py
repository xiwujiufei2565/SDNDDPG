# 导入相关依赖
import numpy as np
from SDNDDPG import SoteInit
from SDNDDPG import SplitRatioOpt
import warnings
warnings.filterwarnings("ignore")

# 定义文件路径常量
tm_path = 'F:\PythonProjects\ReinforcementLearning\SDNDDPG\src\TM-2004-03-01-0000.txt' # 流量矩阵信息

# 根据当前链路分配情况，计算最大链路利用率
def get_max_link_utilization(link_distribute, link_capacity, vertex_num):
    u = 0
    for i in range(vertex_num):
        for j in range(vertex_num):
            if link_capacity[i][j] != 0:
                temp = link_distribute[i][j] / link_capacity[i][j]
                if temp > u:
                    u = temp
    return u

# Sote算法计算当前网络的最大链路利用率
def Sote(vertex_num, sdn, max_iteration):
    # vertex_num---节点的个数    sdn---SDN节点的位置序列
    # max_iteration---最大迭代次数

    # 建立流量分配矩阵
    tm = SoteInit.build_traffic_matrix(tm_path, vertex_num)
    # 初始化拓扑信息
    topology, link_capacity = SoteInit.build_topology(vertex_num)
    # 计算该链路权重下的最大链路利用率
    link_distribute_current, U_current = SplitRatioOpt.split_ratio_opt(topology, vertex_num, sdn, tm, link_capacity)

    # 以下为优化权重部分
    # # 初始化权重矩阵信息
    # link_distribute_best, topology, link_capacity = SoteInit.flow_distribution(vertex_num)
    # # 计算初始的最大链路利用率
    # U_best = get_max_link_utilization(link_distribute_best, link_capacity, vertex_num)
    # # 初始化迭代次数
    # iteration_times = 1
    # while iteration_times <= max_iteration:
    #     # 建立流量分配矩阵
    #     tm = SoteInit.build_traffic_matrix(path, vertex_num)
    #     # 计算该链路权重下的最大链路利用率
    #     link_distribute_current, U_current = SplitRatioOpt.split_ratio_opt(topology, vertex_num, sdn, tm, link_capacity)
    #     # 寻找全局最小的最大链路利用率
    #     if U_current < U_best:
    #         link_distribute_best = link_distribute_current
    #         U_best = U_current
    #     # print("U_current: " + str(U_current) + " U_best: " + str(U_best))
    #     iteration_times += 1

    return link_distribute_current, U_current   # 返回最大链路利用率

# 获取初始化状态信息---即网络初始化的链路流量分配及最大链路利用率
def get_state_init(vertex_num):
    # 初始化权重矩阵信息
    link_distribute, topology_best, link_capacity = SoteInit.flow_distribution(vertex_num)
    # 计算初始的最大链路利用率
    U_best = get_max_link_utilization(link_distribute, link_capacity, vertex_num)
    link_distribute = np.array(link_distribute)
    U_best = np.array(U_best)
    return link_distribute, U_best

# 根据sdn的部署序列，获取状态（在当前sdn部署序列下获取最大链路利用率的链路流量分配）以及最大链路利用率
def get_state(a_t):
    link_distribute_best, U_best = Sote(len(a_t), a_t, 2)
    link_distribute_best = np.array(link_distribute_best)
    link_distribute_best = np.ravel(link_distribute_best)
    U_best = np.array(U_best)
    return link_distribute_best, U_best

# 根据初始的最大链路利用率和加入SDN序列得到的最大链路利用率计算reward
def get_reward(u_1, u_t):
    r = 0.0
    alpha = u_1 / u_t
    if alpha < 1:
        r = - np.e**(2.0 * (1.0 / alpha - 1.0))
    elif alpha > 1:
        r = np.e**(2.0 * (alpha - 1.0))
    return r


