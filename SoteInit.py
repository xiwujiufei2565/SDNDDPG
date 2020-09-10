# 处理流量数据

# 导入相关模块
import numpy as np
import copy
import sys
import warnings
warnings.filterwarnings("ignore")

WEIGHT_MAX = 10000
path = "F:\PythonProjects\ReinforcementLearning\SDNDDPG\src\Topology.txt"

# 读取数据流量矩阵文件
def load_traffic_matrix(path):
    # path---流量分配文件路径

    tm = np.loadtxt(path, delimiter=" ")
    return tm.tolist()

# 将流量请求转化为矩阵形式
def build_traffic_matrix(path, vertex_num):
    # path---流量分配文件路径     vertex_num---节点的个数

    tm = load_traffic_matrix(path)
    matrix_traffic = np.zeros([vertex_num, vertex_num], dtype='float32').tolist()
    for item in tm:
        in_v = int(item[0]-1)
        out_v = int(item[1]-1)
        traffic = item[2]
        matrix_traffic[in_v][out_v] = traffic
    return matrix_traffic

# 建立拓扑信息图结构和链路容量信息（邻接矩阵法）
def build_topology(vertex_num):
    # vertex_num---顶点数量

    vertex_num = vertex_num
    # 读取拓扑信息
    fw = np.loadtxt(path, delimiter=' ', dtype='int32')
    # print(type(fw))
    # 邻接表----23个顶点, 0下标使用，之后顶点标号要-1
    topology = np.zeros([vertex_num, vertex_num], dtype='int32')    # 权重
    link_capacity = np.zeros([vertex_num, vertex_num], dtype='float32')  # 链路容量

    # 遍历拓扑信息，存储到图结构中
    # temp[0]--源节点  temp[1]--目的节点   temp[2]--链路容量最大值    temp[3]--链路权重值
    for temp in fw:
        in_vertex = temp[0] - 1
        to_vertex = temp[1] - 1
        capacity = temp[2]
        weights = temp[3]
        topology[in_vertex, to_vertex] = weights
        link_capacity[in_vertex, to_vertex] = capacity
    # 返回邻接表, 链路容量表----列表结构
    return topology.tolist(), link_capacity.tolist()

# 使用迪杰斯特拉算法，计算所有节点到其他节点的的最短路径
def dijkstra_init(topology, vertex_num):
    # topology---网络的拓扑结构信息      vertex_num---节点的个数

    # 存储最短路径
    # 由于最短路径可能会有多条，所以用列表来进行存储
    # path[x][y]-----表示顶点x到y的最短路径需要经过的上一个节点，若有不同值则表示有多条路径
    path = [[[] for y in range(vertex_num)] for x in range(vertex_num)]

    # 计算所有节点到目的节点的最短路径
    # 若不存在则其值为0
    for source in range(vertex_num):
        cost = []  # 存储最短路径值,0表示未接通
        # 初始化cost和set数组,以及对应的path数组
        for dst in range(vertex_num):
            cost.append(topology[source][dst])
            if topology[source][dst] != 0:
                path[source][dst].append(source)

        set = [0 for x in range(vertex_num)]  # 顶点是否被访问过
        set[source] = 1

        for i in range(vertex_num):
            min = WEIGHT_MAX
            min_index = -1
            # 寻找最小顶点
            for j in range(vertex_num):
                if (set[j] != 1 and cost[j] != 0 and cost[j] < min):
                    min = cost[j]
                    min_index = j
            if min_index == -1:
                continue
            set[min_index] = 1  # 该节点访问过
            # 更新其他节点到达目的节点的最短路径
            for j in range(vertex_num):
                if set[j] != 1 and topology[min_index][j] != 0:
                    if cost[j] == 0 or topology[min_index][j] + cost[min_index] < cost[j]:
                        cost[j] = topology[min_index][j] + cost[min_index]
                        # 由于出现了别之前存储的路径更小的，所以之前不管有多条路径都要清空
                        path[source][j].clear()
                        path[source][j].append(min_index )
                    elif cost[j] != 0 and topology[min_index][j] + cost[min_index] == cost[j]:
                        # 如果相等，表示可能存在多条路径，所以需要加入到当前集合中
                        # 在初始化时，已经将source加入了
                        path[source][j].append(min_index)
    return path

# 根据最短路径，生成源节点到目的节点的经过的链路集合
def generate_link(source, dst, path):
    # source----源节点(记得减1) dst----目的节点 path----最短路径集合

    link_path = []  # 链路路径
    pre_list = copy.deepcopy(path[source][dst])    # 当前源节点到目的节点路径的前驱节点集合
    dst_list = [dst for i in pre_list]  # 每个前驱节点对应要到的目的节点
    source_num = pre_list.count(source) # 源节点是否存在前驱节点中
    while (source_num == 0 and len(pre_list) >= 1) or (source_num != 0 and len(pre_list) >= 1):
        # 情况1：如果路径的前驱不存在源节点，则说明源节点到目的节点需要经过其他链路路径
        # 情况2：如果路径的前驱节点存在源节点，但是其前驱节点集合不止该源节点，说明存在多条最短路径
        while source_num > 0:
            link_path.append((source, dst_list[pre_list.index(source)]))   # 情况2，需要生成链路----(1, 2)
            dst_list.pop(pre_list.index(source))    # 将该节点的目的节点也删除
            pre_list.remove(source)     # 将生成路径的节点移除
            source_num = source_num - 1
        temp_pre_list = []
        temp_dst_list = []
        for index in range(len(pre_list)):
            current_node = pre_list[index]
            link_path.append((pre_list[index], dst_list[index]))    # 将对应的源到目的节点的链路加入
            for temp in path[source][current_node]:
                temp_pre_list.append(temp)     # 将从源节点到达当前节点的前驱节点加入,可能会有多个列表前驱
                temp_dst_list.append(current_node)                   # 其要达到的节点就是当前节点
        # 进行下一次迭代
        pre_list = temp_pre_list
        dst_list = temp_dst_list
        source_num = pre_list.count(source)  # 源节点是否存在前驱节点中

    return link_path

# 根据tm流量请求，分配对所有链路进行流量分配
def flow_distribution(vertex_num):
    # vertex_num---节点的个数

    # 建立拓扑
    topology, link_capacity = build_topology(vertex_num)

    # 根据拓扑信息，建立每个节点到其他节点的最短路径
    shortest_path = dijkstra_init(topology, vertex_num)

    # 获取流量需求矩阵
    file_path = 'F:\PythonProjects\ReinforcementLearning\SDNDDPG\src\TM-2004-03-01-0000.txt'
    tm = load_traffic_matrix(file_path)

    # 创建链路分配矩阵，link_distribute[i][j]----i和j这条链路上的流量分配
    link_distribute = np.zeros((vertex_num, vertex_num)).tolist()
    for item in tm:
        from_vertex = int(item[0]) - 1  # 源节点 将float类型转化为int类型
        to_vertex = int(item[1]) - 1    # 目的节点，-1是因为从0开始
        flow_num = item[2]          # 流量请求量
        link_paths = generate_link(from_vertex, to_vertex, shortest_path) # 求取源节点到目的节点的链路
        link_num = len(link_paths)
        if link_num == 0:
            continue
        # 计算源节点到目的节点的出链路有几条，则流量均分为几份
        count = 0;
        for link_path in link_paths:
            if link_path[0] == from_vertex:
                count += 1
        flow_num = flow_num / count
        for link_path in link_paths:
            link_distribute[link_path[0]][link_path[1]] += flow_num # 每条链路加上新增的流量

    return link_distribute, topology, link_capacity  # 返回最大链路分配

# test
if __name__ == '__main__':

    # graph, link = build_topology(4)
    # print(graph)
    # path = dijkstra_init(graph, 4)
    # print(path)
    # link_path = generate_link(0, 3, path)
    # print(link_path)

    # link_distribute = flow_distribution(23)
    # link_distribute = np.array(link_distribute)
    # print(link_distribute[link_distribute != 0].shape)

    u = flow_distribution(23)
    print(u)

