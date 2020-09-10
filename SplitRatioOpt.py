# 导入相关模块
import numpy as np
import random
from SDNDDPG import SoteInit
from SDNDDPG import ConNeo4j
import copy
import sys
import warnings
import cplex
warnings.filterwarnings("ignore")

# 启发式搜索一个新的分布式链路权重设置
def neighbor_search(topology, vertex_num, ratio):
    # topology---网络带权重的拓扑     vertex_num---节点的个数
    # ratio---需要随机选取节点的比率

    # 先将所有节点到达其他节点的最短路径计算出来
    total_cost = dijkstra(topology, vertex_num)

    p_size = int(vertex_num * ratio)    # 计算P集合的大小
    P = []
    for i in range(p_size):
        # 随机获取源节点到目的节点的集合
        source = random.randint(0, vertex_num - 1)  # 顶点下标从0开始
        dst = random.randint(0, vertex_num - 1)
        while dst == source:
            dst = random.randint(0, vertex_num - 1) # 如果源节点和目的节点相同，则重新选取
        P.append((source, dst)) # 将生成的节点对存入集合中

    # 遍历集合P
    # node[0]---source  node[1]---dst
    for node in P:
        source = node[0]    # 源节点
        dst = node[1]       # 目的节点
        S = []  # source的邻节点集合
        for i in range(len(topology[source])):
            if topology[source][i] != 0:
                S.append(i) # 如果source到节点i有路径，则i为其中一个邻节点

        # 计算节点si到目的节点dst之间的最短路径的链路权重值di
        max_distance = 0
        for si in S:
            if total_cost[si][dst] > max_distance:
                max_distance = total_cost[si][dst]
        if max_distance == 0:
            # 表示source的所有邻节点si到目的节点dst都没有路径
            continue
        # 重新设置链路权重
        for si in S:
            # 由于计算完了全部最短路径，所以直接更新
            topology[source][si] = max_distance + 1 - total_cost[si][dst]

    return topology # 返回新的拓扑结构

# 使用迪杰斯特拉计算所有节点之间的最短路径值
def dijkstra(topology, vertex_num):
    # topology---网络带权重的拓扑     vertex_num---节点的个数

    WEIGHT_MAX = 1000000
    # 计算所有节点到目的节点的最短路径
    # 若不存在则其值为0
    total_cost = []
    for source in range(vertex_num):
        cost = []  # 存储最短路径值,0表示未接通
        # 初始化cost和set数组
        for dst in range(vertex_num):
            cost.append(topology[source][dst])
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

        total_cost.append(cost) # 将每个节点的cost存入
    return total_cost

# 对于每个目的节点，利用迪杰斯特拉算法构造所有节点到目的节点的有向无环图DAG
def build_DAG(topology, vertex_num, dst):
    # topology---网络拓扑图  vertex_num---网络顶点个数   dst---目的节点

    # 使用邻接矩阵存储dag图
    DAG = np.zeros([vertex_num, vertex_num], dtype='int32').tolist()    # 初始化DAG图
    path = SoteInit.dijkstra_init(topology, vertex_num)    # 使用迪杰斯特拉算法计算所有节点到其他节点的路径
    # 每个节点到目的节点最短路径需要经过的节点（为加入SDN出链路做准备）    PS：其实可以搜索DAG图的，但是考虑时间复杂度，算了
    shortest_lists = []     # shortest_list[i]表示节点i到目的节点需要经过的节点集合

    # 生成每个节点到目的节点的DAG图
    for source in range(vertex_num):
        shortest_list = []  # 记录当前节点source到dst节点最短路径上的节点
        if source != dst:
            # 除去目的节点到目的节点的情况
            link_path = SoteInit.generate_link(source, dst, path)
            # print(link_path)
            # 根据节点到目的节点的链路信息，建立DAG图
            for item in link_path:
                DAG[item[0]][item[1]] = 1   # 1表示连通，0表示不连通
                # 将路径上的节点加入
                if shortest_list.count(item[0]) == 0:
                    shortest_list.append(item[0])
                if shortest_list.count(item[1]) == 0:
                    shortest_list.append(item[1])
        shortest_lists.append(shortest_list)    # 将其加入到大集合中

    return DAG, shortest_lists

# 将满足条件的SDN节点的出链路加入到DAG中
def add_DAG(topology, DAG, sdn, vertex_num, shortest_lists):
    # topology---网络拓扑结构     DAG---某一个目的节点构建的DAG图   sdn---SDN节点的位置序列
    # vertex_num---网络节点的数量     shortest_lists---其他节点到目的节点的最短路径的节点集合

    for index in range(vertex_num):
        if sdn[index] == 0:
            continue    # 不是SDN节点则遍历下一个

        for out_node in range(vertex_num):
            # 遍历SDN节点的出链路
            if topology[index][out_node] == 0:
                continue    # 不存在链路
            # 判断SDN节点和普通节点是否具有偏序关系，或者不可比
            if out_node in shortest_lists[index] or (out_node not in shortest_lists[index]
                                                     and index not in shortest_lists[out_node]):
                if sdn[out_node] != 1:  # 两个不能都是SDN
                    # print("(" + str(index) + ", " + str(out_node) + ")")
                    DAG[index][out_node] = 1    # 将满足条件的出链路加入到DAG中

    return DAG

# 对一个有向无环图DAG进行拓扑排序
def topological_sort(DAG, vertex_num):
    # DAG---有向无环图   vertex_num---节点的数量

    DAG_temp = copy.deepcopy(DAG)   # 因为要修改
    set = np.zeros(vertex_num, dtype='int32').tolist()  # 判断节点是否被访问过
    vertex_order = []   # 经过拓扑排序的顺序
    count = 0   # 设置找到顶点的次数
    while count < vertex_num:
        # 寻找下一个没有前驱的节点
        for i in range(vertex_num):
            if set[i] == 1:
                continue  # 当前节点访问过

            flag = True  # 设置是否具有前驱节点的标志
            for j in range(vertex_num):
                if DAG_temp[j][i] == 1:
                    flag = False  # 该节点有前驱节点
                    break
            if flag == False:
                continue  # 遍历下一个节点

            set[i] = 1  # 设置当前节点访问过当前节点访问
            vertex_order.append(i)  # 将当前节点加入
            for j in range(vertex_num):
                # 将加入节点的出边设置为0
                DAG_temp[i][j] = 0
            break
        count += 1
    # if 0 in set:
    #     print('Error：存在环路, 错误DAG图如下：')
    #     print(DAG)
    #     sys.exit(1)

    return vertex_order

# 创建链路表达式集合
def build_link_exps(vertex_num):
    # vertex_num---节点的个数

    # 初始化每条链路上的约束表达式
    # exps[i][j] = {variable:{'x1': 1, 'x3': 2, ...}, constant=28000}
    # 表示链路ij上的情况
    link_exps = []
    for i in range(vertex_num):
        temp_exp = []
        for j in range(vertex_num):
            temp_exp.append(dict(variable={}, constant=0.0))
        link_exps.append(temp_exp)
    return link_exps

# 将一个DAG生成的链路约束表达式添加到总的链路表达式中
def add_link_exps(link_exps, link_exp):
    # link_exps---总的链路约束表达式     link_exp---一个DAG图产生的约束表达式

    for i in range(len(link_exp)):
        for j in range(len(link_exp[i])):
            # 变量合并
            for key in link_exp[i][j]['variable'].keys():
                if link_exps[i][j]['variable'].get(key) == None:
                    link_exps[i][j]['variable'][key] = link_exp[i][j]['variable'][key]  # 不存在该变量，则创建并赋值
                else:
                    link_exps[i][j]['variable'][key] += link_exp[i][j]['variable'][key]  # 存在变量
            # 常量合并
            link_exps[i][j]['constant'] += link_exp[i][j]['constant']

# 使用Cplex解决这个线性规划问题
def cplex_solve(link_exps, global_exps, link_capacity, vertex_num):
    # link_exps---链路上的约束信息      global_exps---SDN节点上的约束信息
    # link_capacity---链路的容量信息

    # 将link_expsh和global_exps生成.lp文件
    file_name = build_cplex_file(link_exps, global_exps, link_capacity)
    # 求解生成的lp文件
    cplex_model = cplex.Cplex()
    cplex_model.set_results_stream(None)
    cplex_model.read(file_name)
    cplex_model.solve()
    u = cplex_model.solution.get_objective_value()  # 最大链路利用率
    # 根据变量的值，计算流量分配矩阵
    link_distribute = np.zeros([vertex_num, vertex_num]).tolist()
    for i in range(vertex_num):
        for j in range(vertex_num):
            num = 0.0
            for key, value in link_exps[i][j]['variable'].items():
                num += cplex_model.solution.get_values(key) * value
            num += link_exps[i][j]['constant'] # 加上常量部分的值
            link_distribute[i][j] = num
    return link_distribute, u

# 将link_expsh和global_exps生成.lp文件
def build_cplex_file(link_exps, global_exps, link_capacity):
    # link_exps---链路上的约束信息      global_exps---SDN节点上的约束信息
    # link_capacity---链路的容量信息

    file_name = "exps.lp"
    restrict_num = 1    # 约束条件的个数
    with open(file_name, "w+") as f:
        # 目标
        f.write("Minimize" + "\n")
        f.write("obj: " + "U" + "\n")
        # 约束
        f.write("Subject To" + "\n")
        # 链路约束
        for i in range(len(link_exps)):
            for j in range(len(link_exps[i])):
                if link_capacity[i][j] != 0:
                    # 约束索引
                    exp = 'r' + str(restrict_num) + ': '    # 生成的约束表达式
                    restrict_num += 1
                    # 约束表达式的变量部分
                    for key, value in link_exps[i][j]['variable'].items():
                        exp = exp + str(value) + ' ' + key + ' + '
                    # 约束表达式的链路容部分
                    exp = exp.rstrip(' + ')
                    exp = exp + ' - ' + str(link_capacity[i][j]) + ' U'
                    # 约束表达式的常量部分
                    exp = exp + ' <=' + ' - ' + str(link_exps[i][j]['constant'])
                    # 写入文件
                    f.write(exp + '\n')
        # 全局约束
        for item in global_exps:
            f.write('r' + str(restrict_num) + ': ' + item + '\n')
            restrict_num += 1
        # 结束标志
        f.write('End')
    return file_name

# 根据一个DAG图、其节点的拓扑排序、流量需求矩阵，建立其约束表达式
def route_flow(DAG, tm, sdn, dst, vertex_num, global_exps, var_num):
    # DAG---构建的最短路径有向无环图
    # tm---流量需求矩阵   sdn---SDN节点序列   dst---当前DAG图要到的目的节点
    # vertex_num---节点的个数    exps---每条链路上的约束     var_num---当前创建到的变量数量
    # global_exps---变量的约束表达式集合

    # 使用拓扑排序对DAG图节点进行优先关系的排序
    vertex_order = topological_sort(DAG, vertex_num)

    # 创建每条链路上的约束表达式集合
    link_exps = build_link_exps(vertex_num)

    # 按照拓扑排序结果进行建立约束表达式
    for v in vertex_order:
        if v == dst:
            continue    # 当前节点和目的节点一样，不需要建立

        # 当前节点入链路的信息收集
        in_vertex_exp = dict(variable={}, constant=0.0)
        for i in range(vertex_num):
            if i == dst:
                continue
            if DAG[i][v] == 0:
                continue  # 节点i到节点v之间不存在链路
            # 将其变量进行合并
            for key, value in link_exps[i][v]['variable'].items():
                if in_vertex_exp['variable'].get(key) == None:
                    in_vertex_exp['variable'][key] = value  # 不存在则直接新建把值复制
                else:
                    in_vertex_exp['variable'][key] += value  # 存在则相加
            # 将其常量进行合并
            in_vertex_exp['constant'] += link_exps[i][v]['constant']

        # 当前节点的出链路信息收集
        out_link = []
        for i in range(vertex_num):
            if DAG[v][i] == 1:
                # 当前节点到该节点有出链路
                out_link.append(i)
        split_num = len(out_link)  # 由于为分布式节点，该节点将流量进行均分，其均分到每条出链路上
        if split_num == 0:
            print(dst)
            print(v)
            print(sdn)
            print(vertex_order)
            neo_driver = ConNeo4j.NeoUtil()
            neo_driver.show_DAG(DAG, vertex_num)
            sys.exit()

        if sdn[v] == 0:
            # 当前节点为非SDN的分布式节点
            # 根据分布式节点的分流规则---流出等于流入的流量均分到出链路上
            in_vertex_exp['constant'] = (in_vertex_exp['constant'] + tm[v][dst]) / split_num
            for key in in_vertex_exp['variable'].keys():
                in_vertex_exp['variable'][key] = in_vertex_exp['variable'][key] / split_num
            # 更新节点v的出链路
            for out_v in out_link:
                link_exps[v][out_v]['constant'] += in_vertex_exp['constant'] # 修改该链路上的常量值
                # 修改该链路上的变量的值
                for key in in_vertex_exp['variable'].keys():
                    if link_exps[v][out_v]['variable'].get(key) == None:
                        link_exps[v][out_v]['variable'][key] = in_vertex_exp['variable'][key]    # 不存在该变量，则创建并赋值
                    else:
                        link_exps[v][out_v]['variable'][key] += in_vertex_exp['variable'][key]   # 存在变量，直接相加
        else:
            # 当前节点为SDN集中式节点
            # 根据集中式节点的分流规则---可以任意分流，所以需要为其创建变量和约束
            exps_str = ''   # 变量的约束
            # 等式左边
            for out_v in out_link:
                key = 'x' + str(var_num)
                var_num += 1    # 总的变量数加1
                link_exps[v][out_v]['variable'][key] = 1    # 创建新变量
                exps_str = exps_str + '+' + key
            exps_str = exps_str.lstrip('+')
            exps_str += '-'
            # 等式右边
            for key, value in in_vertex_exp['variable'].items():
                exps_str = exps_str + str(value)  + key + '-'
            exps_str = exps_str.rstrip('-')
            exps_str += '='
            # if(len(out_link) == 0):
            #     print(dst)
            #     print(sdn)
            #     print(vertex_order)
            #     neo_driver = ConNeo4j.NeoUtil()
            #     neo_driver.show_DAG(DAG, vertex_num)
            #     sys.exit()

            exps_str += str(in_vertex_exp['constant'] + tm[v][dst]) # 流量矩阵
            global_exps.append(exps_str)    # 将该约束添加到全局约束中
    return var_num, link_exps

# 计算SDN节点的分流比和网络的最小链路利用率
def split_ratio_opt(topology, vertex_num, sdn, tm, link_capacity):
    # topology---网络拓扑图  vertex_num---网络节点的数量   sdn---SDN节点的位置序列
    # link_capacity---链路的容量信息

    # 全局约束表达式
    global_exps = []

    # 初始化每条链路上的约束表达式
    link_exps = build_link_exps(vertex_num)

    # 创建的变量数量
    var_num = 1

    # print(exps)
    # 以每个节点作为目的节点进行遍历
    for v in range(vertex_num):
        # 构建目的节点为v的DAG图
        DAG, shortest_lists= build_DAG(topology, vertex_num, v)
        # 将集中式SDN节点满足条件的出链路加入DAG中
        DAG = add_DAG(topology, DAG, sdn, vertex_num, shortest_lists)
        # print(shortest_lists)
        # 计算链路约束表达式和全局约束表达式
        var_num, link_exp = route_flow(DAG, tm, sdn, v, vertex_num, global_exps, var_num)
        add_link_exps(link_exps, link_exp)
        # print('*' * 50)
        # print(DAG)
        # print(link_exps)
        # print(global_exps)
        # print('*'*50)

    # 调用Cplex API求解该线性规划问题
    link_distribute, min_u = cplex_solve(link_exps, global_exps, link_capacity, vertex_num)
    return link_distribute, min_u

# 测试
if __name__ == '__main__':
    vertex = 4
    topology, link = SoteInit.build_topology(vertex)
    # print(topology)
    # topology = neighbor_search(topology, vertex, 0.8)
    # print(topology)
    DAG, shortest_lists= build_DAG(topology, vertex, 3)
    # print(DAG)
    # print(shortest_lists)
    sdn = [0, 0, 1, 0]
    DAG = add_DAG(topology, DAG, sdn, vertex, shortest_lists)
    path = 'F:\PythonProjects\ReinforcementLearning\SDNDDPG\datatm.txt'
    tm = SoteInit.build_traffic_matrix(path, vertex)
    print(tm)
    # print(DAG)
    U = split_ratio_opt(topology, vertex, sdn, tm, link)
    print(U)
    # print(link_exps)
    # print(global_exps)

