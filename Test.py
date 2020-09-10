import cplex
import numpy as np
from SDNDDPG import ConNeo4j
import os
from py2neo import Graph,Node,Relationship, NodeMatcher

def cplex_text():
    cplex_model = cplex.Cplex()
    cplex_model.read('exps.lp')
    cplex_model.set_results_stream(None)
    cplex_model.solve()
    # print(cplex_model.solution.get_objective_value())
    # print(cplex_model.solution.get_values())
    # print(cplex_model.solution.get_values('x4'))

def neo4j_text():
    path = "F:\PythonProjects\ReinforcementLearning\SDNDDPG\Topology.txt"
    fw = np.loadtxt(path, delimiter=' ', dtype='int32')
    neo_driver = ConNeo4j.NeoUtil()

    # 遍历拓扑信息，存储到图结构中
    # temp[0]--源节点  temp[1]--目的节点   temp[2]--链路容量最大值    temp[3]--链路权重值
    for temp in fw:
        neo_driver.add_link(int(temp[0]), int(temp[1]), int(temp[3]))

def sort_pre():
    output = np.array([0.2, 0.3, 0.4, 1.48, 1.2, 0.9])
    output[np.argpartition(output, 3)[:][-3:]] = 1.0
    output[np.argpartition(output, 3)[:][0:-3]] = 0.0
    print(output)

def sort_dict():
    chroms = []
    chroms.append(dict(s=1, u=10))
    chroms.append(dict(s=2, u=20))
    chroms.sort(key=lambda x: x['u'], reverse=True)
    print(chroms)

def rename_file():
    path = ""
    fileList = os.listdir(path)
    # 修改文件名
    for filename in fileList:
        index = filename.find("")
        if index != -1:
            old_name = path + "\\" + filename
            new_name = path + "\\" + filename[10:]
            os.rename(old_name, new_name)



if __name__ == '__main__':
    rename_file()
