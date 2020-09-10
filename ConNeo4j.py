# 导入相关依赖
from py2neo import Graph,Node,Relationship, NodeMatcher

# 将图信息存储到Neo4j图数据库中存储
class NeoUtil(object):
    def __init__(self):
        self.neo_graph = Graph()    # 创建连接
        self.matcher = NodeMatcher(self.neo_graph)

    # 为图添加节点以及节点之间的链路
    def add_link(self, node1, node2, weight):
        node1 = int(node1)
        node2 = int(node2)
        weight = int(weight)
        node_1 = self.matcher.match("Node", name=str(node1)).first()
        if node_1 == None:
            node_1 = Node("Node", name=str(node1))  # 不存在则创建新节点
            self.neo_graph.create(node_1)

        node_2 = self.matcher.match("Node", name=str(node2)).first()
        if node_2 == None:
            node_2 = Node("Node", name=str(node2))  # 不存在则创建新节点
            self.neo_graph.create(node_2)
        node_1_call_node_2 = Relationship(node_1, 'CON', node_2)
        node_1_call_node_2['weight'] = weight
        self.neo_graph.create(node_1_call_node_2)

    # 将DAG图添加到图数据库中
    def show_DAG(self, DAG, vertex_num):
        for i in range(vertex_num):
            for j in range(vertex_num):
                if DAG[i][j] == 1:
                    # 存在链路
                    self.add_link(i + 1, j + 1, 1)