#encoding=utf-8
import h5py
import numpy as np
from scipy.io import loadmat
'''
    作者：程文鑫 2024.12.12
    功能：根据提取熵值特征之后的矩阵，计算节点之间（特征矩阵的每一列）的对数欧式距离矩阵，并归一化，根据阈值得到（加权）无向图邻接矩阵,再根据图结构，根据对偶图原理构建对偶超图
'''
def load_ft(data_dir, lbls_dir):
    print(data_dir)
    print(lbls_dir)
    # 使用h5py打开文件
    with h5py.File(data_dir, 'r') as f:
        # 读取数据
        fts = f['Feature_ME'][:][:]
    y = loadmat(lbls_dir)
    lbls = y['y_label'][0]
    lbls_len = len(lbls)
    # 设置随机种子以获得可重复的结果
    np.random.seed(0)
    # 创建一个长度为lbls_len的随机向量，元素为0或1
    idx = np.random.choice([0, 1], size=lbls_len, p=[0.3, 0.7])
    idx_train = np.where(idx == 1)[0][:]
    idx_test = np.where(idx == 0)[0][:]
    return fts, lbls, idx_train, idx_test

def get_Log_Euclidean_Distance(fts):
    # 计算特征矩阵中的每一列（后续图结构中的节点）元素之间的对数欧氏距离
    dis = []
    tem = []
    nodes_num = fts.shape[1]
    print(nodes_num)
    for i in range(nodes_num):
        for j in range(i + 1, nodes_num):
            c = (np.log(fts[:, i]) - np.log(fts[:, j])) ** 2
            distemp = np.sqrt(np.sum(c))
            tem = [i, j, distemp]
            dis.append(tem)
    return dis

def constructing_Graph_Structure(nodes_num,dis):
    # dis矩阵为n(n-1)/2行3列的矩阵，其中n为图的节点数（EEG中的样本数），第一列和第二列分别为两两节点的下标，第三列为前两列节点之间的距离
    # 构建图结构：无向加权图的邻接矩阵,权值为对数欧氏距离
    graph_str = np.zeros((nodes_num, nodes_num))  # 对称矩阵，行数列数均为nodes_num，对角线上元素为0，其余位置上元素为两节点之间的距离值
    dis = np.array(dis)
    N = dis.shape[0]  # 计算dis矩阵的行数
    for i in range(N):
        x = int(dis[i][0])
        y = int(dis[i][1])
        graph_str[x][y] = dis[i][2]
        graph_str[y][x] = dis[i][2]
    #归一化处理，并设置阈值从而得到无向无权邻接矩阵
    #进行最大最小归一化
    min_val = np.min(graph_str)
    max_val = np.max(graph_str)
    normalized_graph = (graph_str - min_val) / (max_val - min_val)
    #设置阈值为0.5，得到01邻接矩阵
    for i in range(nodes_num):
        for j in range(nodes_num):
            if normalized_graph[i, j] < 0.4:
                normalized_graph[i, j] = 1
            else:
                normalized_graph[i, j] = 0
            if i == j:
                normalized_graph[i, j] = 0
    return graph_str, normalized_graph

def graph_to_hypergraph_incidence_matrix(adj_matrix):
    # Step 1: Identify all edges in the original graph
    num_nodes = adj_matrix.shape[0]
    edges = []
    edge_to_index = {}
    index = 0

    # Use a set to store unique edges (since the graph is undirected)
    edge_set = set()

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Only consider upper triangle to avoid duplicates
            if adj_matrix[i, j] != 0:
                edge = tuple(sorted((i, j)))  # Sort to ensure edge is represented consistently
                if edge not in edge_set:
                    edges.append(edge)
                    edge_to_index[edge] = index
                    index += 1
                    edge_set.add(edge)

    # Step 2: Create incidence matrix for the hypergraph
    num_edges = len(edges)
    incidence_matrix = np.zeros((num_edges, num_nodes), dtype=int)

    # Populate the incidence matrix
    for edge, edge_idx in edge_to_index.items():
        u, v = edge
        incidence_matrix[edge_idx, u] = 1
        incidence_matrix[edge_idx, v] = 1

    # Note: The incidence matrix represents which edges (hypergraph nodes) are incident to which nodes (hyperedges)
    # In this context, each row corresponds to an edge in the original graph (a node in the hypergraph)
    # and each column corresponds to a node in the original graph (a hyperedge in the hypergraph).
    # A 1 in the matrix indicates that the edge (hypergraph node) is incident to the node (hyperedge).

    # However, typically in hypergraph incidence matrices, we might want to represent hyperedges as rows
    # and nodes (or edges in the original graph) as columns, with entries indicating membership.
    # Since the problem statement is a bit ambiguous, I'll provide both interpretations.
    # For the interpretation where hyperedges are rows and edges are columns, we need to transpose:
    hypergraph_incidence_matrix_transposed = incidence_matrix.T

    # Output both for clarity (you can choose which one to use based on your definition of the incidence matrix)
    print("Incidence matrix (edges as rows, nodes as columns):")
    print(incidence_matrix)

    print("Incidence matrix (transposed, nodes as rows, edges as columns):")
    print(hypergraph_incidence_matrix_transposed)

    # If you only want the transposed version (which is more common for hypergraph incidence matrices):
    return hypergraph_incidence_matrix_transposed

def load_feature_construct_H(data_dir,lbls_dir):
    """
    :param data_dir: directory of feature data
    :param lbls_dir: directory of labels data
    :return:
    """
    print(data_dir)
    print(lbls_dir)
    #特征、标签获取，划分测试集和训练集
    fts, lbls, idx_train, idx_test = load_ft(data_dir, lbls_dir)
    #计算特征矩阵中的每一列（后续图结构中的节点）元素之间的对数欧氏距离,以构建图结构
    dis = get_Log_Euclidean_Distance(fts)
    #获取图结构的节点数（fts特征矩阵的列数）
    nodes_num = fts.shape[1]
    #构建图的邻接矩阵（无向加权，权值为对数欧氏距离）
    graph_str, normalized_graph = constructing_Graph_Structure(nodes_num,dis)
    print(graph_str.shape)
    print(normalized_graph.shape)
    #构建超图结构
    # 判断normalized_graph矩阵是否为对称矩阵
    if np.allclose(normalized_graph, normalized_graph.T):
        lower_triangle = np.tril(normalized_graph)
        non_zero_count = np.count_nonzero(lower_triangle)
    else:
        print("Error in constructing the graph adjacency matrix!")
    print(non_zero_count)
    H = graph_to_hypergraph_incidence_matrix(normalized_graph)
    if H is None:
        raise Exception('None feature to construct hypergraph incidence matrix!')
    fts = fts.T
    return fts, lbls, idx_train, idx_test, H
    #fts:182*1416  lbls:182*1     idx_train:127*1    idx_test:55*1     H:182*5985
    #fts:5234*64   lbls:5234*1    idx_train:4187*1   idx_test:1047*1   H:5234*10468


