
import numpy as np
import torch
from torch.utils.data import Dataset

# 划分训练集与验证集
def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

# 对于输入数据（train_data, valid_data, test_data)，进行处理
def handle_data(inputData,train_len=None):
    # 求出会话最大长度max_len
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    us_pois = [list(upois) + [0] * (max_len - le) for upois, le in zip(inputData, len_data)]
    # 反转序列并补0，使得每个会话的长度相同。反转序列使得最近点击的item位于会话的最前面。
    # us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
    #        for upois, le in zip(inputData, len_data)]

    us_msks = [[1] * (le) + [0] * (max_len - le) for le in len_data]
    # 返回反转并补0后的会话序列；us_msks是mask，有点击的为1，补0的为0；
    return us_pois, us_msks, max_len


# 这个函数在main.py中调用
# 输入分别是adj:item的邻居,num_node:item的总个数，opt.n_sample_all:邻居个数，num:邻居出现的次数
def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    # 为所有的item建立一个邻居关系的矩阵adj_entity及出现次数的矩阵num_entity
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    # 对于每个item，取出其对应的邻居neighbor和出现次数neighbor_weight
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        # n_neighbor表示该item的邻居个数（因为很多item没有opt.n_sample_all个邻居）
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        # 由此有了item的邻居矩阵adj_entity和对应出现的次数num_entity
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity

# 继承Dataset类
class Data(Dataset):
    def __init__(self, data, opt, shuffle=False, graph=None):
        inputs = data[0]
        # cut_inputs = []
        # for input_ in inputs:
        #     cut_inputs.append(input_[-opt.cutnum:])
        inputs, mask, len_max = handle_data(inputs)
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def __getitem__(self, index):
        # 根据索引index的值取出对应的会话序列u_input, mask, 以及target
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]
        # max_n_node存储会话的最大长度，node是对输入会话进行去重，items对node进行补0
        max_n_node = self.len_max
        node = np.unique(u_input)
        items = node.tolist() + (max_n_node - len(node)) * [0]
        # 构建邻接矩阵
        adj = np.zeros((max_n_node, max_n_node))
        w = np.where(node == u_input[-1])[0][0]
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1   # 添加self-loop
            adj[w][u] = 1
            adj[u][w] = 1
            # 如果下一个item是0，则退出
            if u_input[i + 1] == 0:
                break
            # 找到u_input[i]和u_input[i + 1]在node中的位置，用u和v表示
            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1
            adj[u][v] = 1
            # 如果v有到u的边，由于现在u到v也有边，因此u和v之间具有双向的边，双向的边设为4
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            # 如果v到u没边，即只有u到v之间的边，因此adj[u][v]=2(表示u的出边）, adj[v][u]=3（表示v的入边）
            else:
                adj[u][v] = 2
                adj[v][u] = 3
        # 记录会话序列中所有item在node中对应的位置
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        return [torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input)]

    def __len__(self):
        return self.length
# 38.21  17.87
