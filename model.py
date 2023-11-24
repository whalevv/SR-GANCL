# -*- coding: utf-8 -*-
"""
@Time ： 2023/6/21 21:56
@Auth ： whale
@File ：model.py.py
@IDE ：PyCharm
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from tqdm import tqdm

def layer_normalization(x):
    x = x - torch.mean(x, -1).unsqueeze(-1)
    norm_x = torch.sqrt(torch.sum(x**2, -1)).unsqueeze(-1)
    y = x / norm_x
    return y
# 这里forword函数的输入分别是会话中所有item的embedding：hidden， item的邻居item矩阵：adj，掩码：mask_item
class SGAT(Module):
    def __init__(self, hidden_size, step=3):
        super(SGAT, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.dim = hidden_size
        self.dropout = 0.0

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(0.2)

    def GAT(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]
        # 这里是公式(7)中点积部分：h_{v_i} \cdot h_{v_j}
        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)
        # 对a_input进行四种不同的线性映射表示四种类型的边
        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        # 计算注意系数
        # squeeze(-1): 去除最后维度值为1的维度         view(): 转换成N*N
        # 进行LeakyReLU激活，完成公式(7)其余部分
        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)
        # 创建一个值全为-9e15的矩阵，如果adj中的值为1，则让alpha中对应的值为e_0中对应的值，否则alpha中对应的值为-9e15
        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        # 这样alpha即由1和-9e15两个值组成，若其中的值等于2，则alpha中对应的值为e_1中对应的值，否则保持不变，3和4同理
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        # 对alpha进行softmax操作
        alpha = torch.softmax(alpha, dim=-1)
        # 公式(9)加权求和
        output = torch.matmul(alpha, h)
        return output

    def ave_pooling(self, hidden, graph_mask):
        length = torch.sum(graph_mask, 1)
        hidden = hidden * graph_mask.unsqueeze(-1).float()
        output = torch.sum(hidden, 1) / length.unsqueeze(-1).float()
        return output

    def att_pooling(self, hidden, star_node, graph_mask):
        sim = torch.matmul(hidden, star_node.unsqueeze(-1)).squeeze()
        sim = torch.exp(sim)
        sim_mask = sim * graph_mask.float()
        sim_each = torch.sum(sim_mask, -1).unsqueeze(-1) + 1e-24
        sim = sim_mask/sim_each
        output = torch.sum(sim.unsqueeze(-1) * hidden, 1)
        return output

    def forward(self, A, hidden, graph_mask):
        star_node = self.ave_pooling(hidden, graph_mask)
        embs = []
        for i in range(self.step):
            hidden = self.GAT(hidden,A)
            sim = torch.matmul(hidden, star_node.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.hidden_size)
            alpha = torch.sigmoid(sim).unsqueeze(-1)
            bs, item_num = hidden.shape[0], hidden.shape[1]
            star_node_repeat = star_node.repeat(1, item_num).view(bs, item_num, self.hidden_size)


            hidden = (1-alpha) * hidden + alpha * star_node_repeat
            random_noise = torch.rand_like(hidden).cuda()
            hidden += torch.sign(hidden) * F.normalize(random_noise, dim=-1) * 0.4
            star_node = self.att_pooling(hidden, star_node, graph_mask)
            embs.append(hidden)
        final_embeddings = torch.stack(embs,dim=1)
        embeddings_cl = torch.mean(final_embeddings,dim=1)
        return hidden, star_node,embeddings_cl


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.tau = opt.tau
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.pos_embedding = nn.Embedding(200, self.hidden_size)
        # self.gnn = SGNN(self.hidden_size, step=opt.step)
        self.gnn = SGAT(self.hidden_size, step=opt.step)
        self.linear_hn = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_hn1 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_four = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_one1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_four1 = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_transform1 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, seq_hidden, hg1, star_node, mask, graph_mask):
        # Pos Embedding
        bs, item_num = hg1.shape[0], hg1.shape[1]
        index = torch.arange(item_num).unsqueeze(0)
        pos_index = index.repeat(bs, 1).view(bs, item_num)
        pos_index = trans_to_cuda(torch.Tensor(pos_index.float()).long())
        pos_hidden = self.pos_embedding(pos_index)
        # random_noise = torch.rand_like(hidden).cuda()
        # hidden += torch.sign(hidden) * F.normalize(random_noise, dim=-1) * 0.2
        hg1 = hg1 + pos_hidden
         # Last Item
        ht = seq_hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        ht1 = hg1[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(star_node).view(star_node.shape[0], 1, star_node.shape[1])  # batch_size x 1 x latent_size
        q3 = self.linear_three(seq_hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_four(torch.sigmoid(q1 + q2 + q3))
        q11 = self.linear_one1(ht1).view(ht1.shape[0], 1, ht1.shape[1])  # batch_size x 1 x latent_size
        q21 = self.linear_two1(star_node).view(star_node.shape[0], 1, star_node.shape[1])  # batch_size x 1 x latent_size
        q31 = self.linear_three1(hg1)  # batch_size x seq_length x latent_size
        alpha1 = self.linear_four1(torch.sigmoid(q11 + q21 + q31))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        a = self.linear_transform(torch.cat([a, ht], 1))
        a = layer_normalization(a)

        a1 = torch.sum(alpha1 * hg1 * mask.view(mask.shape[0], -1, 1).float(), 1)
        a1 = self.linear_transform1(torch.cat([a1, ht1], 1))
        a1 = layer_normalization(a1)
        pos_ratings_user = torch.sum(a * a1, dim=-1)
        tot_ratings_user = torch.matmul(a,
                                        torch.transpose(a1, 0, 1))  # [batch_size, num_users]
        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]  # [batch_size, num_users]
        clogits_user = torch.logsumexp(ssl_logits_user / 0.2, dim=1)
        infonce_loss = torch.sum(clogits_user)
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        b = layer_normalization(b)
        scores = torch.matmul(a, b.transpose(1, 0))
        scores *= self.tau
        return scores, infonce_loss



    def forward(self, inputs, A, graph_mask, item):
        hidden = self.embedding(inputs)
        hidden_update, star_node ,hg1= self.gnn(A, hidden, graph_mask)
        hidden_concat = torch.cat([hidden, hidden_update], -1) # bs * item_num * (2*emb_dim)
        alpha = self.linear_hn(hidden_concat) # bs * item_num * emb_dim
        alpha = torch.sigmoid(alpha)
        output = alpha * hidden + (1-alpha) * hidden_update
        # output = hidden_update

        hidden_concat1 = torch.cat([hidden, hg1], -1)  # bs * item_num * (2*emb_dim)
        alpha1 = self.linear_hn1(hidden_concat1)  # bs * item_num * emb_dim
        alpha1 = torch.sigmoid(alpha1)
        output1 = alpha * hidden + (1 - alpha1) * hg1
        # output1 = hg1
        return output, star_node, output1

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable



def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs= data
    alias_inputs = trans_to_cuda(alias_inputs.long())
    items = trans_to_cuda(items.long())
    adj = trans_to_cuda(adj.float())
    mask = trans_to_cuda(mask.long())
    graph_mask = torch.sign(items)
    inputs = trans_to_cuda(inputs.long())
    hl, star_node,output1 = model(items, adj, graph_mask, inputs)
    get = lambda index: hl[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    get1 = lambda index: output1[index][alias_inputs[index]]
    hl1 = torch.stack([get1(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, hl1, star_node, mask, graph_mask)




def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=False, pin_memory=True)
    # print(model.batch_size)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        # print(data,len(data),data[0].shape)
        targets, a = forward(model, data)
        scores, infoNCE_loss = a[0],a[1]
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss1 = loss + 0.001 * infoNCE_loss
        loss1.backward()
        model.optimizer.step()
        total_loss += loss1
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit, mrr = [], []
    for data in test_loader:
        targets, a = forward(model, data)
        scores, infoNCE_loss = a[0],a[1]
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)

    return result