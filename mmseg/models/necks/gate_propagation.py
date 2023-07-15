import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_gate(x, dim=-1):
    # return F.relu_(x).pow(2.) / x.size()[dim]
    return torch.softmax(x, dim=dim)

def silu(x):
    return x * torch.sigmoid(x)

def multiply_by_ychunks(x, y, chunks=1):
    if chunks <= 1:
        return x @ y
    else:
        return torch.cat([x @ _y for _y in y.chunk(chunks, dim=-1)], dim=-1)


def multiply_by_xchunks(x, y, chunks=1):
    if chunks <= 1:
        return x @ y
    else:
        return torch.cat([_x @ y for _x in x.chunk(chunks, dim=-2)], dim=-2)

class DWConv2d(nn.Module):

    def __init__(self, indim, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(
            indim, indim, 5, dilation=1, padding=2, groups=indim, bias=False)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, x, size_2d):
        h, w = size_2d
        _, bs, c = x.size()
        x = x.view(h, w, bs, c).permute(2, 3, 0, 1)
        x = self.conv(x)
        x = self.dropout(x)
        x = x.view(bs, c, h * w).permute(2, 0, 1)
        return x

class GatedPropagation(nn.Module):
    def __init__(self,
                 d_qk,
                 d_vu,
                 num_head=1,
                 dropout=0.,
                 use_linear=True,
                 d_att=None,
                 use_dis=False,
                 qk_chunks=1,
                 max_mem_len_ratio=-1,
                 top_k=-1,
                 expand_ratio=1.):
        super().__init__()
        expand_ratio = 1
        self.expand_d_vu = int(d_vu * expand_ratio)
        self.d_vu = d_vu
        self.d_qk = d_qk
        self.num_head = num_head
        self.use_dis = use_dis
        self.qk_chunks = qk_chunks
        self.max_mem_len_ratio = float(max_mem_len_ratio)
        self.top_k = top_k

        self.hidden_dim = self.expand_d_vu // num_head
        self.d_att = d_qk // num_head if d_att is None else d_att
        self.T = self.d_att**0.5
        self.use_linear = use_linear
        self.d_middle = self.d_att * self.num_head

        if use_linear:
            self.linear_QK = nn.Linear(d_qk, self.d_middle)
            half_d_vu = self.hidden_dim * num_head // 2
            self.linear_V1 = nn.Linear(d_vu // 4, half_d_vu)
            self.linear_V2 = nn.Linear(d_vu // 4, half_d_vu)
            self.linear_U1 = nn.Linear(d_vu // 4, half_d_vu)
            self.linear_U2 = nn.Linear(d_vu // 4, half_d_vu)

        self.dropout = nn.Dropout(dropout)
        self.drop_prob = dropout

        self.dw_conv = DWConv2d(self.expand_d_vu)
        self.projection = nn.Linear(self.expand_d_vu, d_vu // 4)

        self._init_weight()

    def forward(self, Q, K, V, U, size_2d):
        """
        :param Q: A 3d tensor with shape of [T_q, bs, C_q]
        :param K: A 3d tensor with shape of [T_k, bs, C_k]
        :param V: A 3d tensor with shape of [T_v, bs, C_v]
        """
        num_head = self.num_head
        hidden_dim = self.hidden_dim

        l, bs, _ = Q.size()

        # Linear projections
        if self.use_linear:
            Q = K = self.linear_QK(Q)

            def cat(X1, X2):
                if num_head > 1:
                    X1 = X1.view(-1, bs, num_head, hidden_dim // 2)
                    X2 = X2.view(-1, bs, num_head, hidden_dim // 2)
                    X = torch.cat([X1, X2],
                                  dim=-1).view(-1, bs, num_head * hidden_dim)
                else:
                    X = torch.cat([X1, X2], dim=-1)
                return X
            V1, V2 = torch.split(V, self.d_vu // 4, dim=-1)
            V1 = self.linear_V1(V1)
            V2 = self.linear_V2(V2)
            V = silu(cat(V1, V2))

            U1, U2 = torch.split(U, self.d_vu // 4, dim=-1)
            U1 = self.linear_U1(U1)
            U2 = self.linear_U2(U2)
            U = silu(cat(U1, U2))

        # Scale
        Q = Q / self.T

        if not self.training and self.max_mem_len_ratio > 0:
            mem_len_ratio = float(K.size(0)) / Q.size(0)
            if mem_len_ratio > self.max_mem_len_ratio:
                scaling_ratio = math.log(mem_len_ratio) / math.log(
                    self.max_mem_len_ratio)
                Q = Q * scaling_ratio

        # Multi-head
        Q = Q.view(-1, bs, num_head, self.d_att).permute(1, 2, 0, 3)
        K = K.view(-1, bs, num_head, self.d_att).permute(1, 2, 3, 0)
        V = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)

        # Multiplication
        QK = multiply_by_ychunks(Q, K, self.qk_chunks)
        if self.use_dis:
            QK = 2 * QK - K.pow(2).sum(dim=-2, keepdim=True)

        # Activation
        if not self.training and self.top_k > 0 and self.top_k < QK.size()[-1]:
            top_QK, indices = torch.topk(QK, k=self.top_k, dim=-1)
            top_attn = linear_gate(top_QK, dim=-1)
            attn = torch.zeros_like(QK).scatter_(-1, indices, top_attn)
        else:
            attn = linear_gate(QK, dim=-1)

        # Dropouts
        attn = self.dropout(attn)

        # Weighted sum
        outputs = multiply_by_xchunks(attn, V,
                                      self.qk_chunks).permute(2, 0, 1, 3)

        # Restore shape
        outputs = outputs.reshape(l, bs, -1) * U

        # outputs = self.dw_conv(outputs, size_2d)
        outputs = self.projection(outputs)

        return outputs, attn

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)