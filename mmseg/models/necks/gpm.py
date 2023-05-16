import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


def seq_to_2d(tensor, size_2d):
    h, w = size_2d
    _, n, c = tensor.size()
    tensor = tensor.view(h, w, n, c).permute(2, 3, 0, 1).contiguous()
    return tensor


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


def _get_norm(indim, type='ln', groups=8):
    if type == 'gn':
        return GroupNorm1D(indim, groups)
    else:
        return nn.LayerNorm(indim)


def silu(x):
    return x * torch.sigmoid(x)


def linear_gate(x, dim=-1):
    # return F.relu_(x).pow(2.) / x.size()[dim]
    return torch.softmax(x, dim=dim)


class GroupNorm1D(nn.Module):

    def __init__(self, indim, groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(groups, indim)

    def forward(self, x):
        return self.gn(x.permute(1, 2, 0)).permute(2, 0, 1)


class DropPath(nn.Module):

    def __init__(self, drop_prob=None, batch_dim=0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.batch_dim = batch_dim

    def forward(self, x):
        return self.drop_path(x, self.drop_prob)

    def drop_path(self, x, drop_prob):
        if drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - drop_prob
        shape = [1 for _ in range(x.ndim)]
        shape[self.batch_dim] = x.shape[self.batch_dim]
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


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
        self.projection = nn.Linear(self.expand_d_vu, d_vu // 2)

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

        outputs = self.dw_conv(outputs, size_2d)
        outputs = self.projection(outputs)

        return outputs, attn

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class LocalGatedPropagation(nn.Module):

    def __init__(self,
                 d_qk,
                 d_vu,
                 num_head,
                 dropout=0.,
                 max_dis=7,
                 dilation=1,
                 use_linear=True,
                 enable_corr=True,
                 d_att=None,
                 use_dis=False,
                 expand_ratio=2.):
        super().__init__()
        expand_ratio = 1
        self.expand_d_vu = int(d_vu * expand_ratio)
        self.d_qk = d_qk
        self.d_vu = d_vu
        self.dilation = dilation
        self.window_size = 2 * max_dis + 1
        self.max_dis = max_dis
        self.num_head = num_head
        self.hidden_dim = self.expand_d_vu // num_head
        self.d_att = d_qk // num_head if d_att is None else d_att
        self.T = self.d_att**0.5
        self.use_dis = use_dis

        self.d_middle = self.d_att * self.num_head
        self.use_linear = use_linear
        if use_linear:
            self.linear_QK = nn.Conv2d(d_qk, self.d_middle, kernel_size=1)
            self.linear_V = nn.Conv2d(
                d_vu, self.expand_d_vu, kernel_size=1, groups=2)
            self.linear_U = nn.Conv2d(
                d_vu, self.expand_d_vu, kernel_size=1, groups=2)

        self.relative_emb_k = nn.Conv2d(
            self.d_middle,
            num_head * self.window_size * self.window_size,
            kernel_size=1,
            groups=num_head)

        self.enable_corr = enable_corr

        if enable_corr:
            from spatial_correlation_sampler import SpatialCorrelationSampler
            self.correlation_sampler = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=self.window_size,
                stride=1,
                padding=0,
                dilation=1,
                dilation_patch=self.dilation)

        self.dw_conv = DWConv2d(self.expand_d_vu)
        self.projection = nn.Linear(self.expand_d_vu, d_vu // 2)

        self.dropout = nn.Dropout(dropout)

        self.drop_prob = dropout

        self.local_mask = None
        self.last_size_2d = None
        self.qk_mask = None

    def forward(self, q, k, v, u, size_2d):
        n, c, h, w = v.size()
        hidden_dim = self.hidden_dim

        if self.use_linear:
            q = k = self.linear_QK(q)
            v = silu(self.linear_V(v))
            u = silu(self.linear_U(u))
            if self.num_head > 1:
                v = v.view(-1, 2, self.num_head, hidden_dim // 2,
                           h * w).permute(0, 2, 1, 3, 4).reshape(n, -1, h, w)
                u = u.view(-1, 2, self.num_head, hidden_dim // 2,
                           h * w).permute(4, 0, 2, 1, 3).reshape(h * w, n, -1)
            else:
                u = u.permute(2, 3, 0, 1).reshape(h * w, n, -1)

        if self.qk_mask is not None and (h, w) == self.last_size_2d:
            qk_mask = self.qk_mask
        else:
            memory_mask = torch.ones((1, 1, h, w), device=v.device).float()
            unfolded_k_mask = self.pad_and_unfold(memory_mask).view(
                1, 1, self.window_size * self.window_size, h * w)
            qk_mask = 1 - unfolded_k_mask
            self.qk_mask = qk_mask

        relative_emb = self.relative_emb_k(q)

        # Scale
        q = q / self.T

        q = q.view(-1, self.d_att, h, w)
        k = k.view(-1, self.d_att, h, w)
        v = v.view(-1, self.num_head, hidden_dim, h * w)

        relative_emb = relative_emb.view(n, self.num_head,
                                         self.window_size * self.window_size,
                                         h * w)

        if self.enable_corr:
            qk = self.correlation_sampler(q, k).view(
                n, self.num_head, self.window_size * self.window_size, h * w)
        else:
            unfolded_k = self.pad_and_unfold(k).view(
                n * self.num_head, hidden_dim,
                self.window_size * self.window_size, h, w)
            qk = (q.unsqueeze(2) * unfolded_k).sum(dim=1).view(
                n, self.num_head, self.window_size * self.window_size, h * w)
        if self.use_dis:
            qk = 2 * qk - self.pad_and_unfold(
                k.pow(2).sum(dim=1, keepdim=True)).view(
                    n, self.num_head, self.window_size * self.window_size,
                    h * w)

        qk = qk + relative_emb

        qk -= qk_mask * 1e+8 if qk.dtype == torch.float32 else qk_mask * 1e+4

        local_attn = linear_gate(qk, dim=2)

        local_attn = self.dropout(local_attn)

        global_attn = self.local2global(local_attn, h, w)

        agg_value = (global_attn @ v.transpose(-2, -1)).permute(
            2, 0, 1, 3).reshape(h * w, n, -1)

        output = agg_value * u

        output = self.dw_conv(output, size_2d)
        output = self.projection(output)

        self.last_size_2d = (h, w)
        return output, local_attn

    def local2global(self, local_attn, height, width):
        batch_size = local_attn.size()[0]

        pad_height = height + 2 * self.max_dis
        pad_width = width + 2 * self.max_dis

        if self.local_mask is not None and (height,
                                            width) == self.last_size_2d:
            local_mask = self.local_mask
        else:
            ky, kx = torch.meshgrid([
                torch.arange(0, pad_height, device=local_attn.device),
                torch.arange(0, pad_width, device=local_attn.device)
            ])
            qy, qx = torch.meshgrid([
                torch.arange(0, height, device=local_attn.device),
                torch.arange(0, width, device=local_attn.device)
            ])

            offset_y = qy.reshape(-1, 1) - ky.reshape(1, -1) + self.max_dis
            offset_x = qx.reshape(-1, 1) - kx.reshape(1, -1) + self.max_dis

            local_mask = (offset_y.abs() <= self.max_dis) & (
                offset_x.abs() <= self.max_dis)
            local_mask = local_mask.view(1, 1, height * width, pad_height,
                                         pad_width)
            self.local_mask = local_mask

        global_attn = torch.zeros(
            (batch_size, self.num_head, height * width, pad_height, pad_width),
            device=local_attn.device)
        global_attn[local_mask.expand(batch_size, self.num_head,
                                      -1, -1, -1)] = local_attn.transpose(
                                          -1, -2).reshape(-1)
        global_attn = global_attn[:, :, :, self.max_dis:-self.max_dis,
                                  self.max_dis:-self.max_dis].reshape(
                                      batch_size, self.num_head,
                                      height * width, height * width)

        return global_attn

    def pad_and_unfold(self, x):
        pad_pixel = self.max_dis * self.dilation
        x = F.pad(
            x, (pad_pixel, pad_pixel, pad_pixel, pad_pixel),
            mode='constant',
            value=0)
        x = F.unfold(
            x,
            kernel_size=(self.window_size, self.window_size),
            stride=(1, 1),
            dilation=self.dilation)
        return x


class GatedPropagationModule(nn.Module):
    def __init__(self,
                 d_model,
                 self_nhead,
                 att_nhead,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 local_dilation=1,
                 max_local_dis=7,
                 layer_idx=0,
                 expand_ratio=2.):
        super().__init__()
        expand_ratio = expand_ratio
        expand_d_model = int(d_model * expand_ratio)
        self.expand_d_model = expand_d_model
        self.d_model = d_model
        self.att_nhead = att_nhead

        d_att = d_model // 2 if att_nhead == 1 else d_model // att_nhead
        self.d_att = d_att
        self.layer_idx = layer_idx

        # Long Short-Term Attention
        self.norm1 = _get_norm(d_model)
        self.linear_QV = nn.Linear(d_model, d_att * att_nhead + expand_d_model)
        self.linear_U = nn.Linear(d_model, expand_d_model)


        self.long_term_attn = GatedPropagation(
            d_qk=self.d_model,
            d_vu=self.d_model * 2,
            num_head=att_nhead,
            use_linear=False,
            dropout=lt_dropout,
            d_att=d_att,
            top_k=-1,
            expand_ratio=expand_ratio)

        self.short_term_attn = LocalGatedPropagation(
            d_qk=self.d_model,
            d_vu=self.d_model * 2,
            num_head=att_nhead,
            dilation=local_dilation,
            use_linear=False,
            dropout=st_dropout,
            d_att=d_att,
            max_dis=max_local_dis,
            expand_ratio=expand_ratio)

        self.lst_dropout = nn.Dropout(max(lt_dropout, st_dropout), True)
        self.droppath_lst = droppath_lst

        # Self-attention
        self.norm2 = _get_norm(d_model)
        self.id_norm2 = _get_norm(d_model)
        self.self_attn = GatedPropagation(
            d_model, d_model * 2, self_nhead, d_att=d_att)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                long_term_memory=None,
                short_term_memory=None,
                size_2d=(30, 30)):

        # Long Short-Term Attention
        _tgt = self.norm1(tgt)

        curr_QV = self.linear_QV(_tgt)
        curr_QV = torch.split(
            curr_QV, [self.d_att * self.att_nhead, self.expand_d_model], dim=2)
        curr_Q = curr_K = curr_QV[0]
        local_Q = seq_to_2d(curr_Q, size_2d)
        curr_V = silu(curr_QV[1])
        curr_U = self.linear_U(_tgt)
        curr_U = silu(curr_U)

        if long_term_memory is None:
            global_K, global_V = curr_K, curr_V
        else:
            global_K, global_V = long_term_memory

        if short_term_memory is None:
            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)
        else:
            local_K, local_V = short_term_memory
            local_K = seq_to_2d(local_K, size_2d=size_2d)
            local_V = seq_to_2d(local_V, size_2d=size_2d)


        tgt2, _ = self.long_term_attn(curr_Q, global_K, global_V, curr_U,
                                      size_2d)
        tgt3, _ = self.short_term_attn(local_Q, local_K, local_V, curr_U,
                                       size_2d)

        if self.droppath_lst:
            tgt = tgt + self.droppath(tgt2 + tgt3)
        else:
            tgt = tgt + self.lst_dropout(tgt2 + tgt3)

        # Self-attention
        _tgt = self.norm2(tgt)
        q = k = v = u = _tgt

        tgt4, _ = self.self_attn(q, k, v, u, size_2d)

        tgt = tgt + self.droppath(tgt4)

        return tgt, [[curr_K, curr_V], [global_K, global_V],
                     [local_K, local_V]]

    def fuse_key_value_id(self, key, value, id_emb):
        ID_K = None
        if value is not None:
            ID_V = silu(self.linear_ID_V(torch.cat([value, id_emb], dim=2)))
        else:
            ID_V = silu(self.linear_ID_V(id_emb))
        return ID_K, ID_V

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



@MODELS.register_module()
class GPM(nn.Module):
    def __init__(self,
                 num_layers=3,
                 d_model=64,
                 self_nhead=1,
                 att_nhead=1,
                 emb_dropout=0.,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 return_intermediate=False,
                 intermediate_norm=True,
                 final_norm=True):

        super().__init__()
        # memories init
        self.long_term_memories = [None,None,None]
        self.short_term_memories = [None,None,None]

        self.intermediate_norm = intermediate_norm
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.emb_dropout = nn.Dropout(emb_dropout, True)
        # self.mask_token = nn.Parameter(torch.randn([1, 1, d_model]))

        block = GatedPropagationModule

        layers = [
            block(
                d_model=d_model*2,
                self_nhead=self_nhead,
                att_nhead=att_nhead,
                droppath=droppath,
                lt_dropout=lt_dropout,
                st_dropout=st_dropout,
                droppath_lst=droppath_lst,
                layer_idx=0,
            ),
        ] 
        self.layers = nn.ModuleList(layers)

        num_norms = num_layers - 1 if intermediate_norm else 0
        if final_norm:
            num_norms += 1
        self.decoder_norms = [
            _get_norm(d_model * 2, type='gn', groups=2)
            for _ in range(num_norms)
        ] if num_norms > 0 else None

        if self.decoder_norms is not None:
            self.decoder_norms = nn.ModuleList(self.decoder_norms)
            
    def forward(self, tgt, size_2d=None):
        # dropout
        for i in range(len(tgt)):
            n,c,h,w = tgt[i].size()
            tgt[i] = tgt[i].view(n,c,h*w).permute(2,0,1)
            tgt[i] = self.emb_dropout(tgt[i])
        size_2d=(h,w)

        intermediate = []
        intermediate_memories = []
        outputs = []
        save_memories = []

        for idx, layer in enumerate(self.layers):
            output, memories = layer(
                tgt[idx],
                self.long_term_memories[idx]
                if self.long_term_memories is not None else None,
                self.short_term_memories[idx]
                if self.short_term_memories is not None else None,
                size_2d=size_2d)

            outputs.append(output)
            save_memories.append(memories)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_memories.append(memories)

        self.update_memories(save_memories)
                
        if self.decoder_norms is not None:
            if self.final_norm:
                for output in outputs:
                    output = self.decoder_norms[-1](output)
        
        h,w = size_2d
        for i in range(len(outputs)):
            _, n, c = outputs[i].size()
            outputs[i] = outputs[i].view(h,w,n,c).permute(2,3,0,1)

        return outputs

    def clear_memories(self):
        self.long_term_memories = [None for i in range(len(self.layers))]
        self.short_term_memories = [None for i in range(len(self.layers))]

    def update_memories(self,memories):
        # long_term_memories
        if self.long_term_memories[0] is None:
            long_term_memories = []
            for memory in memories:
                long_term_memories.append(memory[1])
            self.long_term_memories = long_term_memories
        # short_term_memories
        short_term_memories = []
        for memory in memories:
            short_term_memories.append(memory[0])
        self.short_term_memories = short_term_memories

        

