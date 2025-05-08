import torch
import torch.nn as nn

import numpy as np
import scipy.linalg as linalg

from lib.utils import l2norm, scan_attention


class SCAN(nn.Module):

    def __init__(self, opt):
        super(SCAN, self).__init__()

        self.opt = opt

        if opt.use_sim_metric:
            self.sim_tran_w = nn.Parameter(torch.ones(opt.embed_size), requires_grad=True)
        elif opt.use_sim_matrix:
            self.n_block = opt.embed_size // opt.sim_dim

            self.sim_tran_M = nn.ModuleList([
                nn.Linear(opt.sim_dim, opt.sim_dim, bias=False) for _ in range(self.n_block)])
            for j in range(self.n_block):
                nn.init.eye_(self.sim_tran_M[j].weight)

    def forward(self, img_emb, cap_emb, cap_lens):
        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            # local alignment construction
            if self.opt.attn_type == 't2i':
                query, context = cap_i_expand, img_emb
            elif self.opt.attn_type == 'i2t':
                query, context = img_emb, cap_i_expand
            else:
                raise ValueError('Invalid value of opt.attn_type')

            wcontext = scan_attention(query, context, smooth=self.opt.attn_smooth)
            wcontext = l2norm(wcontext, dim=-1)

            if self.opt.use_sim_metric:
                sim_i = torch.sum(query * wcontext * self.sim_tran_w, dim=-1).mean(dim=1, keepdim=True)

            elif self.opt.use_sim_matrix:
                queryM = list()
                for j, block in enumerate(torch.split(query, self.opt.sim_dim, dim=-1)):
                    queryM.append(self.sim_tran_M[j](block))
                queryM = torch.cat(queryM, dim=-1)
                sim_i = torch.sum(queryM * wcontext, dim=-1).mean(dim=1, keepdim=True)

            else:
                sim_i = torch.sum(query * wcontext, dim=-1).mean(dim=1, keepdim=True)

            sim_all.append(sim_i)

        # (n_image, n_caption)
        sim_all = torch.cat(sim_all, 1)

        return sim_all


# class SCAN(nn.Module):
#
#     def __init__(self, opt):
#         super(SCAN, self).__init__()
#
#         self.opt = opt
#
#         if opt.use_sim_metric:
#             self.sim_tran_w = nn.Linear(opt.embed_size, opt.embed_size, bias=False)
#             self.mask = torch.eye(opt.embed_size).cuda()
#             nn.init.eye_(self.sim_tran_w.weight)
#         elif opt.use_sim_matrix:
#             self.n_block = opt.embed_size // opt.sim_dim
#
#             self.sim_tran_w = nn.Linear(opt.embed_size, opt.embed_size, bias=False)
#             ones_matrix = (np.ones((opt.sim_dim, opt.sim_dim)) for _ in range(self.n_block))
#             block_matrix = linalg.block_diag(*ones_matrix)
#             self.mask = torch.FloatTensor(block_matrix).cuda()
#             nn.init.eye_(self.sim_tran_w.weight)
#
#
#     def forward(self, img_emb, cap_emb, cap_lens):
#         sim_all = []
#         n_image = img_emb.size(0)
#         n_caption = cap_emb.size(0)
#
#         for i in range(n_caption):
#             # get the i-th sentence
#             n_word = cap_lens[i]
#             cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
#             cap_i_expand = cap_i.repeat(n_image, 1, 1)
#
#             # local alignment construction
#             if self.opt.attn_type == 't2i':
#                 query, context = cap_i_expand, img_emb
#             elif self.opt.attn_type == 'i2t':
#                 query, context = img_emb, cap_i_expand
#             else:
#                 raise ValueError('Invalid value of opt.attn_type')
#
#             wcontext = scan_attention(query, context, smooth=self.opt.attn_smooth)
#             wcontext = l2norm(wcontext, dim=-1)
#
#             if self.opt.use_sim_metric or self.opt.use_sim_matrix:
#                 weights = torch.eye(self.opt.embed_size).to(img_emb.device)
#                 weights = self.sim_tran_w(weights) * self.mask
#                 query = query @ weights
#
#             sim_i = torch.sum(query * wcontext, dim=-1).mean(dim=1, keepdim=True)
#
#             sim_all.append(sim_i)
#
#         # (n_image, n_caption)
#         sim_all = torch.cat(sim_all, 1)
#
#         return sim_all