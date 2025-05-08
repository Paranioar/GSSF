import torch
import torch.nn as nn

from lib.utils import l2norm, bfan_attention


class BFAN(nn.Module):

    def __init__(self, opt):
        super(BFAN, self).__init__()

        self.opt = opt

        if opt.use_sim_metric:

            self.sim_tran_w_t2i = nn.Parameter(torch.ones(opt.embed_size), requires_grad=True)
            self.sim_tran_w_i2t = nn.Parameter(torch.ones(opt.embed_size), requires_grad=True)

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

        if self.opt.use_sim_metric:
            img_emb_M = img_emb * self.sim_tran_w_i2t
        elif self.opt.use_sim_matrix:
            img_emb_M = list()
            img_emb_split = torch.split(img_emb, self.opt.sim_dim, dim=-1)
            for j in range(self.n_block):
                img_emb_M.append(self.sim_tran_M[j](img_emb_split[j]))
            img_emb_M = torch.cat(img_emb_M, dim=-1)
        else:
            img_emb_M = img_emb

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            # local alignment construction
            context_img = bfan_attention(cap_i_expand, img_emb, self.opt.attn_smooth, self.opt.attn_type)
            context_img = l2norm(context_img, dim=-1)

            context_cap = bfan_attention(img_emb, cap_i_expand, self.opt.attn_smooth, self.opt.attn_type)
            context_cap = l2norm(context_cap, dim=-1)

            if self.opt.use_sim_metric:
                cap_i_expand_M = cap_i_expand * self.sim_tran_w_t2i
            elif self.opt.use_sim_matrix:
                cap_i_expand_M = list()
                cap_i_expand_split = torch.split(cap_i_expand, self.opt.sim_dim, dim=-1)
                for j in range(self.n_block):
                    cap_i_expand_M.append(self.sim_tran_M[j](cap_i_expand_split[j]))
                cap_i_expand_M = torch.cat(cap_i_expand_M, dim=-1)
            else:
                cap_i_expand_M = cap_i_expand

            sim_t2i_i = torch.sum(cap_i_expand_M * context_img, dim=-1).mean(dim=1, keepdim=True)
            sim_i2t_i = torch.sum(img_emb_M * context_cap, dim=-1).mean(dim=1, keepdim=True)

            sim_i = sim_t2i_i + sim_i2t_i

            sim_all.append(sim_i)

        # (n_image, n_caption)
        sim_all = torch.cat(sim_all, 1)

        return sim_all