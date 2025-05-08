import torch
import torch.nn as nn


from lib.utils import l2norm, scan_attention


class IMRAM(nn.Module):

    def __init__(self, opt):
        super(IMRAM, self).__init__()

        self.opt = opt

        self.linear_t2i = nn.Linear(opt.embed_size * 2, opt.embed_size)
        self.gate_t2i = nn.Linear(opt.embed_size * 2, opt.embed_size)
        self.linear_i2t = nn.Linear(opt.embed_size * 2, opt.embed_size)
        self.gate_i2t = nn.Linear(opt.embed_size * 2, opt.embed_size)

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
        sim_set = [[] for _ in range(self.opt.attn_layer)]

        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        if self.opt.use_sim_metric:
            img_emb_M = img_emb * self.sim_tran_w
        elif self.opt.use_sim_matrix:
            img_emb_M = list()
            img_emb_split = torch.split(img_emb, self.opt.sim_dim, dim=-1)
            for n in range(self.n_block):
                img_emb_M.append(self.sim_tran_M[n](img_emb_split[n]))
            img_emb_M = torch.cat(img_emb_M, dim=-1)
        else:
            img_emb_M = img_emb

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            if self.opt.use_sim_metric:
                cap_i_expand_M = cap_i_expand * self.sim_tran_w
            elif self.opt.use_sim_matrix:
                cap_i_expand_M = list()
                cap_i_expand_split = torch.split(cap_i_expand, self.opt.sim_dim, dim=-1)
                for m in range(self.n_block):
                    cap_i_expand_M.append(self.sim_tran_M[m](cap_i_expand_split[m]))
                cap_i_expand_M = torch.cat(cap_i_expand_M, dim=-1)
            else:
                cap_i_expand_M = cap_i_expand

            query_t2i, context_t2i = cap_i_expand, img_emb
            query_i2t, context_i2t = img_emb, cap_i_expand

            for j in range(self.opt.attn_layer):

                wcontext_t2i = scan_attention(query_t2i, context_t2i, smooth=self.opt.attn_smooth)
                wcontext_i2t = scan_attention(query_i2t, context_i2t, smooth=self.opt.attn_smooth)

                sim_t2i_ij = torch.sum(cap_i_expand_M * l2norm(wcontext_t2i, dim=-1),
                                       dim=-1).mean(dim=1, keepdim=True)
                sim_i2t_ij = torch.sum(img_emb_M * l2norm(wcontext_i2t, dim=-1),
                                       dim=-1).mean(dim=1, keepdim=True)

                sim_set[j].append(sim_t2i_ij + sim_i2t_ij)

                query_t2i = self.gated_memory(query_t2i, wcontext_t2i, self.linear_t2i, self.gate_t2i)
                query_i2t = self.gated_memory(query_i2t, wcontext_i2t, self.linear_i2t, self.gate_i2t)

        for j in range(self.opt.attn_layer):

            sim_j = torch.cat(sim_set[j], 1).double()
            sim_all.append(sim_j)

        # (n_image, n_caption)
        sim_all = torch.stack(sim_all, 0).sum(0)

        return sim_all

    def gated_memory(self, input_0, input_1, layer_fc, layer_gate):

        input_cat = torch.cat([input_0, input_1], 2)
        input_1 = torch.tanh(layer_fc(input_cat))
        gate = torch.sigmoid(layer_gate(input_cat))
        output = input_0 * gate + input_1 * (1 - gate)

        return output




