import torch
import torch.nn as nn
import scipy.linalg as linalg

import numpy as np

from lib.utils import l2norm


class VSE(nn.Module):

    def __init__(self, opt):
        super(VSE, self).__init__()

        self.opt = opt

        if opt.attn_type == 'cosine':
            self.sim_cosine_w = None

        elif opt.use_sim_metric:
            self.sim_metric_w = nn.Parameter(torch.ones(opt.embed_size), requires_grad=True)

            self.mask = torch.ones(self.opt.embed_size).cuda()
            self.dropout = nn.Dropout(opt.attn_dp)

        elif opt.use_sim_matrix:
            self.n_block = opt.embed_size // opt.sim_dim
            self.sim_matrix_w = nn.Parameter(torch.eye(opt.embed_size), requires_grad=True)

            ones_matrix = (np.ones((opt.sim_dim, opt.sim_dim)) for _ in range(self.n_block))
            block_matrix = linalg.block_diag(*ones_matrix)
            self.mask = torch.FloatTensor(block_matrix).cuda()
            self.dropout = nn.Dropout(opt.attn_dp)

        elif opt.attn_type == 'fc':
            self.sim_fc_w = nn.Linear(opt.embed_size, opt.embed_size, bias=False)
            nn.init.eye_(self.sim_fc_w.weight)

        elif opt.attn_type == 'mlp':
            self.sim_mlp_w = nn.Sequential(nn.Linear(opt.embed_size, opt.sim_dim),
                                           nn.Tanh(),
                                           nn.Dropout(opt.attn_dp),
                                           nn.Linear(opt.sim_dim, 1),
                                           nn.Sigmoid())
        elif opt.attn_type == 'vector':
            self.sim_vector_w1 = nn.Linear(opt.embed_size, opt.sim_dim)
            self.sim_vector_w2 = nn.Sequential(nn.Linear(opt.sim_dim, 1), nn.Sigmoid())

        elif opt.attn_type == 'tensor':
            self.sim_tensor_w1 = nn.Sequential(nn.Linear(opt.embed_size, opt.embed_size * opt.attn_layer),
                                               nn.Dropout(opt.attn_dp))
            self.sim_tensor_w2 = nn.Sequential(nn.Linear(opt.embed_size, opt.embed_size * opt.attn_layer),
                                               nn.Dropout(opt.attn_dp))
            self.sim_tensor_w3 = nn.Sequential(nn.Linear(opt.embed_size, 1), nn.Sigmoid())
        else:
            raise ValueError('Invalid input of opt.attn_type')

    def forward(self, img_emb, cap_emb, cap_lens):

        n_image, n_caption = img_emb.size(0), cap_emb.size(0)

        if isinstance(cap_lens, list):
            cap_lens = torch.FloatTensor(cap_lens).to(img_emb.device)

        img_emb = l2norm(img_emb.mean(1), dim=-1)
        cap_emb = l2norm(cap_emb.sum(1) / cap_lens.unsqueeze(1), dim=-1)

        if self.opt.use_sim_metric:
            img_emb = l2norm(self.dropout(img_emb), dim=-1)
            cap_emb = l2norm(self.dropout(cap_emb), dim=-1)

            weights = torch.abs(self.mask * self.sim_metric_w)
            sim_all = (img_emb * weights) @ cap_emb.t()

        elif self.opt.use_sim_matrix:
            img_emb = l2norm(self.dropout(img_emb), dim=-1)
            cap_emb = l2norm(self.dropout(cap_emb), dim=-1)

            weights = torch.abs(self.mask * self.sim_matrix_w)
            sim_all = (img_emb @ weights) @ cap_emb.t()

        elif self.opt.attn_type == 'fc':
            sim_all = self.sim_fc_w(img_emb) @ cap_emb.t()

        elif self.opt.attn_type == 'mlp':
            img_emb = img_emb.unsqueeze(1).repeat(1, n_caption, 1)
            sim_all = self.sim_mlp_w(img_emb + cap_emb.unsqueeze(0)).squeeze(-1)

        elif self.opt.attn_type == 'vector':
            img_emb = img_emb.unsqueeze(1).repeat(1, n_caption, 1)
            sim_vec = torch.pow(torch.sub(img_emb, cap_emb.unsqueeze(0)), 2)
            sim_vec = l2norm(self.sim_vector_w1(sim_vec), dim=-1)
            sim_all = self.sim_vector_w2(sim_vec).squeeze(-1)

        elif self.opt.attn_type == 'tensor':
            img_emb = self.sim_tensor_w1(img_emb)
            cap_emb = self.sim_tensor_w2(cap_emb)
            sim_vec = img_emb.unsqueeze(1).repeat(1, n_caption, 1) * cap_emb.unsqueeze(0)
            sim_vec = sim_vec.reshape(n_image, n_caption, self.opt.attn_layer, self.opt.embed_size)
            sim_all = self.sim_tensor_w3(sim_vec.sum(2)).squeeze(-1)

        else:
            sim_all = img_emb @ cap_emb.t()

        return sim_all


if __name__ == '__main__':
    import arguments
    import os
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()
    opt.sim_dim = 2
    opt.use_sim_matrix = True
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuid
    model = VSE(opt)
    model.cuda()
    tensorA = torch.randn(128, 36, 1024).cuda()
    tensorB = torch.randn(128, 27, 1024).cuda()
    len = [27] * 128
    out = model(tensorA, tensorB, len)

    print(out.size())