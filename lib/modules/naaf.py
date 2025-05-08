import torch
import torch.nn as nn
from lib.utils import l2norm, cosine_similarity, get_mask_attention, intra_relation, inter_relations


class NAAF(nn.Module):
    def __init__(self, opt):
        super(NAAF, self).__init__()
        self.opt = opt

        if opt.use_sim_metric:
            self.sim_tran_w = nn.Parameter(torch.ones(opt.embed_size), requires_grad=True)
        elif opt.use_sim_matrix:
            self.n_block = opt.embed_size // opt.sim_dim

            self.sim_tran_w = nn.ModuleList([
                nn.Linear(opt.sim_dim, opt.sim_dim, bias=False) for _ in range(self.n_block)])

            for j in range(self.n_block):
                nn.init.eye_(self.sim_tran_w[j].weight)
        else:
            self.sim_tran_w = None

    def forward(self, img_emb, cap_emb, cap_lens):

        if self.training:
            return xattn_score_for_train(img_emb, cap_emb, cap_lens, self.opt, self.sim_tran_w)
        else:
            return xattn_score_for_test(img_emb, cap_emb, cap_lens, self.opt, self.sim_tran_w)


def xattn_score_for_train(images, captions, cap_lens, opt=None, module=None):
    """
    Note that this function is used to train the model with Discriminative Mismatch Mining.
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    max_pos = []
    max_neg = []
    max_pos_aggre = []
    max_neg_aggre = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    cap_len_i = torch.zeros(1, n_caption)
    n_region = images.size(1)
    batch_size = n_image
    N_POS_WORD = 0
    A = 0
    B = 0
    mean_pos = 0
    mean_neg = 0

    for i in range(n_caption):

        # Get the i-th text description
        n_word = cap_lens[i]
        cap_len_i[0, i] = n_word
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()

        if opt.use_sim_metric:
            cap_i = cap_i * module
        elif opt.use_sim_matrix:
            queryM = list()
            for num, block in enumerate(torch.split(cap_i, opt.sim_dim, dim=-1)):
                queryM.append(module[num](block))
            cap_i = torch.cat(queryM, dim=-1)

        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)

        # text-to-image direction
        t2i_sim = torch.zeros(batch_size * n_word).double().cuda()
        # --> (batch, d, sourceL)
        contextT = torch.transpose(images, 1, 2)

        # attention matrix between all text words and image regions
        attn = torch.bmm(cap_i_expand, contextT)
        attn_i = torch.transpose(attn, 1, 2).contiguous()
        attn_thres = attn - torch.ones_like(attn) * opt.thres

        # Neg-Pos Branch Matching
        # negative attention
        batch_size, queryL, sourceL = images.size(0), cap_i_expand.size(1), images.size(1)
        attn_row = attn_thres.view(batch_size * queryL, sourceL)
        Row_max = torch.max(attn_row, 1)[0].unsqueeze(-1)
        attn_neg = Row_max.lt(0).float()
        t2i_sim_neg = Row_max * attn_neg
        # negative effects
        t2i_sim_neg = t2i_sim_neg.view(batch_size, queryL)

        # positive attention
        # 1) positive effects based on aggregated features
        attn_pos = get_mask_attention(attn_row, batch_size, sourceL, queryL, opt.attn_smooth)
        weiContext_pos = torch.bmm(attn_pos, images)

        t2i_sim_pos_f = cosine_similarity(cap_i_expand, weiContext_pos, dim=2)

        # 2) positive effects based on relevance scores
        attn_weight = inter_relations(attn_i, batch_size, n_region, n_word, opt.attn_smooth)
        t2i_sim_pos_r = attn.mul(attn_weight).sum(-1)

        t2i_sim_pos = t2i_sim_pos_f + t2i_sim_pos_r

        t2i_sim = t2i_sim_neg + t2i_sim_pos
        sim = t2i_sim.mean(dim=1, keepdim=True)

        # Discriminative Mismatch Mining
        wrong_index = sim.sort(0, descending=True)[1][0].item()
        # Based on the correctness of the calculated similarity ranking,
        # we devise to decide whether to update at each sampling time.
        if (wrong_index == i):
            # positive samples
            attn_max_row = torch.max(attn.reshape(batch_size * n_word, n_region).squeeze(), 1)[0].cuda()
            attn_max_row_pos = attn_max_row[(i * n_word): (i * n_word + n_word)].cuda()

            # negative samples
            neg_index = sim.sort(0)[1][0].item()
            attn_max_row_neg = attn_max_row[(neg_index * n_word): (neg_index * n_word + n_word)].cuda()

            max_pos.append(attn_max_row_pos)
            max_neg.append(attn_max_row_neg)
            N_POS_WORD = N_POS_WORD + n_word
            if N_POS_WORD > 200:  # 200 is the empirical value to make adequate samplings
                max_pos_aggre = torch.cat(max_pos, 0)
                max_neg_aggre = torch.cat(max_neg, 0)
                mean_pos = max_pos_aggre.mean().cuda()
                mean_neg = max_neg_aggre.mean().cuda()
                stnd_pos = max_pos_aggre.std()
                stnd_neg = max_neg_aggre.std()

                A = stnd_pos.pow(2) - stnd_neg.pow(2)
                B = 2 * ((mean_pos * stnd_neg.pow(2)) - (mean_neg * stnd_pos.pow(2)))
                C = (mean_neg * stnd_pos).pow(2) - (mean_pos * stnd_neg).pow(2) + 2 * (stnd_pos * stnd_neg).pow(
                    2) * torch.log(stnd_neg / (opt.alpha * stnd_pos) + 1e-8)

                thres = opt.thres
                thres_safe = opt.thres_safe
                opt.stnd_pos = stnd_pos.item()
                opt.stnd_neg = stnd_neg.item()
                opt.mean_pos = mean_pos.item()
                opt.mean_neg = mean_neg.item()

                E = B.pow(2) - 4 * A * C
                if E > 0:
                    # A more simple way to calculate the learning boundary after alpha* adjustement
                    # In implementation, we can use a more feasible opt.thres_safe,
                    # i.e. directly calculate the empirical lower bound, as in the Supplementary Material.
                    # (note that alpha* theoretically unifies the opt.thres at training
                    # and opt.thres_safe at testing into the same concept)
                    opt.thres = ((-B + torch.sqrt(E)) / (2 * A + 1e-10)).item()
                    opt.thres_safe = (mean_pos - 3 * opt.stnd_pos).item()

                if opt.thres < 0:
                    opt.thres = 0
                if opt.thres > 1:
                    opt.thres = 0

                if opt.thres_safe < 0:
                    opt.thres_safe = 0
                if opt.thres_safe > 1:
                    opt.thres_safe = 0

                opt.thres = 0.7 * opt.thres + 0.3 * thres
                opt.thres_safe = 0.7 * opt.thres_safe + 0.3 * thres_safe

        if N_POS_WORD < 200:
            opt.thres = 0
            opt.thres_safe = 0

        similarities.append(sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


def xattn_score_for_test(images, captions, cap_lens, opt=None, module=None):
    """
    Note that there is no need to perform sampling and updating operations in the test stage,
    so we only keep the part except the Discriminative Mismatch Mining module as this test function.

    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    batch_size = n_image

    opt.using_intra_info = True

    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()

        if opt.use_sim_metric:
            cap_i = cap_i * module
        elif opt.use_sim_matrix:
            queryM = list()
            for num, block in enumerate(torch.split(cap_i, opt.sim_dim, dim=-1)):
                queryM.append(module[num](block))
            cap_i = torch.cat(queryM, dim=-1)

        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        # --> (batch, d, sourceL)
        contextT = torch.transpose(images, 1, 2)

        # attention matrix between all text words and image regions
        attn = torch.bmm(cap_i_expand, contextT)
        attn_i = torch.transpose(attn, 1, 2).contiguous()
        attn_thres = attn - torch.ones_like(attn) * opt.thres_safe
        # attn_thres = attn - torch.ones_like(attn) * opt.thres

        # Neg-Pos Branch Matching
        # negative attention
        batch_size, queryL, sourceL = images.size(0), cap_i_expand.size(1), images.size(1)
        attn_row = attn_thres.view(batch_size * queryL, sourceL)
        Row_max = torch.max(attn_row, 1)[0].unsqueeze(-1)

        if opt.using_intra_info:
            attn_intra = intra_relation(cap_i, cap_i, 5)
            attn_intra = attn_intra.repeat(batch_size, 1, 1)
            Row_max_intra = torch.bmm(attn_intra, Row_max.reshape(batch_size, n_word).unsqueeze(-1)).reshape(
                batch_size * n_word, 1)
            attn_neg = Row_max_intra.lt(0).double()
            t2i_sim_neg = Row_max * attn_neg
        else:
            attn_neg = Row_max.lt(0).float()
            t2i_sim_neg = Row_max * attn_neg

        # negative effects
        t2i_sim_neg = t2i_sim_neg.view(batch_size, queryL)

        # positive attention
        # 1) positive effects based on aggregated features
        attn_pos = get_mask_attention(attn_row, batch_size, sourceL, queryL, opt.attn_smooth)
        weiContext_pos = torch.bmm(attn_pos, images)

        t2i_sim_pos_f = cosine_similarity(cap_i_expand, weiContext_pos, dim=2)

        # 2) positive effects based on relevance scores
        attn_weight = inter_relations(attn_i, batch_size, n_region, n_word, opt.attn_smooth)
        t2i_sim_pos_r = attn.mul(attn_weight).sum(-1)

        t2i_sim_pos = t2i_sim_pos_f + t2i_sim_pos_r

        t2i_sim = t2i_sim_neg + t2i_sim_pos
        sim = t2i_sim.mean(dim=1, keepdim=True)

        similarities.append(sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities
