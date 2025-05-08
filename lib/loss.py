import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import kl_loss

import logging
logger = logging.getLogger(__name__)


class BirankLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False, **kwargs):
        super(BirankLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores, **kwargs):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        mask = mask.to(scores.device)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class PolyLoss(nn.Module):
    def __init__(self, margin=0, **kwargs):
        super(PolyLoss, self).__init__()
        self.eps = 1e-5
        self.margin = margin

    def forward(self, scores, **kwargs):
        size = scores.size(0)
        hh = scores.t()
        label = torch.Tensor([i for i in range(size)])

        loss = list()
        for i in range(size):
            pos_pair_ = scores[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - self.eps]
            neg_pair_ = scores[i][label != label[i]]

            pos_pair = pos_pair_
            if len(pos_pair) < 1:
                continue

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            if len(neg_pair) < 1:
                continue

            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)
            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)

            loss.append(pos_loss + neg_loss)

        for i in range(size):
            pos_pair_ = hh[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - self.eps]
            neg_pair_ = hh[i][label != label[i]]

            pos_pair = pos_pair_
            if len(pos_pair) < 1:
                continue

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            if len(neg_pair) < 1:
                continue

            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)
            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)

            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True).to(scores.device)

        loss = sum(loss) / size
        return loss


class CMPMLoss(nn.Module):
    def __init__(self, smooth=20.0, **kwargs):
        super(CMPMLoss, self).__init__()
        self.eps = 1e-8
        self.smooth = smooth
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, scores, labels=None, **kwargs):
        labels = torch.FloatTensor(labels).to(scores.device)
        label_mask = labels.view(-1, 1) - labels.view(1, -1)
        label_mask = (torch.abs(label_mask) < 0.5).float()
        label_mask_norm = F.normalize(label_mask, p=1, dim=-1)

        v2t_pred = self.softmax(self.smooth * scores)
        t2v_pred = self.softmax(self.smooth * scores.t())

        v2t_loss = kl_loss(label_mask_norm, v2t_pred, self.eps)
        t2v_loss = kl_loss(label_mask_norm, t2v_pred, self.eps)

        loss = v2t_loss.mean() + t2v_loss.mean()

        return loss


class InfoNCELoss(nn.Module):
    def __init__(self, smooth=20.0, **kwargs):
        super(InfoNCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, scores, **kwargs):
        sim_i2t = scores * self.smooth
        sim_t2i = scores.t() * self.smooth

        target = torch.zeros(sim_i2t.size()).to(scores.device)
        target.fill_diagonal_(1.0)

        loss_i2t = - torch.sum(F.log_softmax(sim_i2t, dim=1) * target, dim=1).mean()
        loss_t2i = - torch.sum(F.log_softmax(sim_t2i, dim=1) * target, dim=1).mean()

        loss = (loss_i2t + loss_t2i) / 2
        return loss


__factory = {
    'birank': BirankLoss,
    'cmpm': CMPMLoss,
    'poly': PolyLoss,
    'infonce': InfoNCELoss
}


def get_loss_function(name, **kwargs):
    if name not in __factory.keys():
        raise ValueError('Invalid value of opt.loss_name')
    return __factory[name](**kwargs)
