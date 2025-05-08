import torch
import torch.nn as nn
import torch.nn.functional as F


def l1norm(X, dim=-1, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def kl_loss(pred1, pred2, eps=1e-8):
    single_loss = torch.sum(pred2 * torch.log(eps + pred2 / (pred1 + eps)), 1)
    return single_loss


def cosine_similarity(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w2).clamp(min=eps)).squeeze()


def scan_attention(query, context, smooth):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL)
    attn = F.softmax(attn * smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext


def bfan_equal(attn, nbatch, nquery, ncontext):
    """
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (nbatch, nquery, ncontext)
    """
    funcF = attn * ncontext - torch.sum(attn, dim=-1, keepdim=True)
    attn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return attn


def bfan_prob(attn, nbatch, nquery, ncontext):
    """
    consider the confidence g(x) for each fragment as the sqrt
    of their similarity probability to the query fragment
    sigma_{j} (xi - xj)gj = sigma_{j} xi*gj - sigma_{j} xj*gj
    attn: (nbatch, nquery, ncontext)
    """
    # -> (nbatch, nquery, ncontext, 1)
    xi = attn.unsqueeze(-1).contiguous()
    # -> (nbatch, nquery, 1, ncontext)
    xj = attn.unsqueeze(2).contiguous()
    # -> (nbatch, nquery, 1, ncontext)
    xj_confi = torch.sqrt(xj)

    xi = xi.view(nbatch * nquery, ncontext, 1)
    xj = xj.view(nbatch * nquery, 1, ncontext)
    xj_confi = xj_confi.view(nbatch * nquery, 1, ncontext)

    # -> (nbatch*nquery, ncontext, ncontext)
    term1 = torch.bmm(xi, xj_confi)
    term2 = xj * xj_confi
    # -> (nbatch*nquery, ncontext)
    funcF = torch.sum(term1 - term2, dim=-1)
    funcF = funcF.view(nbatch, nquery, ncontext)

    attn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return attn


def bfan_attention(query, context, smooth, attn_type):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    nbatch, nquery, ncontext = context.size(0), query.size(1), context.size(1)

    # Step 1: preassign attention
    # --> (nbatch, d, nquery)
    queryT = torch.transpose(query, 1, 2)

    # (nbatch, ncontext, d)(nbatch, d, nquery)
    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (nbatch, nquery, ncontext)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (nbatch*nquery, ncontext)
    attn = attn.view(nbatch * nquery, ncontext)
    attn = nn.Softmax(dim=1)(attn * smooth)
    # --> (nbatch, nquery, ncontext)
    attn = attn.view(nbatch, nquery, ncontext)

    # Step 2: identify irrelevant fragments
    # Learning an indicator function H, one for relevant, zero for irrelevant
    if attn_type == 'equal':
        funcH = bfan_equal(attn, nbatch, nquery, ncontext)
    elif attn_type == 'prob':
        funcH = bfan_prob(attn, nbatch, nquery, ncontext)
    else:
        raise ValueError("Invalid value of opt.attn_type")

    # Step 3: reassign attention
    tmp_attn = funcH * attn
    attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
    re_attn = tmp_attn / attn_sum

    # --> (nbatch, d, ncontext)
    contextT = torch.transpose(context, 1, 2)
    # --> (nbatch, ncontext, nquery)
    re_attnT = torch.transpose(re_attn, 1, 2).contiguous()

    # --> (nbatch, d, nquery)
    weightedContext = torch.bmm(contextT, re_attnT)
    # --> (nbatch, nquery, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext


def get_mask_attention(attn, batch_size, sourceL, queryL, lamda=1):
    # attn --> (batch, sourceL, queryL)
    # positive attention
    mask_positive = attn.le(0)
    attn_pos = attn.masked_fill(mask_positive, torch.tensor(-1e9))
    attn_pos = torch.exp(attn_pos * lamda)
    attn_pos = l1norm(attn_pos, 1)
    attn_pos = attn_pos.view(batch_size, queryL, sourceL)

    return attn_pos


def intra_relation(K, Q, xlambda):
    """
    Q: (n_context, sourceL, d)
    K: (n_context, sourceL, d)
    return (n_context, sourceL, sourceL)
    """
    batch_size, KL = K.size(0), K.size(1)
    K = torch.transpose(K, 1, 2).contiguous()
    attn = torch.bmm(Q, K)

    attn = attn.view(batch_size*KL, KL)
    attn = nn.Softmax(dim=1)(attn*xlambda)
    attn = attn.view(batch_size, KL, KL)
    return attn


def inter_relations(attn, batch_size, sourceL, queryL, xlambda):
    """
    Q: (batch, queryL, d)
    K: (batch, sourceL, d)
    return (batch, queryL, sourceL)
    """

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * xlambda)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    return attn