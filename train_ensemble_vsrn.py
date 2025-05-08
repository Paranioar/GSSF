"""Training script"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import kl_loss
from lib.vocab import deserialize_vocab
from lib.datasets import image_caption
from lib.model import Framework
from lib.loss import get_loss_function
from lib.evaluation import evalrank, AverageMeter, LogCollector

from vsrn import VSRN

import logging
import tensorboard_logger as tb_logger
from torch.nn.utils import clip_grad_norm_

import arguments


def main():
    # Hyper Parameters
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuid

    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    logger = logging.getLogger(__name__)
    logger.info(opt)

    # Load Vocabulary
    if 'coco' in opt.data_name:
        vocab_file = 'coco_precomp_vocab.json'
    else:
        vocab_file = 'f30k_precomp_vocab.json'

    vocab = deserialize_vocab(os.path.join(opt.vocab_path, vocab_file))
    opt.vocab_size = len(vocab)

    train_loader, val_loader = image_caption.get_loaders(
        opt.data_path, opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    model = Framework(opt)
    model.cuda()
    model.Eiters = 0

    resume = 'vsrn/f30k_baseline_dp/model_best.pth.tar'

    # optionally resume from a checkpoint
    if os.path.isfile(resume):
        logger.info("=> loading checkpoint '{}'".format(resume))
        checkpoint0 = torch.load(resume)
        opt0 = checkpoint0['opt']
        opt0.vocab_size = len(vocab)
        opt0.sim_type, opt0.sim_dim = 'baseline', 2
        if opt.distill_type == 'vector':
            opt0.sim_type = 'vector'
        elif opt.distill_type == 'matrix':
            opt0.sim_type = 'matrix'

        logger.info(opt0)

        model0 = VSRN(opt0)
        model0.load_state_dict(checkpoint0['model'])
        teacher_param = model0.freeze_parameter()

        logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                    .format(resume, checkpoint0['epoch'], checkpoint0['best_rsum']))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume))

    lr_schedules = [opt.lr_update, ]
    optimizer = torch.optim.AdamW([
            {'params': teacher_param, 'lr': opt.learning_rate * opt.distill_factor},
            {'params': model.parameters(), 'lr': opt.learning_rate},
        ], lr=opt.learning_rate)

    criterion = get_loss_function(name=opt.loss_name,
                                  margin=opt.margin,
                                  smooth=opt.loss_smooth,
                                  max_violation=opt.max_violation)
    criterion_coop = DMLLoss(smooth=opt.distill_smooth)

    # Train the Model
    best_rsum = 0
    for epoch in range(opt.num_epochs):
        logger.info(opt.logger_name)
        logger.info(opt.model_name)

        adjust_learning_rate(optimizer, epoch, lr_schedules)

        # average meters to record the training statistics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_logger = LogCollector()
        logger.info('trainable parameters: {}'.format(count_params(model)))

        end = time.time()
        for i, (images, captions, lengths, ids, img_ids) in enumerate(train_loader):
            # switch to train mode
            model.train()
            model0.train_start()

            # measure data loading time
            data_time.update(time.time() - end)
            model.logger = train_logger

            model.Eiters += 1
            model.logger.update('Eit', model.Eiters)
            model.logger.update('Lr', optimizer.param_groups[0]['lr'])

            # Update the model
            optimizer.zero_grad()

            sims = model(images, captions, lengths)
            sims0 = model0.train_forward(images, captions, lengths)

            loss_raw = criterion(sims.t(), labels=img_ids)
            loss_raw0 = criterion(sims0, labels=img_ids)
            loss_coop = criterion_coop(sims.t(), sims0)
            loss = loss_raw + loss_coop

            model.logger.update('Loss_raw', loss_raw.item(), sims.size(1))
            model.logger.update('Loss_raw0', loss_raw0.item(), sims.size(1))
            model.logger.update('Loss_coop', loss_coop.item(), sims.size(1))
            model.logger.update('Loss_all', loss.item(), sims.size(1))

            loss.backward()
            if opt.grad_clip > 0:
                clip_grad_norm_(model.parameters(), opt.grad_clip)
                clip_grad_norm_(teacher_param, opt.grad_clip)
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print log info
            if model.Eiters % opt.log_step == 0:
                logging.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        .format(epoch, i, len(train_loader), batch_time=batch_time,
                                data_time=data_time, e_log=str(model.logger)))

            # Record logs in tensorboard
            tb_logger.log_value('epoch', epoch, step=model.Eiters)
            tb_logger.log_value('step', i, step=model.Eiters)
            tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
            tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
            model.logger.tb_log(tb_logger, step=model.Eiters)

        # evaluate on validation set
        rsum = evalrank(model, val_loader, opt, step=model.Eiters)
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint.pth', prefix=opt.model_name + '/')


def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):
    logger = logging.getLogger(__name__)
    tries = 15

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + 'model_best.pth')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(optimizer, epoch, lr_schedules):
    logger = logging.getLogger(__name__)
    """Sets the learning rate to the initial LR
       decayed by 10 every opt.lr_update epochs"""
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.1
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class DMLLoss(nn.Module):
    def __init__(self, smooth=10.0):
        super(DMLLoss, self).__init__()
        self.smooth = smooth
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, scores, scores_t):

        v2t_loss = self.loss_kl(F.log_softmax(self.smooth * scores, dim=-1),
                                F.softmax(self.smooth * scores_t, dim=-1))
        t2v_loss = self.loss_kl(F.log_softmax(self.smooth * scores.t(), dim=-1),
                                F.softmax(self.smooth * scores_t.t(), dim=-1))

        loss = v2t_loss + t2v_loss

        return loss


class DistillationLoss(nn.Module):
    def __init__(self, smooth=10.0, eps=1e-8):
        super(DistillationLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, scores, scores_t):

        v2t_pred = self.softmax(self.smooth * scores)
        t2v_pred = self.softmax(self.smooth * scores.t())
        v2t_pred_t = self.softmax(self.smooth * scores_t)
        t2v_pred_t = self.softmax(self.smooth * scores_t.t())

        v2t_loss = kl_loss(v2t_pred_t, v2t_pred, self.eps)
        t2v_loss = kl_loss(t2v_pred_t, t2v_pred, self.eps)

        loss = v2t_loss.mean() + t2v_loss.mean()

        return loss


if __name__ == '__main__':
    main()
