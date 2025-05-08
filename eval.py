import os
import logging

import torch
import torch.nn as nn

from lib.datasets import image_caption
from lib.model import Framework
from lib import evaluation
from lib.vocab import Vocabulary, deserialize_vocab

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main(model_path, split, gpuid='0', fold5=False):
    print("use GPU:", gpuid)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    # load vocabulary used by the model
    if 'coco' in opt.data_name:
        vocab_file = 'coco_precomp_vocab.json'
    else:
        vocab_file = 'f30k_precomp_vocab.json'

    vocab = deserialize_vocab(os.path.join(opt.vocab_path, vocab_file))
    opt.vocab_size = len(vocab)

    # construct model
    model = Framework(opt)
    model.cuda()
    model = nn.DataParallel(model)

    # load model state
    model.load_state_dict(checkpoint['model'])

    data_loader = image_caption.get_test_loader(split, opt.data_name, vocab,
                                                opt.batch_size, opt.workers, opt)

    logger.info(opt)
    logger.info('Computing results with checkpoint_{}'.format(checkpoint['epoch']))

    evaluation.evalrank(model.module, data_loader, opt, split, fold5)


if __name__ == '__main__':

    # main('runs/f30k_scan_baseline_t2i_dp/model_best.pth', 'test', '1', False)
    # main('runs/f30k_scan_baseline_i2t_dp/model_best.pth', 'test', '1', False)
    main('runs/f30k_scan_vector_t2i_dp/model_best.pth', 'test', '1', False)
    main('runs/f30k_scan_vector_i2t_dp/model_best.pth', 'test', '1', False)
    main('runs/f30k_scan_matrix_t2i_dp_sim256/model_best.pth', 'test', '1', False)
    main('runs/f30k_scan_matrix_i2t_dp_sim256/model_best.pth', 'test', '1', False)

    # main('runs/distill_f30k_scan_t2i_by_naaf_use_cosine/model_best.pth', 'test', '0', False)
    # main('runs/distill_f30k_scan_t2i_by_naaf_use_vector/model_best.pth', 'test', '0', False)
    # main('runs/distill_f30k_scan_t2i_by_naaf_use_matrix/model_best.pth', 'test', '0', False)

    # main('runs/distill_f30k_vse_by_vsrn_use_cosine_e3/model_best.pth', 'test', '0', False)
    # main('runs/distill_f30k_vse_by_vsrn_use_vector_e3/model_best.pth', 'test', '0', False)
    # main('runs/distill_f30k_vse_by_vsrn_use_matrix_e3/model_best.pth', 'test', '0', False)

    # main('runs/distill_f30k_vse_by_scan_t2i_use_cosine/model_best.pth', 'test', '3', False)
    # main('runs/distill_f30k_vse_by_scan_t2i_use_vector/model_best.pth', 'test', '3', False)
    # main('runs/distill_f30k_vse_by_scan_t2i_use_matrix/model_best.pth', 'test', '3', False)
    #
    # main('runs/distill_f30k_vse_by_vse_use_cosine/model_best.pth', 'test', '3', False)
    # main('runs/distill_f30k_vse_by_vse_use_vector/model_best.pth', 'test', '3', False)
    # main('runs/distill_f30k_vse_by_vse_use_matrix/model_best.pth', 'test', '3', False)
    #
    # main('runs/distill_f30k_vse_by_scan_t2i_use_cosine/model_best.pth', 'test', '3', False)
    # main('runs/distill_f30k_vse_by_scan_t2i_use_vector/model_best.pth', 'test', '3', False)
    # main('runs/distill_f30k_vse_by_scan_t2i_use_matrix/model_best.pth', 'test', '3', False)

    # main('runs/coco_vse_cosine_dp/model_best.pth', 'testall', '1', True)
    # main('runs/coco_vse_cosine_dp/model_best.pth', 'testall', '1', False)
    # main('runs/coco_vse_metric_dp/model_best.pth', 'testall', '1', True)
    # main('runs/coco_vse_metric_dp/model_best.pth', 'testall', '1', False)
    # main('runs/coco_vse_matrix_dp_sim2/model_best.pth', 'testall', '1', True)
    # main('runs/coco_vse_matrix_dp_sim2/model_best.pth', 'testall', '1', False)
    # main('runs/coco_vse_matrix_dp_sim4/model_best.pth', 'testall', '1', True)
    # main('runs/coco_vse_matrix_dp_sim4/model_best.pth', 'testall', '1', False)
    # main('runs/coco_vse_matrix_dp_sim8/model_best.pth', 'testall', '1', True)
    # main('runs/coco_vse_matrix_dp_sim8/model_best.pth', 'testall', '1', False)
    # main('runs/coco_vse_matrix_dp_sim16/model_best.pth', 'testall', '1', True)
    # main('runs/coco_vse_matrix_dp_sim16/model_best.pth', 'testall', '1', False)

    # mask = torch.eye(3)
    # tensorA = torch.randn(3, 3)
    # diag_A = torch.abs(tensorA.diag())
    # tensorB = torch.diag(diag_A) * mask + (1 - mask) * tensorA
    # print(tensorA)
    # print(diag_A)
    # print(tensorB)
    # tensor = torch.eye(3, 3)
    # print(min(tensor[1][1][False]))
    print('finished')
    print('hello')
