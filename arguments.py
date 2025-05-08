import argparse


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/f30k',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./data/vocab',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--logger_name', default='./runs/test/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/test',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--reset_start_epoch', action='store_true',
                        help='Whether restart the start epoch when load weights')

    parser.add_argument('--gpuid', default='0', type=str,
                        help='The number of gpuid')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=200, type=int,
                        help='Number of steps to logger.info and record the log.')
    parser.add_argument('--optim_name', default='AdamW', type=str,
                        help='Name of optimizer')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')

    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--sim_dim', default=1024, type=int,
                        help='Dimensionality of the sim embedding.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--no_simnorm', action='store_true',
                        help='Do not normalize the sim embeddings.')
    parser.add_argument('--use_sim_metric', action='store_true',
                        help='Use similarity vector representation.')
    parser.add_argument('--use_sim_matrix', action='store_true',
                        help='Use similarity matrix representation.')

    parser.add_argument('--loss_name', default='birank', type=str,
                        help='Name of loss function')
    parser.add_argument('--loss_smooth', default=20, type=float,
                        help='Temperature of loss function.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_false',
                        help='Use max instead of sum in the rank loss.')

    parser.add_argument('--attn_name', default='none', type=str,
                        help='Name of framework')
    parser.add_argument('--attn_type', default='none', type=str,
                        help='Type of attention')
    parser.add_argument('--attn_dim', default=256, type=int,
                        help='Dimensionality of the attention alignments.')
    parser.add_argument('--attn_layer', default=3, type=int,
                        help='Number of attention layer.')
    parser.add_argument('--attn_smooth', default=9.0, type=float,
                        help='Smooth of attention temperature.')
    parser.add_argument('--attn_dp', default=0, type=float,
                        help='Dropout rate.')

    parser.add_argument('--mean_neg', default=0, type=float,
                        help='Mean value of mismatched distribution.')
    parser.add_argument('--stnd_neg', default=0, type=float,
                        help='Standard deviation of mismatched distribution.')
    parser.add_argument('--mean_pos', default=0, type=float,
                        help='Mean value of matched distribution.')
    parser.add_argument('--stnd_pos', default=0, type=float,
                        help='Standard deviation of matched distribution.')
    parser.add_argument('--thres', default=0, type=float,
                        help='Optimal learning  boundary.')
    parser.add_argument('--thres_safe', default=0, type=float,
                        help='Optimal learning  boundary.')
    parser.add_argument('--alpha', default=2.0, type=float,
                        help='Initial penalty parameter.')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='Weight of the positive similarity.')
    parser.add_argument('--using_intra_info', action='store_true',
                        help='Intra-modal propagation in the inference.')

    parser.add_argument('--distill_name', default='none', type=str,
                        help='Name of distillation (vse, scan_t2i)')
    parser.add_argument('--distill_type', default='none', type=str,
                        help='Type of distillation (cosine, vector, matrix)')
    parser.add_argument('--distill_smooth', default=10.0, type=float,
                        help='Temperature of distillation.')
    parser.add_argument('--distill_factor', default=1.0, type=float,
                        help='Factor of raw learning rate for distillation.')

    return parser
