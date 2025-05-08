import torch
import torch.nn.init
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
import numpy as np

from vsrn.GCN_lib.Rs_GCN import Rs_GCN
import scipy.linalg as linalg


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, use_abs=False,
                 no_imgnorm=False, sim_type=None, sim_dim=None):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    img_enc = EncoderImagePrecompAttn(
        img_dim, embed_size, data_name, use_abs, no_imgnorm, sim_type, sim_dim)

    return img_enc


class EncoderImagePrecompAttn(nn.Module):

    def __init__(self, img_dim, embed_size, data_name, use_abs=False,
                 no_imgnorm=False, sim_type=None, sim_dim=None):
        super(EncoderImagePrecompAttn, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.data_name = data_name
        
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

        # GSR
        self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)

        # GCN reasoning 
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)

        # --------------- similarity processing ----------------#
        self.sim_type = sim_type  # baseline, vector, matrix
        self.sim_dim = sim_dim

        if self.sim_type == 'vector':
            self.sim_tran_module = nn.Parameter(torch.ones(embed_size), requires_grad=True)
            self.mask = torch.ones(self.embed_size).cuda()
        elif self.sim_type == 'matrix':
            self.n_block = embed_size // sim_dim
            self.sim_tran_module = nn.Parameter(torch.eye(embed_size), requires_grad=True)

            ones_matrix = (np.ones((sim_dim, sim_dim)) for _ in range(self.n_block))
            block_matrix = linalg.block_diag(*ones_matrix)
            self.mask = torch.FloatTensor(block_matrix).cuda()
        # ------------------------------------------------------#
        # self.sim_tran_module = nn.Linear(embed_size, embed_size, bias=False)
        # nn.init.eye_(self.sim_tran_module.weight)

        if self.data_name == 'f30k_precomp':
            self.bn = nn.BatchNorm1d(embed_size)        

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""

        fc_img_emd = self.fc(images)
        if self.data_name != 'f30k_precomp':
            fc_img_emd = l2norm(fc_img_emd)

        # GCN reasoning
        # -> B,D,N
        GCN_img_emd = fc_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

        GCN_img_emd = l2norm(GCN_img_emd)

        rnn_img, hidden_state = self.img_rnn(GCN_img_emd)

        # features = torch.mean(rnn_img,dim=1)
        features = hidden_state[0]

        if self.data_name == 'f30k_precomp':
            features = self.bn(features)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

            # --------------- similarity processing ----------------#
            if self.sim_type == 'vector':
                weights = self.mask * self.sim_tran_module
                features = features * weights
            elif self.sim_type == 'matrix':
                weights = self.mask * self.sim_tran_module
                features = features @ weights

            # ------------------------------------------------------#
            # a = weights.data.cpu().numpy()
            # features = self.sim_tran_module(features)

            # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features, GCN_img_emd


# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        x = self.dropout(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out


class VSRN(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        # tutorials/09 - Image Captioning
        # Build Models
        self.grad_clip = opt.grad_clip

        self.img_enc = EncoderImage(opt.data_name,
                                    opt.img_dim,
                                    opt.embed_size,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm,
                                    sim_type=opt.sim_type,
                                    sim_dim=opt.sim_dim)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_abs=opt.use_abs)

        self.img_enc.cuda()
        self.txt_enc.cuda()
        cudnn.benchmark = True

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def freeze_parameter(self):

        unfreeze_param = list()
        for name, param in self.img_enc.named_parameters():
            if 'sim_tran_module' in name:
                param.requires_grad = True
                unfreeze_param.append(param)
            else:
                param.requires_grad = False

        for name, param in self.txt_enc.named_parameters():
            param.requires_grad = False

        return unfreeze_param

    def train_forward(self, images, captions, lengths):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        cap_emb = self.txt_enc(captions, lengths)
        img_emb, GCN_img_emd = self.img_enc(images)

        return img_emb.mm(cap_emb.t())
