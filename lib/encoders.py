"""VSE modules"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

from lib.utils import l2norm

import logging
logger = logging.getLogger(__name__)


def get_image_encoder(img_dim, embed_size, no_imgnorm=False, **kwargs):
    return EncoderImage(img_dim, embed_size, no_imgnorm=no_imgnorm)


def get_text_encoder(vocab_size, embed_size, word_dim, no_txtnorm=False, **kwargs):
    return EncoderText(vocab_size, embed_size, word_dim, no_txtnorm=no_txtnorm)


class EncoderImage(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False, **kwargs):
        super(EncoderImage, self).__init__()

        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, **kwargs):
        """Extract image feature vectors."""
        features = self.fc(images)

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features


# Language Model with BiGRU
class EncoderText(nn.Module):

    def __init__(self, vocab_size, embed_size, word_dim, no_txtnorm=False, **kwargs):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers=1, batch_first=True, bidirectional=True)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths, **kwargs):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)
        x_emb = self.dropout(x_emb)

        sorted_lengths, indices = torch.sort(lengths, descending=True)
        x_emb = x_emb[indices]
        inv_ix = indices.clone()
        inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)

        packed = pack_padded_sequence(x_emb, sorted_lengths.data.tolist(), batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()

        # Forward propagate RNN
        out, _ = self.rnn(packed)
        cap_emb, _ = pad_packed_sequence(out, batch_first=True)
        cap_emb = cap_emb[inv_ix]
        cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) // 2)] + cap_emb[:, :, int(cap_emb.size(2) // 2):]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        # For multi-GPUs
        if cap_emb.size(1) < x_emb.size(1):
            pad_size = x_emb.size(1) - cap_emb.size(1)
            pad_emb = torch.zeros(cap_emb.size(0), pad_size, cap_emb.size(2)).to(cap_emb.device)
            cap_emb = torch.cat([cap_emb, pad_emb], 1)

        return cap_emb
