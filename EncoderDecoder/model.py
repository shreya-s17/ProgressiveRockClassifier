import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def weights_init(l):
    # Initilize weights for linear layers
    nn.init.xavier_uniform_(l.weight)


# @ Deepspeech.pytorch
class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.contiguous().view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class ChannelAvgPool1D(nn.AvgPool1d):
    """
    Average pool leaving spatial dims in-tact.
    Inspired by https://stackoverflow.com/questions/46562612/pytorch-maxpooling-over-channels-dimension
    """
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w*h)


class BaseRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, batch_norm=False, dropout_p=0):
        super(BaseRNN, self).__init__()
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout_p)

    def __call__(self, x, h, output_lengths, hidden=None, batch_first=False):
        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x = self.dropout(x)

        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x, h) if h else self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum

        return x, h


class AttentionRNN(BaseRNN):
    def __init__(self, input_size, hidden_size, bidirectional=False, batch_norm=False):
        super(AttentionRNN, self).__init__(input_size, hidden_size, bidirectional, batch_norm)

        self.attention = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x, h, output_lengths, hidden=None, batch_first=False):
        # enc_x = x
        # enc_x = enc_x.transpose(0, 1)  # (T x B x *) -> (B x T x *)
        # enc_h_n, enc_s_n = h  # Only valid for LSTM
        # # (B x T x H) * (B x H x 1) -> (B x T)
        # attn = torch.bmm(enc_x, enc_h_n.unsqueeze(2)).squeeze(2)
        # # (B x 1 x T)
        # attn = F.softmax(attn, dim=1).unsqueeze(1)
        # # (B x 1 x T) * (B x T x H) -> (B x H)
        # context = torch.bmm(attn, enc_x).squeeze(1)

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x = self.dropout(x)

        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, (h_n, c_n) = self.rnn(x, h) if h else self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
            h_n = h_n.view(1, 2, h_n.size(1), -1).sum(1).view(h_n.size(1), -1)  # (2xNxH) -> (1x2xNxH) -> (1x2xNxH) -> (NxH)
            # enc_h_n = enc_h_n.view(1, 2, enc_h_n.size(1), -1).sum(1).view(enc_h_n.size(1), -1)  # (2xNxH) -> (1x2xNxH) -> (1x2xNxH) -> (NxH)

        # (B x T x H) * (B x H x 1) -> (B x T)
        attn = torch.bmm(x.transpose(0, 1), h_n.unsqueeze(2)).squeeze(2)
        attn = F.softmax(attn, dim=1).unsqueeze(1)


        return x


class EncoderRNN(BaseRNN):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__(input_size, hidden_size)


class DecoderRNN(BaseRNN):
    def __init__(self, input_size, hidden_size):
        super(DecoderRNN, self).__init__(input_size, hidden_size)


class Softmax(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, fc_out):
        sm = F.softmax(fc_out, dim=-1)
        sm_bar = sm.mean(dim=self.dim)
        return sm_bar
