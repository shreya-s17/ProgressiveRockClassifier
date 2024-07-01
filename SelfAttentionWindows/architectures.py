import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from SelfAttentionWindows.model import BaseRNN, SelfAttention, ResidualBlock, weights_init, MaskConv


class LSfcAttX(nn.Module):
    def __init__(self, opt):
        super(LSfcAttX, self).__init__()

        self.resnet = MaskConv(nn.Sequential(
            nn.Conv1d(20, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResidualBlock(256, 1024, stride=1, downsample=True),
            ResidualBlock(1024, 1024, stride=1, downsample=True),
            ResidualBlock(1024, 1024, stride=1, downsample=True),
            # ResidualBlock(512, 512, stride=1, downsample=True),
            # Down to 1 channel
            # nn.Conv1d(512, 1, kernel_size=3, stride=1),
            # nn.BatchNorm1d(1),
            # nn.ReLU()
        ))

        # self.lstm = BaseRNN(113, 512, batch_norm=True, bidirectional=True, dropout_p=0)  # 104
        self.lstm = BaseRNN(1024, 512, batch_norm=True, bidirectional=True, dropout_p=0)  # 104
        self.lstm2 = BaseRNN(512, 512, batch_norm=True, bidirectional=True, dropout_p=0)  # 104
        self.self_attention = SelfAttention(512, 32, context_hops=1)

        # self.lstm2 = AttentionRNN(512, 512, bidirectional=True, batch_norm=True)

        # self.fc0 = nn.Linear(512, 32)

        # self.bn = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(512, 2, bias=False)

        # weights_init(self.fc0)
        # weights_init(self.fc1)

    def forward(self, inputs, input_percentages, give_attention=False):
        real_lengths = (inputs.shape[3] * input_percentages.cpu()).int()

        inputs = inputs.squeeze(1)
        # inputs = self.conv(inputs)
        inputs, real_lengths = self.resnet(inputs, real_lengths)
        inputs = inputs.unsqueeze(1)

        # inputs = torch.mean(inputs, dim=1)
        # inputs = inputs.unsqueeze(1)

        # inputs = inputs.transpose(1, 3).transpose(1, 2)
        # Squeeze channel dim, transpose seq len with feature len

        inputs = inputs.squeeze(1).transpose(1, 2)

        # Just transpose to (T x B x *) since everything expects batch to be second...
        inputs = inputs.transpose(0, 1)

        x, h = self.lstm(inputs, None, real_lengths)
        x, h = self.lstm2(x, h, real_lengths)

        # Transpose back to (B x T x *)
        x = x.transpose(0, 1)
        # Attention weights matrix (B x vector if context_hops == 1)
        # (B x T  x r)
        A = self.self_attention(x)

        # (B x T x r) x (B x T x H)
        M = A * x
        x = M

        # x = torch.tanh(self.fc0(x))
        # (B x H)
        x = x.sum(dim=1)

        # x = self.bn(x)

        # (B x C x H) x (B x H)^T = (B x C)
        wx = self.fc1(x)
        # wx = self.fc2(wx)
        # wx = wx.mean(dim=1)
        if give_attention:
            return wx, A
        else:
            return wx


class lsfc100res(nn.Module):
    def __init__(self, opt):
        super(lsfc100res, self).__init__()
        self.resnet = MaskConv(nn.Sequential(
            nn.Conv1d(20, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResidualBlock(256, 1024, stride=1, downsample=True),
            ResidualBlock(1024, 1024, stride=1, downsample=True),
            ResidualBlock(1024, 1024, stride=1, downsample=True),
        ))

        self.lstm = BaseRNN(1024, 512, batch_norm=True, bidirectional=True, dropout_p=0)  # 104
        self.lstm2 = BaseRNN(512, 512, batch_norm=True, bidirectional=True, dropout_p=0)  # 104

        self.fc0 = nn.Linear(512, 32)

        self.bn = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32, 2, bias=False)

    def forward(self, inputs, input_percentages):
        real_lengths = (inputs.shape[3] * input_percentages.cpu()).int()

        inputs = inputs.squeeze(1)
        inputs, real_lengths = self.resnet(inputs, real_lengths)
        inputs = inputs.unsqueeze(1)

        inputs = inputs.squeeze(1).transpose(1, 2)

        # Just transpose to (T x B x *) since everything expects batch to be second...
        inputs = inputs.transpose(0, 1)

        x, h = self.lstm(inputs, None, real_lengths)
        x, h = self.lstm2(x, h, real_lengths)
        # x, h = self.lstm3(x, real_lengths, batch_first=True)
        # x, h = self.lstm31(x, real_lengths, batch_first=True)
        # x, h = self.lstm32(x, real_lengths, batch_first=True)
        # x, h = self.lstm4(x, real_lengths, batch_first=True)

        # x = self.lstm3(x, real_lengths, batch_first=True)
        # x = self.lstm4(x, real_lengths, batch_first=True)
        # x = self.lstm5(x, real_lengths, batch_first=True)
        # x = self.lstm6(x, real_lengths, batch_first=True)

        # Transpose back to (B x T x *)
        x = x.transpose(0, 1)
        x = torch.tanh(self.fc0(x))
        x = torch.mean(x, dim=1)

        x = self.bn(x)
        wx = self.fc1(x)
        # wx = self.fc2(wx)
        # wx = wx.mean(dim=1)

        return wx


class lsfc100(nn.Module):
    def __init__(self, opt):
        super(lsfc100, self).__init__()

        self.lstm = BaseRNN(20, 512, batch_norm=True, bidirectional=True, dropout_p=0)  # 104
        self.lstm2 = BaseRNN(512, 512, batch_norm=True, bidirectional=True, dropout_p=0)  # 104
        # self.lstm2 = AttentionRNN(512, 512, bidirectional=True, batch_norm=True)

        # self.lstm3 = BaseRNN(2048, 2048, batch_norm=True, bidirectional=True)  # 104
        # self.lstm31 = BaseRNN(2048, 2048, batch_norm=True, bidirectional=True)  # 104
        # self.lstm32 = BaseRNN(2048, 2048, batch_norm=True, bidirectional=True)  # 104
        # self.lstm4 = BaseRNN(256, 3, batch_norm=True, bidirectional=True, dropout_p=0)  # 104
        # self.lstm4 = BaseRNN(2048, 2048, batch_norm=True, bidirectional=True)  # 104

        self.fc0 = nn.Linear(512, 32)

        # self.lstm2 = BaseRNN(512, 256)
        # self.lstm3 = BaseRNN(1, 1, batch_norm=True, bidirectional=True)
        # self.lstm4 = BaseRNN(256, 256)
        # self.lstm5 = BaseRNN(1024, 512)

        # self.lstm6 = BaseRNN(512, 128)
        self.bn = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32, 2, bias=False)
        # self.fc2 = nn.Linear(2, 2, bias=False)

        # weights_init(self.fc0)
        # weights_init(self.fc1)

    def forward(self, inputs, input_percentages):

        # inputs = self.conv(inputs)
        # inputs = torch.mean(inputs, dim=1)
        # inputs = inputs.unsqueeze(1)
        real_lengths = (inputs.shape[3] * input_percentages.cpu()).int()

        # inputs = inputs.transpose(1, 3).transpose(1, 2)
        # Squeeze channel dim, transpose seq len with feature len
        inputs = inputs.squeeze(1).transpose(1, 2)

        # Just transpose to (T x B x *) since everything expects batch to be second...
        inputs = inputs.transpose(0, 1)

        x, h = self.lstm(inputs, None, real_lengths)
        x, h = self.lstm2(x, h, real_lengths)
        # x, h = self.lstm3(x, real_lengths, batch_first=True)
        # x, h = self.lstm31(x, real_lengths, batch_first=True)
        # x, h = self.lstm32(x, real_lengths, batch_first=True)
        # x, h = self.lstm4(x, real_lengths, batch_first=True)

        # x = self.lstm3(x, real_lengths, batch_first=True)
        # x = self.lstm4(x, real_lengths, batch_first=True)
        # x = self.lstm5(x, real_lengths, batch_first=True)
        # x = self.lstm6(x, real_lengths, batch_first=True)

        # Transpose back to (B x T x *)
        x = x.transpose(0, 1)
        x = torch.tanh(self.fc0(x))
        x = torch.mean(x, dim=1)

        x = self.bn(x)
        wx = self.fc1(x)
        # wx = self.fc2(wx)
        # wx = wx.mean(dim=1)

        return wx
