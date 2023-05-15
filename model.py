import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from ddpm import DDPM


class ResBlock(nn.Module):
    def __init__(self, channels=256, spectral_norm=True, alpha=0.01, dropout=0, groups=1):
        super().__init__()
        self.c1 = nn.Conv1d(channels, channels, 5, 1, 2, groups=groups)
        self.act = nn.LeakyReLU(alpha)
        self.c2 = nn.Conv1d(channels, channels, 5, 1, 2, groups=groups)
        self.dropout_rate = dropout
        if spectral_norm:
            self.c1 = torch.nn.utils.spectral_norm(self.c1)
            self.c2 = torch.nn.utils.spectral_norm(self.c2)

    def forward(self, x):
        res = x
        x = self.act(self.c1(x))
        x = F.dropout(x, p=self.dropout_rate)
        x = self.c2(x)
        x = F.dropout(x, p=self.dropout_rate)
        return x + res


class SpectrogramEncoder(nn.Module):
    def __init__(self, n_fft=256, num_layers=4):
        super().__init__()
        self.input_layer = nn.Conv1d(n_fft // 2 + 1, 256, 5, 1, 2)
        self.mid_layers = nn.Sequential(
                *[ResBlock(256, False) for _ in range(num_layers)])

    def forward(self, x):
        return self.mid_layers(self.input_layer(x))


class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x = (x - x.mean(dim=1, keepdim=True)) / torch.sqrt(x.var(dim=1, keepdim=True) + self.eps)
        return x


class TimeEncoding1d(nn.Module):
    def __init__(self, channels, max_timesteps=10000, return_encoding_only=False):
        super().__init__()
        self.channels = channels
        self.max_timesteps = max_timesteps
        self.return_encoding_only = return_encoding_only

    # t: [batch_size]
    def forward(self, x, t):
        emb = t.unsqueeze(1).expand(t.shape[0], self.channels).unsqueeze(-1)
        e1, e2 = torch.chunk(emb, 2, dim=1)
        factors = 1 / (self.max_timesteps ** (torch.arange(self.channels//2, device=x.device) / (self.channels//2)))
        factors = factors.unsqueeze(0).unsqueeze(2)
        e1 = torch.sin(e1 * math.pi * factors)
        e2 = torch.cos(e2 * math.pi * factors)
        emb = torch.cat([e1, e2], dim=1).expand(*x.shape)

        ret = emb if self.return_encoding_only else x + emb
        return ret


class GeneratorResBlock(nn.Module):
    def __init__(self, channels, condition_channels=256):
        super().__init__()
        self.time_enc = TimeEncoding1d(channels)
        self.spec_conv = nn.Conv1d(condition_channels, channels, 1, 1, 0)
        self.conv = nn.Conv1d(channels, channels, 7, 1, padding='same', dilation=1)
        self.norm = ChannelNorm(channels)
        self.act = nn.GELU()

    def forward(self, x, t, spec):
        res = x
        x = self.norm(x)
        x = self.time_enc(x, t)
        x = x * F.interpolate(self.spec_conv(spec), x.shape[2])
        x = self.act(self.conv(x))
        return x + res


class GeneratorResStack(nn.Module):
    def __init__(self, channels, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList(
                [GeneratorResBlock(channels) for _ in range(num_layers)])

    def forward(self, x, t, spec):
        for layer in self.layers:
            x = layer(x, t, spec)
        return x


class Vocoder(nn.Module):
    def __init__(self, layers=[4, 4, 4, 4], channels=[32, 64, 128, 256], downsample_rate=[4, 4, 4, 4]):
        super().__init__()
        self.input_conv = nn.Conv1d(1, channels[0], 7, 1, 3)
        self.output_conv = nn.Conv1d(channels[0], 1, 7, 1, 3)
        self.downsamples = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])
        self.encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        
        rate_total = 1
        for l, c, c_next, r, in zip(layers, channels, channels[1:]+[channels[-1]], downsample_rate):
            rate_total = rate_total * r
            self.downsamples.append(nn.Conv1d(c, c_next, r * 2, r, r // 2))
            self.upsamples.insert(0, nn.ConvTranspose1d(c_next, c, r* 2, r, r // 2))
            self.encoder_layers.append(GeneratorResStack(c, num_layers=l))
            self.decoder_layers.insert(0, GeneratorResStack(c, num_layers=l))

    def forward(self, x, time, condition):
        # padding
        if x.shape[1] % 256 != 0:
            pad_len = (256 - x.shape[1] % 256)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)

        x = x.unsqueeze(1)
        x = self.input_conv(x)
        skips = []
        for layer, ds in zip(self.encoder_layers, self.downsamples):
            skips.append(x)
            x = layer(x, time, condition)
            x = ds(x)
        for layer, us, s in zip(self.decoder_layers, self.upsamples, reversed(skips)):
            x = us(x)
            x = layer(x, time, condition)
            x = x + s

        x = self.output_conv(x)
        x = x.squeeze(1)
        return x


class DiffusionVocoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocoder = DDPM(Vocoder())
        self.spectrogram_encoder = SpectrogramEncoder()
        self.to_spectrogram = torchaudio.transforms.Spectrogram(
                n_fft=256
                )
