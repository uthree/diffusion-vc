import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from ddpm import DDPM


class SpeakerEncoder(nn.Module):
    def __init__(self, d_spk=256):
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
                n_fft=512,
                n_mels=80
                )
        self.layers = nn.Sequential(
                nn.Conv1d(80, 64, 4, 1, 2),
                nn.GELU(),
                nn.Conv1d(64, 64, 4, 1, 2),
                nn.GELU(),
                nn.Conv1d(64, 128, 4, 1, 2),
                nn.GELU(),
                nn.Conv1d(128, 128, 4, 1, 2),
                nn.GELU(),
                nn.Conv1d(128, 128, 4, 1, 2),
                nn.GELU(),
                nn.Conv1d(128, 256, 4, 1, 2),
                nn.GELU(),
                nn.Conv1d(256, 256, 4, 1, 2),
                nn.GELU(),
                nn.Conv1d(256, 256, 4, 1, 2),
                nn.GELU(),
                nn.Conv1d(256, d_spk, 4, 1, 2),
                )
    
    def forward(self, x):
        x = self.to_mel(x)
        x = self.layers(x)
        # Spatial mean
        x = x.mean(dim=2, keepdim=True)
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


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 5, 1, 2)
        self.conv2 = nn.Conv1d(channels, channels, 5, 1, 2)
        self.act = nn.GELU()

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x))) + x


class ResStack(nn.Module):
    def __init__(self, channels, num_layers=2):
        super().__init__()
        self.layers = nn.Sequential(*[ResBlock(channels) for _ in range(num_layers)])

    def forward(self, x):
        return self.layers(x)


class ContentEncoder(nn.Module):
    def __init__(self, d_con=4):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Conv1d(1, 32, 5, 1, 2),
                ResStack(32),
                nn.Conv1d(32, 64, 8, 4, 2),
                ResStack(64),
                nn.Conv1d(64, 128, 8, 4, 2),
                ResStack(128),
                nn.Conv1d(128, 256, 8, 4, 2),
                ResStack(256),
                nn.Conv1d(256, 256, 8, 4, 2),
                nn.Conv1d(256, d_con, 5, 1, 2)
                )


    def forward(self, x):
        # padding
        if x.shape[1] % 256 != 0:
            pad_len = (256 - x.shape[1] % 256)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)
        x = x.unsqueeze(1)
        x = self.layers(x)
        return x


class Condition:
    def __init__(self, content, speaker):
        self.content = content
        self.speaker = speaker


class GeneratorResBlock(nn.Module):
    def __init__(self, channels, d_con=4, d_spk=256):
        super().__init__()
        self.time_enc = TimeEncoding1d(channels)
        self.spk = nn.Conv1d(d_spk, channels, 1, 1, 0)
        self.conv1 = nn.Conv1d(channels, channels, 7, 1, padding='same', dilation=1)
        self.conv2 = nn.Conv1d(channels, channels, 7, 1, padding='same', dilation=2)
        self.conv3 = nn.Conv1d(channels, channels, 7, 1, padding='same', dilation=3)
        self.act = nn.GELU()

    def forward(self, x, t, spk):
        res = x
        x = self.time_enc(x, t)
        x = x * self.spk(spk)
        o1 = self.act(self.conv1(x))
        o2 = self.act(self.conv2(x))
        o3 = self.act(self.conv3(x))
        return o1 + o2 + o3 + res


class GeneratorResStack(nn.Module):
    def __init__(self, channels, d_con=4, d_spk=256, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList(
                [GeneratorResBlock(channels, d_con, d_spk) for _ in range(num_layers)])

    def forward(self, x, t, spk):
        for layer in self.layers:
            x = layer(x, t, spk)
        return x


class Generator(nn.Module):
    def __init__(self, d_con=4, layers=[2, 2, 2, 2], channels=[32, 64, 128, 256], downsample_rate=[4, 4, 4, 4]):
        super().__init__()
        self.input_conv = nn.Conv1d(1, channels[0], 7, 1, 3)
        self.output_conv = nn.Conv1d(channels[0], 1, 7, 1, 3)
        self.downsamples = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])
        self.encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        self.content_convs = nn.ModuleList([])
        
        rate_total = 1
        for l, c, c_next, r, in zip(layers, channels, channels[1:]+[channels[-1]], downsample_rate):
            rate_total = rate_total * r
            self.downsamples.append(nn.Conv1d(c, c_next, r * 2, r, r // 2))
            self.upsamples.insert(0, nn.ConvTranspose1d(c_next, c, r* 2, r, r // 2))
            self.encoder_layers.append(GeneratorResStack(c, l))
            self.decoder_layers.insert(0, GeneratorResStack(c, l))
            self.content_convs.insert(0,
                            nn.Conv1d(d_con, c_next, 1, 1, 0))

    def forward(self, x, time, condition):
        # padding
        if x.shape[1] % 256 != 0:
            pad_len = (256 - x.shape[1] % 256)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)

        spk = condition.speaker
        con = condition.content
        x = x.unsqueeze(1)
        x = self.input_conv(x)
        skips = []
        for layer, ds in zip(self.encoder_layers, self.downsamples):
            skips.append(x)
            x = layer(x, time, spk)
            x = ds(x)
        for layer, us, s, c in zip(self.decoder_layers, self.upsamples, reversed(skips), self.content_convs):
            x = x * c(F.interpolate(con, size=x.shape[2]))
            x = us(x)
            x = layer(x, time, spk)
            x = x + s

        x = self.output_conv(x)
        x = x.squeeze(1)
        return x

class DiffusionVC(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_encoder = ContentEncoder()
        self.speaker_encoder = SpeakerEncoder()
        self.generator = DDPM(Generator())
