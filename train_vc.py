import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from tqdm import tqdm

from model import DiffusionVC
from dataset import WaveFileDirectory


parser = argparse.ArgumentParser(description="run training")

parser.add_argument('dataset')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=10000, type=int)
parser.add_argument('-b', '--batch-size', default=64, type=int)
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('-len', '--length', default=65536, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-gacc', '--gradient-accumulation', default=4, type=int)

args = parser.parse_args()

device = torch.device(args.device)

def load_or_init_model(device=torch.device('cpu')):
    model = DiffusionVC().to(device)
    if os.path.exists('./convertor.pt'):
        model.load_state_dict(torch.load('./convertor.pt', map_location=device))
    return model

def save_model(model):
    torch.save(model.state_dict(), './convertor.pt')

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

model = load_or_init_model(device=device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

ds = WaveFileDirectory(
        [args.dataset],
        length=args.length,
        max_files=args.max_data
        )

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

grad_acc = args.gradient_accumulation

weight_kl = 0.0

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))

    for batch, wave in enumerate(dl):
        N = wave.shape[0]
        wave = wave.to(device)
        
        if batch % grad_acc == 0:
            optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            spec = model.to_spectrogram(wave)
            mean, logvar = model.speaker_encoder(spec)
            speaker = mean + torch.exp(logvar) * torch.randn(*logvar.shape, device=logvar.device)
            loss_kl = (-1 -logvar + torch.exp(logvar) + mean**2).mean()
            content = model.content_encoder(spec)
            loss_ddpm = model.generator.calculate_loss(spec, condition=(content, speaker))
            loss = loss_ddpm + loss_kl * weight_kl

        scaler.scale(loss).backward()
        if batch % grad_acc == 0:
            scaler.step(optimizer)
            scaler.update()
        
        bar.set_description(f"DDPM Loss: {loss.item():.6f}, KL: {loss_kl.item():.6f}")
        bar.update(N)
        
        if batch % 300 == 0:
            save_model(model)
            tqdm.write("Saved model!")

        if epoch < 30:
            weight_kl = 0.02



