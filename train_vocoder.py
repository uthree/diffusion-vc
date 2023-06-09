import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from tqdm import tqdm

from model import DiffusionVocoder
from dataset import WaveFileDirectory


parser = argparse.ArgumentParser(description="run training")

parser.add_argument('dataset')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=10000, type=int)
parser.add_argument('-b', '--batch-size', default=8, type=int)
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('-len', '--length', default=32768, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-gacc', '--gradient-accumulation', default=4, type=int)

args = parser.parse_args()

device = torch.device(args.device)

def load_or_init_model(device=torch.device('cpu')):
    model = DiffusionVocoder().to(device)
    if os.path.exists('./vocoder.pt'):
        model.load_state_dict(torch.load('./vocoder.pt', map_location=device))
    return model

def save_model(model):
    torch.save(model.state_dict(), './vocoder.pt')

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
            spec = model.spectrogram_encoder(spec)
            loss = model.vocoder.calculate_loss(wave, spec)

        scaler.scale(loss).backward()
        if batch % grad_acc == 0:
            scaler.step(optimizer)
            scaler.update()
        
        bar.set_description(f"loss: {loss.item():.6f}")
        bar.update(N)
        
        if batch % 300 == 0:
            save_model(model)
            tqdm.write("Saved model!")

