import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from tqdm import tqdm

from model import DiffusionVC, Condition
from dataset import WaveFileDirectory
from lion_pytorch import Lion


parser = argparse.ArgumentParser(description="run training")

parser.add_argument('dataset')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-b', '--batch-size', default=4, type=int)
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('-len', '--length', default=32768, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-gacc', '--gradient-accumulation', default=1, type=int)

args = parser.parse_args()

device = torch.device(args.device)

def load_or_init_model(device=torch.device('cpu')):
    model = DiffusionVC().to(device)
    if os.path.exists('./model.pt'):
        model.load_state_dict(torch.load('./model.pt', map_location=device))
    return model

def save_model(model):
    torch.save(model.state_dict(), './model.pt')

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

model = load_or_init_model(device=device)
Ec = model.content_encoder
Es = model.speaker_encoder
G = model.generator
optimizer = Lion(model.parameters(), lr=1e-4)

ds = WaveFileDirectory(
        [args.dataset],
        length=args.length,
        max_files=args.max_data
        )

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size*2, shuffle=True)


for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))

    for batch, wave in enumerate(dl):
        N = wave.shape[0]
        wave = wave.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            speaker = Es(wave)
            content = Ec(wave)
            condition = Condition(content, speaker)
            loss = G.calculate_loss(wave, condition)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        bar.set_description(f"Loss: {loss.item():.4f}")
        bar.update(N)
        
        if batch % 100 == 0:
            save_model(model)

