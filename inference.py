import argparse
import glob
import os

import torch
import torchaudio
from  torchaudio.functional import resample as resample

from tqdm import tqdm

from model import DiffusionVC, Condition

parser = argparse.ArgumentParser(description="inference")

parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-i', '--inputs', default='./inputs')
parser.add_argument('-t', '--target-speaker', default='./speaker.wav')
parser.add_argument('-s', '--steps', default=10, type=int)
parser.add_argument('--encode-speaker-cpu', default=True, type=bool)

args = parser.parse_args()

device = torch.device(args.device)

model = DiffusionVC().to(device)

if os.path.exists('./model.pt'):
    print("Loading Model...")
    model.load_state_dict(torch.load('./model.pt', map_location=device))

target_wav, sr = torchaudio.load(args.target_speaker)
target_wav = resample(target_wav, sr, 22050)
if args.encode_speaker_cpu:
    target_wav = target_wav.to(torch.device('cpu'))
else:
    target_wav = target_wav.to(device)

print("Encoding target speaker...")
if args.encode_speaker_cpu:
    model.speaker_encoder = model.speaker_encoder.to(torch.device('cpu'))
target_speaker = model.speaker_encoder(target_wav).to(device)

if not os.path.exists("./outputs/"):
    os.mkdir("outputs")

paths = glob.glob(os.path.join(args.inputs, "*.wav"))
for i, path in enumerate(paths):
    wf, sr = torchaudio.load(path)
    print(f"converting {path}")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.fp16):
            wf = wf.to(device)
            wf = resample(wf, sr, 22050)
            content = model.content_encoder(wf)
            condition = Condition(content, target_speaker)
            length = wf.shape[1] + (256 - wf.shape[1] % 256)
            wf = model.generator.sample(x_shape=(1, length), condition=condition, show_progress=True, num_steps=args.steps)
            wf = resample(wf, 22050, sr)

    wf = wf.to('cpu').detach()
    torchaudio.save(src=wf, sample_rate=sr, filepath=f"./outputs/out_{i}.wav")
