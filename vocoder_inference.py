import argparse
import glob
import os

import torch
import torchaudio
from  torchaudio.functional import resample as resample

from tqdm import tqdm

from model import DiffusionVocoder

parser = argparse.ArgumentParser(description="inference")

parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-i', '--inputs', default='./inputs')
parser.add_argument('-s', '--steps', default=10, type=int)
parser.add_argument('-ig', '--input-gain', default=1.0, type=float)
parser.add_argument('-g', '--gain', default=1.0, type=float)
parser.add_argument('-eta', '--eta', default=0, type=float)

args = parser.parse_args()

device = torch.device(args.device)

model = DiffusionVocoder().to(device)

if os.path.exists('./vocoder.pt'):
    print("Loading Model...")
    model.load_state_dict(torch.load('./vocoder.pt', map_location=device))

if not os.path.exists("./outputs/"):
    os.mkdir("outputs")

paths = glob.glob(os.path.join(args.inputs, "*.wav"))
for i, path in enumerate(paths):
    wf, sr = torchaudio.load(path)
    print(f"inferencing {path}")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.fp16):
            wf = wf.to(device)
            wf = resample(wf, sr, 22050) * args.input_gain
            length = wf.shape[1] + (256 - wf.shape[1] % 256)
            condition = model.to_spectrogram(wf)
            condition = model.spectrogram_encoder(condition)
            wf = model.vocoder.sample(x_shape=(1, length),
                    condition=condition,
                    show_progress=True,
                    num_steps=args.steps,
                    eta = args.eta
                    )
            wf = resample(wf, 22050, sr) * args.gain

    wf = wf.to('cpu').detach()
    torchaudio.save(src=wf, sample_rate=sr, filepath=f"./outputs/out_{i}.wav")
