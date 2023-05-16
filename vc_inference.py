import argparse
import glob
import os

import torch
import torchaudio
from  torchaudio.functional import resample as resample

from tqdm import tqdm

from model import DiffusionVocoder, DiffusionVC

import matplotlib.pyplot as plt
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="inference")

parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-i', '--inputs', default='./inputs')
parser.add_argument('-vs', '--vocoder-steps', default=20, type=int)
parser.add_argument('-cs', '--convertor-steps', default=20, type=int)
parser.add_argument('-t', '--target-path', default='./target.wav')
parser.add_argument('-ig', '--input-gain', default=1.0, type=float)
parser.add_argument('-g', '--gain', default=1.0, type=float)
parser.add_argument('-eta', '--eta', default=0, type=float)
parser.add_argument('-ps', '--pitch-shift', default=0, type=int)

args = parser.parse_args()

device = torch.device(args.device)

vocoder = DiffusionVocoder().to(device)
vc = DiffusionVC().to(device)

ps = torchaudio.transforms.PitchShift(22050, args.pitch_shift).to(device) if args.pitch_shift != 0 else torch.nn.Identity()

if os.path.exists('./vocoder.pt'):
    print("Loading Vocoder...")
    vocoder.load_state_dict(torch.load('./vocoder.pt', map_location=device))

if os.path.exists('./convertor.pt'):
    print("Loading Convertor...")
    vc.load_state_dict(torch.load('./convertor.pt', map_location=device))

if not os.path.exists("./outputs/"):
    os.mkdir("outputs")

print("Encoding target speaker...")
wf, sr = torchaudio.load(args.target_path)
wf = resample(wf, sr, 22050)
wf = wf.to(device)
spec = vc.to_spectrogram(wf)
target_speaker, _ = vc.speaker_encoder(spec)

paths = glob.glob(os.path.join(args.inputs, "*.wav"))
for i, path in enumerate(paths):
    wf, sr = torchaudio.load(path)
    print(f"converting {path}")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.fp16):

            wf = wf.to(device)
            wf = ps(wf)
            wf = resample(wf, sr, 22050) * args.input_gain
            length = wf.shape[1] + (256 - wf.shape[1] % 256)

            spec = vocoder.to_spectrogram(wf)
            content = vc.content_encoder(spec)
            spec = vc.generator.sample(
                    x_shape=(1, 129, spec.shape[2]),
                    condition=(content, target_speaker),
                    show_progress=True,
                    num_steps=args.convertor_steps,
                    eta=args.eta
                    )

            plt.imshow(spec[0].cpu() + 1e-4)
            plt.savefig(f"outputs/out_{i}_spectrogram.png", dpi=400)
            plt.imshow(torch.log10(F.interpolate(content.unsqueeze(1).cpu().abs(), (128, content.shape[2]))[0, 0] + 1e-4))
            plt.savefig(f"outputs/out_{i}_content.png", dpi=400)

            spec = vocoder.spectrogram_encoder(spec)
            wf = vocoder.vocoder.sample(
                    x_shape=(1, length),
                    condition=spec,
                    show_progress=True,
                    num_steps=args.vocoder_steps,
                    eta=args.eta
                    )
            wf = resample(wf, 22050, sr) * args.gain

    wf = wf.to('cpu').detach()
    torchaudio.save(src=wf, sample_rate=sr, filepath=f"./outputs/out_{i}.wav")
