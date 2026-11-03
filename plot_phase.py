import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy import ndimage
import torch

from model import HardConstraintPINN

ROOT = Path(__file__).resolve().parent
COLORS = ['#9C8147', '#57479C', '#47629C', '#9C4762']
VMIN = -0.3836963102221489
VMAX = -0.042613446712493896


def predict(model, indices):
    s = torch.arange(0, 1, .001)
    result = []
    with torch.inference_mode():
        for i, j in indices:
            a = (i-112)*.125
            nu = .025*j-1
            v = (nu-1)/2
            X = torch.stack((s, torch.ones(len(s))*a, torch.ones(len(s))*v,
                             torch.ones(len(s))*4, torch.ones(len(s))*(np.pi/2)), dim=1)
            tz = np.cos(model(X).numpy()[:, 2])
            z = np.array([np.sum(tz[:q])*.001 for q in range(len(tz))])
            result.append(z.mean())
    return np.array(result)


def main():
    parser = argparse.ArgumentParser(description='Plot the electrospinning mean-depth phase diagram.')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/zm.csv')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'model/B.m')
    parser.add_argument('--recompute', action='store_true', help='Compute the 129x61 grid from the checkpoint.')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'colormap': COLORS, 'color_limits': [VMIN, VMAX],
                'network_input_v': '(nu-1)/2', 'k': 4, 'initial_tangent_theta': float(np.pi/2)}
    if args.recompute:
        torch.set_num_threads(args.threads)
        model = HardConstraintPINN()
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        Z = np.empty((129, 61))
        for i in range(129):
            Z[i] = predict(model, [(i,j) for j in range(61)])
            if i % 16 == 0:
                print(f'Computed row {i+1}/129', flush=True)
        if not np.isfinite(Z).all():
            raise ValueError('Non-finite mean-depth data')
        target = args.output_dir / 'zm.csv'
        if target.resolve() == (ROOT / 'data/zm.csv').resolve():
            parser.error('Choose an output directory other than data/')
        np.savetxt(target, Z)
        metadata['source'] = 'checkpoint'
        metadata['checkpoint'] = args.checkpoint.name
        metadata['checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    else:
        Z = np.loadtxt(args.data)
        metadata['source'] = 'data'
        metadata['data_file'] = args.data.name
    if Z.shape != (129, 61) or not np.isfinite(Z).all():
        raise ValueError('Expected finite 129x61 zm data')
    display = ndimage.convolve(Z, np.ones((4,4))/16)[:113, :]
    alpha = 10**((np.arange(113)-112)*.125/2)
    nu = np.arange(61)*.025-1
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11})
    fig, ax = plt.subplots(figsize=(8.2,5.8), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list('phase', COLORS, N=256)
    im = ax.pcolormesh(alpha, nu, display.T, shading='nearest', cmap=cmap,
                       vmin=VMIN, vmax=VMAX, rasterized=True)
    ax.set_xscale('log')
    ax.set_xlim(1e-7, 1)
    ax.set_ylim(-1, .5)
    ax.set_yticks([-1,-.5,0,.5])
    ax.set(xlabel=r'Bending stiffness / electric field ratio $\alpha_0$',
           ylabel=r"Poisson's ratio $\nu$", title=r'Mean-depth phase diagram ($k=4$)')
    fig.colorbar(im, ax=ax, label=r'Mean depth $\langle z\rangle/L$', ticks=np.arange(-.35,-.049,.05))
    for ext in ['png','pdf']:
        fig.savefig(args.output_dir / f'phase.{ext}', dpi=200)
    plt.close(fig)
    metadata['grid_shape'] = list(Z.shape)
    metadata['smoothing'] = '4x4 uniform convolution, reflect boundary'
    metadata['display_rows'] = [0,112]
    (args.output_dir / 'phase.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Saved phase.png/pdf/json in {args.output_dir}')


if __name__ == '__main__':
    main()
