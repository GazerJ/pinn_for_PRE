"""Plot archived zm data, or recompute it from the paper checkpoint."""
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


def predict(model, indices, mapping):
    s = torch.arange(0, 1, .001)
    result = []
    with torch.inference_mode():
        for i, j in indices:
            a = (i-112)*.125
            nu = .025*j-1
            v = nu if mapping == 'original' else (nu-1)/2
            X = torch.stack((s, torch.ones(len(s))*a, torch.ones(len(s))*v,
                             torch.ones(len(s))*4, torch.ones(len(s))*(np.pi/2)), dim=1)
            tz = np.cos(model(X).numpy()[:, 2])
            # Preserve the original float32 prefix-sum convention for verification.
            z = np.array([np.sum(tz[:q])*.001 for q in range(len(tz))])
            result.append(z.mean())
    return np.array(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', choices=['original', 'corrected'], default='original')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'model/B.m')
    parser.add_argument('--recompute', action='store_true', help='Recompute all 129x61 entries; takes minutes on CPU.')
    parser.add_argument('--verify', action='store_true', help='Compare 24 checkpoint predictions with archived zm data.')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Z = np.loadtxt(ROOT / 'data' / args.mapping / 'zm.csv')
    if Z.shape != (129, 61) or not np.isfinite(Z).all():
        raise ValueError('Expected finite 129x61 zm data')
    metadata = {'mapping': args.mapping, 'source': 'archived data',
                'colormap': COLORS, 'grid_shape': list(Z.shape)}
    if args.verify or args.recompute:
        torch.set_num_threads(args.threads)
        model = HardConstraintPINN()
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        metadata['checkpoint'] = str(args.checkpoint)
        metadata['checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if args.verify:
            ij = [(i, j) for i in [0,32,64,96,112,128] for j in [0,20,40,60]]
            pred = predict(model, ij, args.mapping)
            ref = np.array([Z[i,j] for i,j in ij])
            error = np.abs(pred-ref)
            metadata['verification'] = {'points':len(ij), 'max_abs_error':float(error.max()),
                                        'rmse':float(np.sqrt(np.mean(error**2)))}
            print(json.dumps(metadata['verification']), flush=True)
            if error.max() > 1e-6:
                raise RuntimeError('Checkpoint does not reproduce the archived data within 1e-6')
        if args.recompute:
            for i in range(129):
                Z[i] = predict(model, [(i,j) for j in range(61)], args.mapping)
                if i % 16 == 0:
                    print(f'recomputed row {i+1}/129', flush=True)
            np.savetxt(args.output_dir / f'zm_{args.mapping}_recomputed.csv', Z)
            metadata['source'] = 'checkpoint recomputation'
    reference = np.loadtxt(ROOT / 'data/original/zm.csv')
    reference_display = ndimage.convolve(reference, np.ones((4,4))/16).T[:, :-128//8]
    vmin, vmax = float(reference_display.min()), float(reference_display.max())
    display = ndimage.convolve(Z, np.ones((4,4))/16)[:113, :]
    alpha = 10**((np.arange(113)-112)*.125/2)
    nu = np.arange(61)*.025-1
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11})
    fig, ax = plt.subplots(figsize=(8.2,5.8), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list('paper_phase', COLORS, N=256)
    im = ax.pcolormesh(alpha, nu, display.T, shading='nearest', cmap=cmap,
                       vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xscale('log')
    ax.set_xlim(1e-7, 1)
    ax.set_ylim(-1, .5)
    ax.set_yticks([-1,-.5,0,.5])
    ax.set(xlabel=r'Bending stiffness / electric field ratio $\alpha_0$',
           ylabel=r"Poisson's ratio $\nu$", title=f'Paper checkpoint B.m | {args.mapping} mapping')
    fig.colorbar(im, ax=ax, label=r'Mean depth $\langle z\rangle/L$', ticks=np.arange(-.35,-.049,.05))
    for ext in ['png','pdf']:
        fig.savefig(args.output_dir / f'phase_{args.mapping}.{ext}', dpi=200)
    plt.close(fig)
    metadata['color_limits'] = [vmin, vmax]
    metadata['smoothing'] = '4x4 uniform convolution, scipy default reflect boundary'
    metadata['display_rows'] = [0,112]
    (args.output_dir / f'phase_{args.mapping}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Saved phase_{args.mapping}.png/pdf/json in {args.output_dir}')


if __name__ == '__main__':
    main()
