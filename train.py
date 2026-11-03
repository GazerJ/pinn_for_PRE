import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from model import HardConstraintPINN
from physics import compute_loss

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Train the electrospinning PINN with spontaneous-curvature-normalized residuals.')
    parser.add_argument('--epochs', type=int, default=600000)
    parser.add_argument('--points', type=int, default=400)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--init', type=Path, help='Optional initial state_dict; no optimizer state is loaded.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/training/B-trained.m')
    parser.add_argument('--log-every', type=int, default=1000)
    args = parser.parse_args()
    if min(args.epochs, args.points, args.threads, args.log_every) < 1 or args.points < 2:
        parser.error('epochs, threads and log-every must be positive; points must be >= 2')
    if args.lr is not None and args.lr <= 0:
        parser.error('lr must be positive')
    protected = ROOT / 'model/B.m'
    if args.output.resolve() == protected.resolve():
        parser.error('model/B.m is protected; choose a different output')
    if args.output.exists():
        parser.error('output already exists; choose a new output path')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = HardConstraintPINN()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location='cpu', weights_only=True))
    lr = args.lr if args.lr is not None else 1e-10 * np.sqrt(args.points)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    best_epoch = None
    best_params = None
    for epoch in range(args.epochs):
        points = torch.rand(args.points, 5)
        points[:, 0] = torch.linspace(0, 1, args.points)
        points[:, 1] *= -8
        points[:, 2] *= 1
        points[:, 3] *= 4
        points[:, 4] = points[:, 4] * np.pi / 8 + np.pi / 2
        loss, score, _ = compute_loss(model, points)
        if not torch.isfinite(loss) or not torch.isfinite(score):
            raise RuntimeError(f'Non-finite loss or residual score at epoch {epoch}')
        if score.item() < best:
            best = score.item()
            best_epoch = epoch
            best_params = copy.deepcopy(model.state_dict())
            torch.save(best_params, args.output)
        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(f'epoch={epoch+1} loss={loss.item():.6e} score={score.item():.6e} best={best:.6e}', flush=True)
        if best < 5e-5:
            break
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    metadata = {'epochs_run': epoch+1, 'best_epoch_zero_based': best_epoch,
                'best_residual_score': best, 'lr': lr, 'points': args.points,
                'seed': args.seed, 'init': str(args.init) if args.init else None,
                'output': str(args.output), 'torch': torch.__version__}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
