"""Check cached preprocessing, subset identity, RNG isolation and finite guards."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from torchvision import transforms
from data import get_loaders
from train import check_gradients


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    torch.set_num_threads(2)
    for name, batch in [('cifar10', 1000), ('mnist', 512)]:
        old, test, _ = get_loaders(name, batch, num_workers=0)
        cached, cached_test, _ = get_loaders(name, batch, num_workers=0, cache=True)
        ids = [0, 1, 13, 22]
        if name == 'cifar10':old.dataset.transform = transforms.Compose(old.dataset.transform.transforms[:-1])
        expected = torch.stack([old.dataset[i][0] for i in ids])
        assert torch.equal(expected, cached.dataset.images[ids]), name
        x, y = cached_test.dataset.__getitems__(ids)
        assert torch.equal(x, torch.stack([test.dataset[i][0] for i in ids]))
        if name == 'cifar10':
            torch.manual_seed(5);mask = torch.rand(len(ids)) < .5
            torch.manual_seed(5);x, _ = cached.dataset.__getitems__(ids)
            assert torch.equal(x, torch.where(mask[:, None, None, None], expected.flip(-1), expected))
        del old, test, cached, cached_test
    old, _, _ = get_loaders('mnist', 16, num_workers=0, train_subset=32)
    cached, _, _ = get_loaders('mnist', 16, num_workers=2, train_subset=32, cache=True)
    assert old.dataset.indices == cached.dataset.indices and cached.persistent_workers
    for _ in range(2):
        x, y = next(iter(cached));assert x.shape == (16, 1, 28, 28) and y.shape == (16,)
    model = torch.nn.Linear(2, 1)
    loss = model(torch.ones(1, 2)).sum();loss.backward();check_gradients(model, loss)
    model.weight.grad.fill_(float('nan'))
    try:check_gradients(model, loss)
    except FloatingPointError:pass
    else:raise AssertionError('nonfinite gradient was accepted')
    model.weight.grad.zero_()
    try:check_gradients(model,torch.tensor(float('inf')))
    except FloatingPointError:pass
    else:raise AssertionError('nonfinite loss was accepted')
    with tempfile.TemporaryDirectory(prefix='efficient_nn_checks_') as tmp:
        for interval in (0, 3):
            subprocess.run([sys.executable, 'code/train.py', '--dataset', 'mnist', '--model', 'lenet',
                '--batch-size', '16', '--subset', '32', '--iters', '5', '--policy', 'const',
                '--max-lr', '.01', '--seed', '4', '--workers', '0', '--device', 'cpu',
                '--cache-data', '--eval-every', str(interval), '--out', f'{tmp}/{interval}'],
                check=True, env={**os.environ, 'OMP_NUM_THREADS': '2'})
        states = [torch.load(f'{tmp}/{i}/model.pt', weights_only=True) for i in (0, 3)]
        assert all(torch.equal(states[0][k], states[1][k]) for k in states[0]), 'evaluation changed training'
        import json
        for interval, expected in [(0, [5]), (3, [3, 5])]:
            rows = json.loads(Path(f'{tmp}/{interval}/history.json').read_text())['history']
            assert [r['iter'] for r in rows] == expected
    print('cached transforms, batch reflection, subset, persistent workers, finite guard and evaluation RNG: PASS')


if __name__ == '__main__':main()
