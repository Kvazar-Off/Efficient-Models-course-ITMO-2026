"""Verify current code, all v2 artifacts, final checkpoints and document links."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import torch
from PIL import Image
from check_solution import main as check_logic
from data import get_loaders
from train import build_model, evaluate, save_json, seed_all

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true')
    a=p.parse_args();os.chdir(ROOT)
    for path in Path('code').glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    for path in Path('code').glob('*.sh'):subprocess.run(['bash','-n',str(path)],check=True)
    check_logic()
    torch.set_num_threads(2)
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    manifest=json.loads(Path('runs/v2/manifest.json').read_text())
    results=[]; missing=[]; loaders={}
    for job in manifest['jobs']:
        directory=Path('runs/v2')/job['name']
        artifact=directory/('range_test.json' if job['script']=='range_test.py' else 'history.json')
        if not artifact.exists():missing.append(job['name']);continue
        d=json.loads(artifact.read_text())
        if d['status']=='running':missing.append(job['name']);continue
        if job['script']=='range_test.py':
            assert len(d['lrs'])==len(d['losses'])==len(d['accs'])==d['steps']
            minimum=min(range(d['steps']),key=lambda i:d['smoothed'][i])
            assert minimum+1==d['min_iter'] and d['lrs'][minimum]==d['lr_at_min']
            continue
        cfg=d['config'];args=cfg['args'];budget=cfg['schedule']['total_iters']
        assert d['completed_iters']<=budget
        if d['status']=='failed':
            results.append({'run':job['name'],'status':'failed','error':d['error']});continue
        assert d['status']=='completed' and d['completed_iters']==budget,job['name']
        rr=d['history'];assert rr and rr[-1]['iter']==budget
        assert all(x['iter']<y['iter'] for x,y in zip(rr,rr[1:]))
        ip=json.loads((directory/'iter_log.json').read_text())
        assert max(r['test_acc'] for r in rr+ip)==d['best']['acc']
        lr=json.loads((directory/'lr_log.json').read_text())
        assert len(lr)==budget and [r['iter'] for r in lr]==list(range(1,budget+1))
        assert lr[-1]['lr']==rr[-1]['lr']
        for filename in ['train.py','data.py','resnet.py','lenet.py','policies.py']:
            digest=cfg['code_sha256'][filename]
            current=Path('code')/filename
            archived=Path('references/source_snapshots')/digest/filename
            source=current if hashlib.sha256(current.read_bytes()).hexdigest()==digest else archived
            assert source.exists() and hashlib.sha256(source.read_bytes()).hexdigest()==digest,filename
        state=torch.load(directory/'model.pt',map_location=dev,weights_only=True)
        assert all(torch.isfinite(t).all() for t in state.values()),job['name']
        key=(args['dataset'],args['normalization'])
        if key not in loaders:
            _,loaders[key],_=get_loaders(args['dataset'],args['batch_size'],num_workers=2,normalization=args['normalization'])
        seed_all(args['seed'])
        model=build_model(args['model'],1 if args['dataset']=='mnist' else 3,args['bn_momentum'],args['shortcut']).to(dev)
        if args.get('channels_last'):model.to(memory_format=torch.channels_last)
        model.load_state_dict(state)
        acc,loss=evaluate(model,loaders[key],dev,channels_last=bool(cfg.get('execution')) and args.get('channels_last',False))
        assert abs(acc-rr[-1]['test_acc'])<1e-9,(job['name'],acc,rr[-1]['test_acc'])
        assert math.isclose(loss,rr[-1]['test_loss'],rel_tol=1e-4,abs_tol=1e-5)
        note=None
        if args['policy'] not in ('onecycle','cosine') and cfg['schedule'].get('final_lr') is not None:
            note='This source version recorded an unused generic final_lr default; used LR is lr_log.json.'
            save_json(directory/'metadata_notes.json',{'note':note,'terminal_lr_used':lr[-1]['lr'],
                'source_sha256':cfg['code_sha256']['train.py']})
        results.append({'run':job['name'],'status':'completed','updates':budget,'test_size':10000,
                        'terminal_lr_used':lr[-1]['lr'],'metadata_note':note,
                        'final_acc':acc,'recomputed_loss':loss,'accuracy_delta':acc-rr[-1]['test_acc']})
        del state,model
    hp=Path('runs/v2/hessian_lr.json')
    if hp.exists():
        hd=json.loads(hp.read_text());alpha=hd['config']['alpha']
        for mode,rows in hd['records'].items():
            assert len(rows)==hd['config']['iters'] and rows[0]['iter']==1 and rows[-1]['iter']==len(rows)
            for key in ('parameters','all_blobs'):
                previous=.1
                for row in rows:
                    assert all(math.isfinite(row[k]) for k in (key,key+'_raw','lr'))
                    assert math.isclose(row[key],alpha*row[key+'_raw']+(1-alpha)*previous,rel_tol=1e-12)
                    previous=row[key]
    elif not a.partial:raise AssertionError('Hessian artifact missing')
    for path in Path('figures').rglob('*.png'):
        with Image.open(path) as im:im.verify()
    for filename in ['report2.md','paper_analysis.md','code/README.md']:
        if not Path(filename).exists():
            if not a.partial:raise AssertionError('missing document '+filename)
            continue
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',Path(filename).read_text()):
            if target.startswith(('http:','https:','#')):continue
            target=target.split('#')[0]
            exists=(Path(filename).parent/target).exists()
            if not exists and a.partial:continue
            assert exists,(filename,target)
    save_json('figures/report2/verification.json',{'checks':results,'pending':missing,
        'complete':not missing,'scope':'26 final checkpoints and 2 LR range tests in submission'})
    print(f'PASS: logic, syntax, images, links; {len(results)} checked runs; pending {len(missing)}')
    if missing and not a.partial:raise AssertionError('unfinished jobs: '+', '.join(missing))

if __name__=='__main__':main()
