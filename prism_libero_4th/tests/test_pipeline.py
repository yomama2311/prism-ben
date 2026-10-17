"""Synthetic smoke/regression checks; NO real LIBERO or GPU required.

Tests checkpoint saving, episode-disjoint conversion, causal MPPI history,
clamped actions, binary gripper and Prior Head compatibility. Real-world test
on LIBERO + actual LeWM is still required.
"""
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parent
sys.path[:0] = [str(PKG), str(REPO), str(HERE)]

try:
    from prior_head import PriorHead
except ImportError:
    from stub_prior_head import PriorHead
    fake = types.ModuleType('prior_head')
    fake.PriorHead = PriorHead
    sys.modules['prior_head'] = fake

from chunk_prior_head import ChunkedPriorHead, masked_beta_nll, load_head
from latent_mppi import LatentPoGMPPI, pog_fusion
from stub_jepa import StubJEPA


def make_fake_libero(dirpath, n_tasks=2, n_demos=6, seed=0):
    rng = np.random.default_rng(seed)
    dirpath.mkdir(parents=True, exist_ok=True)
    for task in range(n_tasks):
        with h5py.File(dirpath / f'task{task}_pick_up_thing_demo.hdf5', 'w') as f:
            grp = f.create_group('data')
            for demo in range(n_demos):
                T = int(rng.integers(105, 118))
                ratio = np.linspace(0, 1, T)
                frame = np.zeros((T, 128, 128, 3), np.uint8)
                frame[:, :, :, task % 3] = (40 + 180*ratio[:,None,None]).astype(np.uint8)
                a = np.asarray(rng.normal(0, 0.5, (T, 7)), np.float32)
                a[:, -1] = np.where(ratio > 0.5, 1., -1.)
                g = grp.create_group(f'demo_{demo}')
                g.create_dataset('actions', data=a)
                obs = g.create_group('obs')
                obs.create_dataset('agentview_rgb', data=frame)
                obs.create_dataset('ee_pos', data=rng.normal(size=(T, 3)))
                obs.create_dataset('ee_ori', data=rng.normal(size=(T, 3)))
                obs.create_dataset('gripper_states', data=rng.normal(size=(T, 2)))


def run(cmd, env):
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    if p.returncode:
        raise AssertionError(f'FAILED cmd={cmd}\nstdout={p.stdout[-4000:]}\nstderr={p.stderr[-4000:]}')
    print(p.stdout[-800:])
    return p.stdout


def test_head_shapes_and_loss():
    head = ChunkedPriorHead(16, 5, 7, chunk_steps=5, max_chunks=4, hidden=64)
    z0, zg = torch.randn(8,16), torch.randn(8,16)
    mu, sigma = head(z0,zg,3)
    assert mu.shape == sigma.shape == (8,15,5,7)
    assert (sigma>=.05).all()
    anc = head.predict_anchors(z0,zg,4)
    assert anc.shape == (8,5,16)
    t = torch.randn(8,4,5,5,7)
    out = head.forward_train(torch.randn(8,5,16), zg)
    mask = torch.zeros(8,4,5,5,1); mask[:,0]=1
    loss1 = masked_beta_nll(out['mu'],out['sigma'],t,mask)
    t[:,1:] += 100
    assert torch.allclose(loss1,masked_beta_nll(out['mu'],out['sigma'],t,mask))
    loss1.backward()
    print('[PASS] head and loss')


def test_planner_causal_action_contract():
    wm = StubJEPA(d=16, act_dim=35)
    p = LatentPoGMPPI(wm,horizon=2,action_dim=35,num_samples=8,n_steps=2,
                      topk=0, gripper_mode='binary')
    z = torch.randn(3,16)
    ah = torch.zeros(2,35)
    mu,sigma = torch.randn(2,35),torch.ones(2,35)*.2
    zeros=torch.zeros(7); ones=torch.ones(7)
    low, high = -ones,ones
    plan, meta = p.plan(z,ah,torch.randn(16),mu,sigma,zeros,ones,
                        zeros,ones,low,high,'pog')
    assert plan.shape == (2,5,7) and plan.abs().max() <= 1
    assert (plan[...,6].abs()==1).all(), 'planning and execution should share gripper sign'
    pred = p.predict_one(z,ah,plan[0],zeros,ones)
    assert pred.shape == (16,) and torch.isfinite(pred).all()
    assert np.isfinite(meta['selected_horizon_cost'])
    try:
        p.predict_one(z[:1],ah,plan[0],zeros,ones)
        raise AssertionError('incorrect history should fail')
    except ValueError:
        pass
    print('[PASS] causal MPPI and action equivalence')


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        source=root/'source'
        make_fake_libero(source)
        fakepath=root/'fake_lewm'
        torch.save(StubJEPA(d=16,act_dim=35),fakepath)
        env = dict(os.environ, STABLEWM_HOME=str(root/'swm'),
                   PYTHONPATH=f'{HERE}:{PKG}:{REPO}:'+os.environ.get('PYTHONPATH',''))
        train=root/'swm/train.h5'; val=root/'swm/val.h5'
        run([sys.executable,str(PKG/'convert_libero_to_swm.py'),
             '--libero-dir',str(source),'--out',str(train),'--val-frac','0.15',
             '--val-out',str(val)],env)
        with h5py.File(train) as tf,h5py.File(val) as vf:
            assert tf['action'].shape[-1]==7 and vf['action'].shape[-1]==7
            assert len(tf['ep_len']) + len(vf['ep_len']) == 12
            assert np.isnan(tf['action'][int(tf['ep_len'][0])-1]).all()
        run([sys.executable,str(PKG/'cache_features_libero.py'),
             '--h5',str(train),'--jepa-ckpt',str(fakepath),
             '--name','fake','--img-size','64','--batch','64'],env)
        prior = root/'prior'
        for arch in ('chunked','flat'):
            name = root/f'{arch}_best.pt'
            stdout=run([sys.executable,str(PKG/'train_chunk_prior_head.py'),
                        '--h5',str(train),'--name','fake','--arch',arch,
                        '--epochs','11','--checkpoint-every','10','--patience','0',
                        '--batch','32','--hidden','64','--warmup','2',
                        '--max-chunks','2','--chunk-steps','2',
                        '--val-stride','1', '--checkpoint-dir',str(prior/arch),
                        '--out',str(name)],env)
            assert '[BEST]' in stdout
            ck = prior/arch
            assert (ck/'best.pt').exists() and (ck/'epoch_010.pt').exists()
            assert (ck/'last.pt').exists()
            assert (ck/'history.jsonl').exists() and (ck/'summary.json').exists()
            h, config = load_head(str(name))
            assert config['config']['arch'] == arch
            assert h(torch.randn(3,16),torch.randn(3,16))[0].shape[-1]==7
        # Use a tiny test-only mock LIBERO to exercise evaluator from observation
        # through prediction, simulator step, JSONL, CSV, summary and success.
        output = root/'test_eval.json'
        mockenv = dict(env)
        mockenv['PYTHONPATH'] = f"{HERE/'fake_libero'}:{env['PYTHONPATH']}"
        stdout = run([sys.executable, str(PKG/'eval_libero_prism.py'),
                      '--suite','libero_spatial','--h5',str(train),
                      '--cache-name','fake','--jepa-ckpt',str(fakepath),
                      '--head',str(root/'chunked_best.pt'),
                      '--tasks','0','--episodes','1','--max-steps','20',
                      '--settle-steps','0','--num-samples','8','--n-steps','2',
                      '--img-size','64','--mode','pog','--goal-mode','subgoal',
                      '--out',str(output)],mockenv)
        assert '[RESULT]' in stdout
        result = json.loads(output.read_text())
        assert result['episodes'][0]['success'] is True
        assert result['episodes'][0]['steps']==11
        trace = Path(result['episodes'][0]['trace'])
        entries = [json.loads(x) for x in trace.read_text().splitlines()]
        assert len([x for x in entries if x['type']=='action']) == 11
        blocks = [x for x in entries if x['type']=='block']
        assert len(blocks)==3 and blocks[0]['actions_executed']==5
        assert blocks[-1]['actions_executed']==1
        assert all('actual_goal_mse' in b and 'prediction_mse' in b for b in blocks)
        assert output.with_suffix('.csv').exists()
        print('[PASS] fake LIBERO closed-loop eval + JSONL action/goal/prediction logs')
        print('[PASS] synthetic convert/cache/train/checkpoints')


if __name__=='__main__':
    torch.set_num_threads(2)
    test_head_shapes_and_loss()
    test_planner_causal_action_contract()
    test_end_to_end()
    print('ALL TESTS PASSED (synthetic only)')
