"""End-to-end smoke tests on synthetic data (no LIBERO, no GPU needed).
Run from the repo root:  python prism_libero/tests/test_pipeline.py
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parent
sys.path[:0] = [str(PKG), str(REPO), str(HERE)]

from chunk_prior_head import ChunkedPriorHead, masked_beta_nll, load_head  # noqa: E402
from latent_mppi import LatentPoGMPPI, pog_fusion  # noqa: E402
from stub_jepa import StubJEPA  # noqa: E402


def make_fake_libero(dirpath: Path, n_tasks=2, n_demos=6, seed=0):
    rng = np.random.default_rng(seed)
    dirpath.mkdir(parents=True, exist_ok=True)
    for t in range(n_tasks):
        with h5py.File(dirpath / f"task{t}_pick_up_thing_demo.hdf5", "w") as f:
            g = f.create_group("data")
            for d in range(n_demos):
                T = int(rng.integers(70, 110))
                prog = np.linspace(0, 1, T)[:, None]
                img = np.zeros((T, 128, 128, 3), np.uint8)
                img[..., t % 3] = (40 + 180 * prog[:, :, None]).astype(np.uint8)
                a = np.concatenate([np.sin(2 * np.pi * (prog + 0.3 * t)) * np.ones((1, 6)) * 0.5,
                                    np.where(prog > 0.5, 1.0, -1.0)], 1) + 0.02 * rng.standard_normal((T, 7))
                a[:, -1] = np.where(prog[:, 0] > 0.5, 1.0, -1.0)
                dm = g.create_group(f"demo_{d}")
                dm.create_dataset("actions", data=a.astype(np.float64))
                ob = dm.create_group("obs")
                ob.create_dataset("agentview_rgb", data=img)
                ob.create_dataset("ee_pos", data=rng.standard_normal((T, 3)))
                ob.create_dataset("ee_ori", data=rng.standard_normal((T, 3)))
                ob.create_dataset("gripper_states", data=rng.standard_normal((T, 2)))


def test_head_shapes_and_loss():
    B, D, Ab, Ar, C, K = 8, 16, 5, 7, 5, 4
    h = ChunkedPriorHead(D, Ab, Ar, chunk_steps=C, max_chunks=K, hidden=64)
    zt, zg = torch.randn(B, D), torch.randn(B, D)
    mu, sg = h(zt, zg, n_chunks=3)
    assert mu.shape == (B, 3 * C, Ab, Ar) and sg.shape == mu.shape and (sg >= 0.05).all()
    an = h.predict_anchors(zt, zg, K)
    assert an.shape == (B, K + 1, D) and torch.allclose(an[:, 0], zt)
    assert torch.allclose(an[:, 1], zt + 0.2 * (zg - zt), atol=1e-2)  # init = 20% step towards goal
    o = h.forward_train(torch.randn(B, K + 1, D), zg)
    assert o["mu"].shape == (B, K, C, Ab, Ar) and o["anc_tf"].shape == (B, K, D)
    tgt = torch.randn(B, K, C, Ab, Ar)
    mask = torch.zeros(B, K, C, Ab, 1); mask[:, 0] = 1
    l_masked = masked_beta_nll(o["mu"], o["sigma"], tgt, mask)
    tgt2 = tgt.clone(); tgt2[:, 1:] += 100.0  # masked region must not matter
    assert torch.allclose(l_masked, masked_beta_nll(o["mu"], o["sigma"], tgt2, mask))
    l_masked.backward()
    print("ok head shapes / masking")


def test_planner_pog():
    jepa = StubJEPA(d=16, act_dim=35)
    pl = LatentPoGMPPI(jepa, horizon=5, action_dim=35, num_samples=64, n_steps=3, topk=8)
    z0, zt = torch.randn(16), torch.randn(16)
    mu_p = torch.full((5, 35), 0.7)
    m, s = pog_fusion(torch.zeros(5, 35), torch.ones(5, 35), mu_p, torch.full((5, 35), 0.1))
    assert (m > 0.69).all() and (s < 0.11).all()  # confident prior dominates
    out = pl.plan(z0, zt, mu_p, torch.full((5, 35), 0.1), mode="pog")
    assert out.shape == (5, 35) and torch.isfinite(out).all()
    assert pl.plan(z0, zt, None, None, mode="none").shape == (5, 35)
    print("ok planner")


def run(cmd, env):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(r.stdout[-1800:])
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def test_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        env = dict(os.environ, STABLEWM_HOME=str(td / "swm"), PYTHONPATH=f"{PKG}:{REPO}:{HERE}")
        make_fake_libero(td / "libero_raw")
        h5 = td / "swm" / "libero" / "fake.h5"
        run([sys.executable, str(PKG / "convert_libero_to_swm.py"), "--libero-dir", str(td / "libero_raw"),
             "--out", str(h5)], env)
        with h5py.File(h5) as f:
            assert f["pixels"].shape[1:] == (128, 128, 3) and f["action"].shape[1] == 7
            last = f["ep_offset"][0] + f["ep_len"][0] - 1
            assert np.isnan(f["action"][last]).all() and not np.isnan(f["action"][0]).any()
        # stub JEPA checkpoint
        ck = td / "stub_object.ckpt"
        torch.manual_seed(0); torch.save(StubJEPA(d=16, act_dim=35), ck)
        run([sys.executable, str(PKG / "cache_features_libero.py"), "--h5", str(h5), "--jepa-ckpt", str(ck),
             "--name", "fake", "--img-size", "64", "--batch", "64"], env)
        cache = td / "swm" / "jepa_cache" / "fake"
        assert (cache / "features.npy").exists() and np.load(cache / "task_goals.npy").shape == (2, 16)
        out = td / "head.pt"
        so = run([sys.executable, str(PKG / "train_chunk_prior_head.py"), "--h5", str(h5), "--name", "fake",
                  "--epochs", "40", "--batch", "128", "--hidden", "128", "--warmup", "20",
                  "--out", str(out), "--force-save", "--val-stride", "1"], env)
        head, ck2 = load_head(str(out))
        assert isinstance(head, ChunkedPriorHead)
        mu, sg = head(torch.randn(3, 16), torch.randn(3, 16), n_chunks=2)
        assert mu.shape == (3, 10, 5, 7)
        out_flat = td / "flat.pt"
        run([sys.executable, str(PKG / "train_chunk_prior_head.py"), "--h5", str(h5), "--name", "fake",
             "--arch", "flat", "--epochs", "40", "--batch", "256", "--hidden", "128", "--warmup", "20",
             "--out", str(out_flat), "--force-save"], env)
        head_f, _ = load_head(str(out_flat))
        assert head_f(torch.randn(2, 16), torch.randn(2, 16))[0].shape == (2, 20, 5, 7)
        print("ok end-to-end")


if __name__ == "__main__":
    test_head_shapes_and_loss()
    test_planner_pog()
    test_pipeline_end_to_end()
    print("ALL TESTS PASSED")
