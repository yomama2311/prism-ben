"""Idle-GPU latency micro-benchmark for the rebuttal's prior-compute column.
Times (CUDA-synced, 40 reps after 8 warmups), batch = 50 envs:
  (a) PriorHead forward           (the PRISM prior)
  (b) DP 16-sample DDIM-10 draw   (the hybrid prior)
"""
import time, torch, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from prior_head import PriorHead
from dp_baseline.model import DPModel
from dp_baseline.scheduler import DDPMScheduler

DEV = "cuda"
def timeit(fn, n=40, warmup=8):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

for task in ["pusht", "cube"]:
    ck = torch.load(f"prior_head_{task}.pt", map_location=DEV, weights_only=False)
    hp = ck["config"]
    head = PriorHead(z_dim=hp["z_dim"], H=hp["H"], A_block=hp["A_block"],
                     A_raw=hp["A_raw"], hidden=hp["hidden"],
                     sigma_floor=hp["sigma_floor"]).to(DEV)
    head.load_state_dict(ck["state_dict"]); head.eval()
    z = torch.randn(50, hp["z_dim"], device=DEV)
    with torch.no_grad():
        ms_head = timeit(lambda: head(z, z))

    dpc = torch.load(f"dp_baseline_{task}.pt", map_location=DEV, weights_only=False)
    dp = DPModel(action_dim=int(dpc["action_dim"])).to(DEV)
    dp.load_state_dict(dpc["ema_model" if "ema_model" in dpc else "model"]); dp.eval()
    sched = DDPMScheduler()
    obs = torch.randn(50, 3, 224, 224, device=DEV)
    def dp_draw():
        with torch.no_grad():
            cond = dp.encode_obs(obs, obs)
            cond16 = cond.repeat_interleave(16, dim=0)
            sched.sample(dp, cond16, sample_shape=(50*16, int(dpc["action_horizon"]),
                         int(dpc["action_dim"])), num_inference_steps=10, device=DEV)
    ms_dp = timeit(dp_draw, n=20, warmup=4)
    print(f"{task}: PriorHead(50 envs) = {ms_head:.3f} ms   "
          f"DP 16-sample draw(50 envs) = {ms_dp:.1f} ms   ratio = {ms_dp/ms_head:.0f}x")
