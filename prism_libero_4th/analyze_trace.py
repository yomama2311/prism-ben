"""Analyze one evaluation JSONL trace into CSV, JSON, and an optional PNG plot.

Example:
  python prism_libero/analyze_trace.py \
      --trace /workspace/stable-wm/evaluations/libero_spatial/RUN/chunked_pog_subgoal_diagnostics/task00_trial000.jsonl --plot
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def describe(rows):
    action = [x for x in rows if x.get('type') == 'action']
    block = [x for x in rows if x.get('type') == 'block']
    complete = [x for x in block if x.get('complete_5_step_block')]
    dist = [(x['step'], x['eef_object_distance']) for x in action
            if x.get('eef_object_distance') is not None]
    z = [x['object_z'] for x in action if x.get('object_z') is not None]
    gap = [abs(x['prediction_gap']) for x in complete if x.get('prediction_gap') is not None]
    wrong = [x for x in complete if x['predicted_delta'] < 0 < x['actual_delta']]
    final = [x for x in rows if x.get('type') == 'summary']
    result = dict(n_actions=len(action), n_full_blocks=len(complete),
                  n_partial_blocks=len(block) - len(complete),
                  n_predicted_goal_improvement_but_actual_worsening=len(wrong),
                  fraction_false_improvement=len(wrong) / len(complete) if complete else None,
                  mean_abs_goal_prediction_gap=float(np.mean(gap)) if gap else None,
                  minimum_eef_object_distance=min(dist, key=lambda t: t[1]) if dist else None,
                  object_height_range=(float(min(z)), float(max(z))) if z else None,
                  final_success=bool(final[-1]['success']) if final else None,
                  gripper_sign_changes=sum(np.sign(a['gripper_action']) != np.sign(b['gripper_action'])
                                           for a,b in zip(action, action[1:])),
                  mean_gripper_action=float(np.mean([x['gripper_action'] for x in action])) if action else None)
    return result, action, block


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace', required=True)
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()
    trace = Path(args.trace)
    rows = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
    stats, actions, blocks = describe(rows)
    csv_write(trace.with_suffix('.actions.csv'), actions)
    csv_write(trace.with_suffix('.blocks.csv'), blocks)
    trace.with_suffix('.diagnostics.json').write_text(json.dumps(stats, indent=2) + '\n')
    print('[TRACE] '+json.dumps(stats, ensure_ascii=False, indent=2))
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        if actions:
            xx = [a['step'] for a in actions]
            yy = [np.nan if a.get('eef_object_distance') is None else a['eef_object_distance'] for a in actions]
            axs[0].plot(xx, yy)
            axs[0].set_ylabel('EEF-object distance (m)')
            axs[2].plot(xx, [a['gripper_action'] for a in actions])
            axs[2].set_ylabel('Gripper action')
            axs[2].set_xlabel('Simulator steps')
        full = [b for b in blocks if b.get('complete_5_step_block')]
        if full:
            xx = [b['step'] for b in full]
            axs[1].plot(xx, [b['current_goal_mse'] for b in full], label='before')
            axs[1].plot(xx, [b['predicted_goal_mse'] for b in full], label='predicted')
            axs[1].plot(xx, [b['actual_goal_mse'] for b in full], label='actual')
            axs[1].legend()
            axs[1].set_ylabel('Final-goal latent MSE')
        for a in axs: a.grid(alpha=0.3)
        fig.tight_layout()
        target = trace.with_suffix('.diagnostics.png')
        fig.savefig(target, dpi=150)
        plt.close(fig)
        print('[PLOT]', target)


if __name__=='__main__':
    main()
