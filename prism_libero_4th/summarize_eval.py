"""Compare completed LIBERO evaluation suites; fails visibly if no valid results."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    results = []
    for p in sorted(Path(args.input_dir).glob('*.json')):
        r = json.loads(p.read_text())
        if 'episodes' not in r:
            continue
        episodes = r['episodes']
        n = len(episodes)
        if not n:
            print('[WARN] empty evaluation', p)
            continue
        wins = sum(bool(e['success']) for e in episodes)
        mses = [e['mean_prediction_mse'] for e in episodes if e.get('mean_prediction_mse') is not None]
        gaps = [e['mean_abs_prediction_gap'] for e in episodes if e.get('mean_abs_prediction_gap') is not None]
        row = dict(model=p.stem, trials=n, successes=wins, success_rate=wins/n,
                   mean_prediction_mse=(sum(mses)/len(mses)) if mses else None,
                   mean_abs_cost_gap=(sum(gaps)/len(gaps)) if gaps else None,
                   mean_steps=sum(e['steps'] for e in episodes)/n)
        results.append(row)
    if not results:
        raise SystemExit('No completed evaluation JSON found')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader(); w.writerows(results)
    print('=== LIBERO EVALUATION COMPARISON ===')
    for row in results:
        print(f"{row['model']:<25} success={row['successes']}/{row['trials']} "
              f"({row['success_rate']:.1%}) pred_mse={row['mean_prediction_mse']} "
              f"mean_abs_cost_gap={row['mean_abs_cost_gap']}")
    print('[SAVED]', out)


if __name__ == '__main__':
    main()
