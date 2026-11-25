"""Success rate per family with a 95% Wilson interval, from an episodes.jsonl file.

  python summarize.py runs/episodes.jsonl
  python summarize.py results/episodes/*.jsonl --csv results/summary.csv
"""
import argparse
import csv
import json
import math

ORDER = ('Standard', 'P2', 'F-obj', 'F-dest', 'F-both')


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--csv')
    args = ap.parse_args()
    rows = [json.loads(line) for f in args.files for line in open(f) if line.strip()]
    table = []
    for fam in sorted({r['family'] for r in rows}, key=lambda f: ORDER.index(f) if f in ORDER else 99):
        rr = [r for r in rows if r['family'] == fam]
        k, n = sum(bool(r['success']) for r in rr), len(rr)
        lo, hi = wilson(k, n)
        table.append(dict(family=fam, episodes=n, successes=k, success_rate_pct=round(100 * k / n, 1),
                          ci95_low_pct=round(100 * lo, 1), ci95_high_pct=round(100 * hi, 1)))
    for t in table:
        print(f"{t['family']:9s} {t['successes']:3d}/{t['episodes']:<3d} {t['success_rate_pct']:5.1f}%  "
              f"[{t['ci95_low_pct']:.1f}, {t['ci95_high_pct']:.1f}]")
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, list(table[0]))
            w.writeheader()
            w.writerows(table)


if __name__ == '__main__':
    main()
