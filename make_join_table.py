#!/usr/bin/env python3
# make_join_table.py
"""
Generate a join table JSON for async decentralized FL:
    { "<client_id>": <join_time_float>, ... }

Usage examples:

# Delay 50% clients with uniform times in [5.0, 50.0]
python make_join_table.py --num_clients 10 --delay_client_ratio 0.5 \
  --min_time 5.0 --max_time 50.0 --dist uniform --seed 42 \
  --out join_emnist_10c.json

# Delay exactly one client (id=7) at time 12.5
python make_join_table.py --num_clients 10 --delay_client_ratio 0.1 \
  --min_time 12.5 --max_time 12.5 --dist single --preset_client_id 7 \
  --out join_single.json
"""

import json
import os
import random
import numpy as np

from argparse import ArgumentParser

# import your utils
from utils.utils import get_client_delay_info


def parse_args():
    p = ArgumentParser(description="Generate async join table JSON for clients.")
    p.add_argument("--num_clients", type=int, required=True, help="Total number of clients.")
    p.add_argument("--delay_client_ratio", type=float, default=0.5,
                   help="Fraction of clients that join late (0~1).")
    p.add_argument("--min_time", type=float, default=0.0,
                   help="Earliest join time among delayed clients (inclusive).")
    p.add_argument("--max_time", type=float, required=True,
                   help="Latest join time among delayed clients (must be >= min_time).")
    p.add_argument("--dist", type=str, default="uniform",
                   choices=["single", "uniform", "even", "normal"],
                   help="Distribution of join times for delayed clients.")
    p.add_argument("--preset_client_id", type=int, default=-1,
                   help="Used when dist=single. If -1, pick a random client.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--out", type=str, required=True, help="Output JSON path.")
    return p.parse_args()


def main():
    args = parse_args()

    # seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # generate delayed join mapping in wall-clock mode
    delayed = get_client_delay_info(
        num_clients=args.num_clients,
        delay_client_ratio=args.delay_client_ratio,
        minimum_round=args.min_time,          # interpreted as min_time in time_mode
        total_rounds=args.max_time,           # interpreted as max_time in time_mode
        dist_type=args.dist,
        preset_client_id=args.preset_client_id,
        time_mode=True
    )

    # Write JSON (only delayed clients; non-delayed default to t=0.0 in coordinator)
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in delayed.items()}, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved join table with {len(delayed)} delayed clients to: {args.out}")


if __name__ == "__main__":
    main()