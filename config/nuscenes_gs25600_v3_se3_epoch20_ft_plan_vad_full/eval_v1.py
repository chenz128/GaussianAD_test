"""Run the repository evaluator with an isolated no-cumsum v1 metric."""

import argparse
import os
import sys

import torch

# The evaluator lives two levels below the repository root.  Put that exact
# checkout first so an environment-level GaussianAD install cannot shadow it.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import test as gaussianad_test
from metric_v1_no_cumsum import compute_planner_metric_v1


gaussianad_test.compute_planner_metric_stp3 = compute_planner_metric_v1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--resume-from', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis-occ', action='store_true', default=False)
    parser.add_argument('--log-name', default='')
    args = parser.parse_args()
    args.gpus = torch.cuda.device_count()
    if args.gpus > 1:
        torch.multiprocessing.spawn(
            gaussianad_test.main, args=(args,), nprocs=args.gpus)
    else:
        gaussianad_test.main(0, args)
