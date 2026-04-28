#!/usr/bin/env python3
"""
Run order-parameter computation for ONE (model, temperature, run) tuple.

Usage:
  python run_single_condition.py tip4p2005 T-20 Run01
  python run_single_condition.py tip5p     T-10 Run01
  python run_single_condition.py swm4ndp   T-20 Run01
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from compute_order_params import (
    process_single_dcd,
    find_matching_pdb,
    TIP4P2005_DATA_DIR,
    TIP5P_DATA_DIR,
    SWM4NDP_DATA_DIR,
    OUTPUT_DIR,
)

_MODEL_DIRS = {
    'tip4p2005': TIP4P2005_DATA_DIR,
    'tip5p':     TIP5P_DATA_DIR,
    'swm4ndp':   SWM4NDP_DATA_DIR,
}


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    model, temp, run = sys.argv[1], sys.argv[2], sys.argv[3]

    if model not in _MODEL_DIRS:
        print(f"Error: invalid model '{model}'. Must be one of {list(_MODEL_DIRS)}")
        sys.exit(1)

    data_dir = _MODEL_DIRS[model]
    dcd_file = os.path.join(data_dir, f"dcd_{model}_{temp}_N1024_{run}_0.dcd")

    if not os.path.exists(dcd_file):
        print(f"Error: DCD file not found: {dcd_file}")
        sys.exit(1)

    pdb_file   = find_matching_pdb(dcd_file)
    output_dir = os.environ.get("ORDER_PARAM_OUT_DIR", OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Processing single condition:")
    print(f"  Model: {model}    Temperature: {temp}    Run: {run}")
    print(f"  DCD:   {dcd_file}")
    print(f"  Out:   {output_dir}")
    print(f"{'='*70}\n")

    if not process_single_dcd(dcd_file, pdb_file, output_dir, model):
        sys.exit(1)


if __name__ == '__main__':
    main()
