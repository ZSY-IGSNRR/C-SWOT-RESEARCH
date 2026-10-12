"""Compatibility entry for the current protocol verifier."""
from pathlib import Path
import argparse
from verify_swot5_results import verify
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results-dir',type=Path,required=True);a=p.parse_args()
    raise SystemExit(0 if verify(a.results_dir)['passed'] else 1)
