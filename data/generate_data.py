import argparse
from privateggm.data import *

parser = argparse.ArgumentParser(description="Generate synthetic data.")

parser.add_argument("--n", type=int, default=1000, help="size of synthetic data")
parser.add_argument("--d", type=int, default=200, help="dimension of synthetic data")

args = parser.parse_args()

outdir = "data/"

generate_well_conditioned_gaussian(outdir, args.n, args.d, rescale, cond_max=10.0)