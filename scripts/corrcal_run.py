"""
Command line interface for running correlation calibration.
"""
from corrcal.parsers import corrcal_parser
from corrcal.corrcal import corrcal_run
from corrcal import LOGGER
import logging

args = corrcal_parser.parse_args()
if args.verbose:
    LOGGER.setLevel(logging.INFO)
corrcal_run(
    args.input_data,
    noise_model=args.noise,
    sky_model=args.sky,
    outdir=args.outdir,
    clobber=args.clobber,
    parallel=args.parallel,
    gpu=args.gpu,
)
