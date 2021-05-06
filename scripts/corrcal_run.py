"""
Command line interface for running correlation calibration.
"""
from corrcal.parsers import corrcal_parser
from corrcal.corrcal import corrcal_pipe
from corrcal import LOGGER
import logging

args = corrcal_parser.parse_args()
if args.verbose:
    LOGGER.setLevel(logging.INFO)
corrcal_pipe(
    args.input_data,
    noise_model=args.noise,
    sky_model=args.sky,
    cov_model=args.cov,
    algorithm=args.algorithm,
    outdir=args.outdir,
    clobber=args.clobber,
    parallel=args.parallel,
    nproc=args.nproc,
    gpu=args.gpu,
    # TODO: figure out whether extra GPU args are required.
)
