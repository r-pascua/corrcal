import argparse

corrcal_parser = argparse.ArgumentParser(
    description="Run correlation calibration from the command line."
)
corrcal_parser.add_argument(
    "input_data",
    nargs="*",
    default=None,
    type=str,
    help="Path to input data file(s). See documentation for details.",
)
corrcal_parser.add_argument(
    "-N", "--noise", default=None, type=str, help="Path to noise model.",
)
corrcal_parser.add_argument(
    "-S",
    "--sky",
    default=None,
    type=str,
    help="Path to sky covariance model.",
)
corrcal_parser.add_argument(
    "-C",
    "--cov",
    default=None,
    type=str,
    help="Path to a sparse covariance model.",
)
corrcal_parser.add_argument(
    "-a",
    "--algorithm",
    default="conjugate gradient",
    type=str,
    help="Name of optimization routine to use.",
)
corrcal_parser.add_argument(
    "-o",
    "--outdir",
    default=None,
    type=str,
    help="Where to write calibration products.",
)
corrcal_parser.add_argument(
    "-v", "--verbose", default=False, action="store_true",
)
corrcal_parser.add_argument(
    "--clobber",
    default=False,
    action="store_true",
    help="Whether to overwrite existing files with name conflicts.",
)
corrcal_parser.add_argument(
    "--parallel",
    default=False,
    action="store_true",
    help="Whether to parallelize calibration.",
)
corrcal_parser.add_argument(
    "--gpu",
    default=False,
    action="store_true",
    help="Whether to use GPU acceleration where possible.",
)
