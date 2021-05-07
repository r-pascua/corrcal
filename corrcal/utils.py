def check_parallel(parallel, gpu):
    """
    Ensure that only parallelization or GPU acceleration is requested.

    Parameters
    ----------
    parallel: bool
        Whether to perform the operation in parallel.
    gpu: bool
        Whether to use GPU acceleration.
    """
    if parallel and gpu:
        raise ValueError(
            "CPU parallelization and GPU acceleration cannot be "
            "performed simultaneously."
        )
