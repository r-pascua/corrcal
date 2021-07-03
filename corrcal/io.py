import numpy as np

from . import Sparse2Level


def read_sparse(fname):
    """Load the contents of a binary file into a Sparse2Level object.

    Parameters
    ----------
    fname
        Path to the binary file to be read.

    Returns
    -------
    sparse
        :class:`~.Sparse2Level` object constructed from the binary file.

    Raises
    ------
    IOError
        If the file is not formatted appropriately (i.e. it does not have
        the correct number of lines).

    Notes
    -----
    The binary file should be formatted as follows::
        * The first line is a 32-bit int specifying the length of the data,
          and it should be twice the number of baselines (since the data is
          structured to alternate between real and imaginary components).
          This number is referred to as ``Ndata`` in the function body.
        * The second line is a 32-bit int specifying whether the covariance
          has been inverted prior to being written to disk (i.e. 0 or 1).
        * The third line is a 32-bit int specifying the number of sources
          used in the point source model. For a single frequency, this
          specifies the size of the 0-th axis of the S-matrix. This number
          is referred to as ``Nsrc`` in the function body.
        * The fourth line is a 32-bit int specifying the number of quasi-
          redundant blocks (or rather the number of redundant groups up to
          some non-redundancy tolerance). This number is referred to as
          ``Nblock`` in the function body.
        * The fifth line is a 32-bit integer specifying the number of rows
          in the matrix describing the covariance from diffuse emission.
          It can be thought of (I think) as the number of eigenmodes needed
          to characterize the contributions to the covariance from diffuse
          emission. This number is referred to as ``Neig`` in the function
          body.
        * The next ``Nblock + 1`` line specify the indices of the edges of
          the quasi-redundant blocks in the data vector, making sure to
          account for the extra length induced from having the data
          alternate between real and imaginary components.
        * The next ``Ndata`` lines give the alternating real/imaginary parts
          of the thermal noise.
        * The next ``Neig * Ndata`` lines give the eigenvectors describing
          the diffuse emission. This should be organized into ``Ndata``
          segments, each of length ``Neig``. Each segment would then
          represent the alternating real/imaginary parts of the corresponding
          eigenvector.
        * The next ``Nsrc * Ndata`` lines provide the point source model.
          These lines should be arranged into ``Ndata`` segments, each of
          length ``Nsrc``, following the same format as the previous section
          of the file.
    """
    # Read the contents of the file.
    with open(fname, "r") as f:
        Ndata = np.fromfile(f, "int32", 1)[0]
        isinv = bool(np.fromfile(f, "int32", 1)[0])
        Nsrc = np.fromfile(f, "int32", 1)[0]
        Nblock = np.fromfile(f, "int32", 1)[0]
        Neig = np.fromfile(f, "int32", 1)[0]
        group_edges = np.fromfile(f, "int32", Nblock + 1)
        noise_variance = np.fromfile(f, "float64", Ndata)
        diffuse_vectors = np.fromfile(f, "float64", Neig * Ndata)
        source_vectors = np.fromfile(f, "float64", Nsrc * Ndata)
        extra_lines = np.fromfile(f)

    if extra_lines.size > 0:
        raise IOError(
            f"{fname} is not formatted properly. Please refer to the "
            "documentation to determine how the file should be formatted."
        )

    if diffuse_vectors.size > 0:
        diffuse_vectors = diffuse_vectors.reshape(Neig, Ndata)

    if source_vectors.size > 0:
        source_vectors = source_vectors.reshape(Nsrc, Ndata)

    return Sparse2Level(
        noise_variance=noise_variance,
        diffuse_vectors=diffuse_vectors,
        source_vectors=source_vectors,
        group_edges=group_edges,
        isinv=isinv,
    )
