#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <complex.h>
#include <math.h>

double woodbury_with_det(
    complex *U, complex *C, complex *out, int n, int k
) {
    /*
     *  double woodbury_with_det(
     *      *complex U, complex *C, complex*out, int n, int k
     *  )
     *
     *  Invert the matrix C + UU^\dagger with the Woodbury Identity and
     *  accumulate the logarithm of the determinant with the Matrix
     *  Determinant Lemma. Instead of implementing the fully general
     *  Woodbury Identity (i.e. inverting A + UCV), this function implements
     *  the inversion routine as used in CorrCal.
     *
     *  Parameters
     *  ----------
     *  U
     *      Complex-valued (n,k) matrix. This is either GS or GD, where G is
     *      the gain matrix, S is the source matrix, and D is the diffuse
     *      matrix.
     *  C
     *      Complex-valued (n,n) matrix. This is either the noise variance N
     *      or the inverse of N + GD(GD)^\dagger.
     *  out
     *      Complex-valued (n,n) matrix where the output will be written.
     *  n
     *      Number of rows in the U/C matrices.
     *  k
     *      Number of columns in the U matrix.
     */

}


void woodbury(
    complex *U, complex *C, complex *out, int n, int k
) {
    /*
     *  void woodbury(
     *      *complex U, complex *C, complex*out, int n, int k
     *  )
     *
     *  Invert the matrix C + UU^\dagger with the Woodbury Identity. Instead
     *  of implementing the fully general Woodbury Identity (i.e. inverting
     *  A + UCV), this function implements the inversion routine as used in
     *  CorrCal.
     *
     *  Parameters
     *  ----------
     *  U
     *      Complex-valued (n,k) matrix. This is either GS or GD, where G is
     *      the gain matrix, S is the source matrix, and D is the diffuse
     *      matrix.
     *  C
     *      Complex-valued (n,n) matrix. This is either the noise variance N
     *      or the inverse of N + GD(GD)^\dagger.
     *  out
     *      Complex-valued (n,n) matrix where the output will be written.
     *  n
     *      Number of rows in the U/C matrices.
     *  k
     *      Number of columns in the U matrix.
     */
    
}


void matmul(complex *left, complex *right, complex *out, int a, int b, int c) {
    /*
     *
     */

}


void tril_inv(complex *mat, complex *out, int n) {
    /*
     *  void tril_inv(complex *mat, complex *out, int n)
     *
     *  Invert a lower-triangular (n,n) matrix.
     */
    for (int i=0; i<n; i++) {
        out[i*n+i] = 1 / mat[i*n+i];
        for (int j=0; j<i; j++) {
            complex tmp = 0;
            for (int k=0; k<i; k++) {
                tmp += mat[i*n+k] * out[k*n+j];
            }
            out[i*n+j] = -tmp / mat[i*n+i];
        }
    }
}


void many_tril_inv(
    complex *blocks, complex *out, int block_size, int n_blocks
) {
    /*
     *  void many_tril_inv(
     *      complex *blocks, complex *out, int block_size, int n_blocks
     *  )
     *
     *  Invert many lower triangular matrices of the same shape in parallel.
     */
    #pragma omp parallel for
    for (int i=0; i<n_blocks; i++) {
        int offset = i*block_size*block_size;
        tril_inv(blocks+offset, out+offset, block_size);
    }
}


void many_irregular_tril_inv(
    complex *blocks, complex *out, int *block_sizes, int n_blocks
) {
    /*
     *  void many_irregular_tril_inv(
     *      complex *blocks, complex *out, int *block_sizes, int n_blocks
     *  )
     *
     *  Compute the inverse of a collection of lower-triangular matrices of
     *  varying sizes.
     *
     *  Parameters
     *  ----------
     *  blocks
     *      Matrices to invert.
     *  out
     *      Where to write the inverses.
     *  block_sizes
     *      Size of each matrix to invert.
     *  n_blocks
     *      Number of matrices to invert.
     */
    int offsets[n_blocks];
    int offset = 0;
    for (int i=0; i<n_blocks; i++) {
        offsets[i] = offset;
        offset += block_sizes[i] * block_sizes[i];
    }

    #pragma omp parallel for
    for (int i=0; i<n_blocks; i++) {
        tril_inv(blocks+offsets[i], out+offsets[i], block_sizes[i]);
    }
}


void cholesky(complex *mat, complex *out, int n) {
    /*
     *  void cholesky(complex *mat, complex *out, int n)
     *
     *  Compute the Cholesky decomposition of an (n,n) Hermitian matrix.
     */
    for (int i=0; i<n; i++) {
        for (int j=0; j<=i; j++) {
            complex tmp = 0;
            for (int k=0; k<j; k++) {
                tmp += out[i*n+k] * conj(out[j*n+k]);
            }
            if (i == j) {
                out[i*n+i] = sqrt(mat[i*n+i] - tmp);
            } else {
                out[i*n+j] = (mat[i*n+j] - tmp) / out[j*n+j];
            }
        }
    }

    // Zero out the upper-triangular section
    for (int i=0; i<n; i++) {
        for (int j=i+1; j<n; j++) {
            out[i*n+j] = 0;
        }
    }
}


void cholesky_inplace(complex *mat, int n) {
    /*
     *  void cholesky_inplace(complex *mat, int n)
     *
     *  Compute the Cholesky decomposition of an (n,n) matrix in-place.
     */
    for (int i=0; i<n; i++) {
        for (int j=0; j<=i; j++) {
            complex tmp = 0;
            for (int k=0; k<j; k++) {
                tmp += mat[i*n+k] * conj(mat[j*n+k]);
            }
            if (i == j) {
                mat[i*n+i] = sqrt(mat[i*n+i] - tmp);
            } else {
                mat[i*n+j] = (mat[i*n+j] - tmp) / mat[j*n+j];
            }
        }
    }

    // Zero out the upper-triangular section.
    for (int i=0; i<n; i++) {
        for (int j=i+1; j<n; j++) {
            mat[i*n+j] = 0;
        }
    }
}


void many_chol(
    complex *blocks, complex *out, int block_size, int n_blocks
) {
    /*
     *  void many_chol(
     *      complex *blocks, complex *out, int block_size, int n_blocks
     *  )
     *
     *  Calculate the Cholesky decomposition of a collection of regularly
     *  sized matrices in parallel.
     */
    #pragma omp parallel for
    for (int i=0; i<n_blocks; i++) {
        int offset = i * block_size * block_size;
        cholesky(blocks+offset, out+offset, block_size);
    }
}


void many_chol_inplace(complex *blocks, int block_size, int n_blocks) {
    /*
     *  void many_chol_inplace(
     *      complex *blocks, int block_size, int n_blocks
     *  )
     *
     *  Calculate the Cholesky decomposition of a collection of regularly
     *  sized matrices in-place, in parallel.
     */
    #pragma omp parallel for
    for (int i=0; i<n_blocks; i++) {
        int offset = i * block_size * block_size;
        cholesky_inplace(blocks+offset, block_size);
    }
}


void many_irregular_chol(
    complex *blocks, complex *out, int *block_sizes, int n_blocks
) {
    /*
     *  void many_irregular_chol(
     *      complex *blocks, complex *out, int *block_sizes, int n_blocks
     *  )
     *
     *  Calculate the Cholesky decomposition of a collection of matrices of
     *  varying sizes.
     *
     *  Parameters
     *  ----------
     *  blocks
     *      Array of matrices.
     *  out
     *      Where to write the Cholesky decomposition of the input matrices.
     *  block_sizes
     *      Size of each matrix.
     *  n_blocks
     *      Number of matrices.
     */ 
    int offsets[n_blocks];
    int offset = 0;
    for (int i=0; i<n_blocks; i++) {
        offsets[i] = offset;
        offset += block_sizes[i] * block_sizes[i];
    }

    #pragma omp parallel for
    for (int i=0; i<n_blocks; i++) {
        cholesky(blocks+offsets[i], out+offsets[i], block_sizes[i]);
    }
}


void many_irregular_chol_inplace(
    complex *blocks, int *block_sizes, int n_blocks
) {
    /*
     *  void many_irregular_chol_inplace(
     *      complex *blocks, int *block_sizes, int n_blocks
     *  )
     *
     *  Calculate the Cholesky decomposition of a list of matrices in-place.
     *
     *  Parameters
     *  ----------
     *  blocks
     *      Array of matrices.
     *  block_sizes
     *      Size of each matrix. Note that these are 32-bit integers.
     *  n_blocks
     *      Number of matrices.
     */
    int offsets[n_blocks];
    int offset = 0;
    for (int i=0; i<n_blocks; i++) {
        offsets[i] = offset;
        offset += block_sizes[i] * block_sizes[i];
    }

    #pragma omp parallel for
    for (int i=0; i<n_blocks; i++) {
        cholesky_inplace(blocks+offsets[i], block_sizes[i]);
    }
}
