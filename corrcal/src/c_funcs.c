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
    return 0;
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


void make_small_block(
    complex *noise_diag, complex *diffuse_mat, complex *out, int n_eig,
    int start, int stop
) {
    /*
     *  void make_small_block(
     *      complex *noise_diag, complex *diffuse_mat, int n_eig, int n_bl
     *  )
     *
     *  Construct the small block D^\dag N^-1 D (see Eq. ? of Pascua+ 23).
     *  It is assumed that ``diffuse_mat`` has already been scaled by the gain
     *  matrix prior to calling this routine.
     */
    for (int i=0; i<n_eig; i++) {
        // We only need to do the upper triangular part, since it's Hermitian
        for (int j=i; j<n_eig; j++) {
            complex sum = 0;
            for (int k=start; k<stop; k++) {
                complex tmp = (
                    conj(diffuse_mat[k*n_eig+i]) * diffuse_mat[k*n_eig+j]
                );
                sum += tmp / noise_diag[k];
            }
            out[i*n_eig+j] = sum;
            if (i != j) {
                out[j*n_eig+i] = conj(sum);
            }
        }
    }
}


void make_all_small_blocks(
    complex *noise_diag, complex *diffuse_mat, complex *out, int *edges,
    int n_eig, int n_block
) {
    /*
     *  void make_all_small_blocks(
     *      complex *noise_diag, complex *diffuse_mat, complex *out,
     *      long *edges, int n_eig, int n_block
     *  )
     *
     *  Make all ``(n_eig,n_eig)`` blocks, given the block-diagonal diffuse
     *  matrix and the noise variance.
     */
    #pragma omp parallel for
    for (int i=0; i<n_block; i++) {
        int n_bl = edges[i+1] - edges[i];
        make_small_block(
            noise_diag,
            diffuse_mat,
            out + i*n_eig*n_eig,
            n_eig,
            edges[i],
            edges[i+1]
        );
    }
}


void block_multiply(){}
