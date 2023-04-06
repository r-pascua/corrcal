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


void cholesky(complex *mat, complex *out, int n) {
    /*
     *  void cholesky(complex *mat, complex *out, int n)
     *
     *  Compute the Cholesky decomposition of an (n,n) Hermitian matrix.
     */
}


void cholesky_inplace(complex *mat, int n) {
    /*
     *  void cholesky_inplace(complex *mat, int n)
     *
     *  Compute the Cholesky decomposition of an (n,n) matrix in-place.
     */
}
