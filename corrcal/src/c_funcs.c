#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <complex.h>
#include <math.h>
#include "c_funcs.h"

struct sparse_cov *init_cov(
    complex *noise,
    complex *diff_mat,
    complex *src_mat,
    int n_bl,
    int n_eig,
    int n_src,
    int n_grp,
    int *edges,
    int isinv
) {
    struct sparse_cov *cov = (struct sparse_cov *)malloc(sizeof(struct sparse_cov));
    cov->noise = noise;
    cov->diff_mat = diff_mat;
    cov->src_mat = src_mat;
    cov->n_bl = n_bl;
    cov->n_eig = n_eig;
    cov->n_src = n_src;
    cov->n_grp = n_grp;
    cov->edges = edges;
    cov->isinv = isinv;

    return cov;
}


void matmul(complex *left, complex *right, complex *out, int a, int b, int c) {
    #pragma omp parallel for
    for (int ij=0; ij<a*c; ij++) {
        int i = ij / c;
        int j = ij % c;
        complex sum = 0;
        for (int k=0; k<b; k++) {
            sum += left[i*b+k] * right[k*c+j];
        }
        out[i*c+j] = sum;
    }
}


void mymatmul(
    complex *left, complex *right, complex *out, int stridea, int strideb,
    int stridec, int m, int n, int l
) {
    /*
     *  Matrix multiplication that supports block-multiplication. For regular
     *  matrix multiplication, with shape(left) = (m,n), shape(right) = (n,l),
     *  set stridea = n, strideb = l, stridec = l.
     *
     */

    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++) {
            complex tmp = 0;
            for (int k=0; k<l; k++) {
                tmp += left[i*stridea+k] * right[k*strideb+j];
            }
            out[i*stridec+j] = tmp;
        }
    }
}


void block_multiply(
    complex *blocks, complex *diffuse_mat, complex *out, long *edges,
    int n_eig, int n_grp, int n
) {
    /*
     *  Multiply diffuse matrix by small block matrices from the left.
     */
    for (int grp=0; grp<n_grp; grp++) {
        mymatmul(
            diffuse_mat+n_eig*edges[grp],
            blocks+grp*n_eig*n_eig,
            out+n_eig*edges[grp],
            n_eig,
            n_eig,
            n_eig,
            edges[grp+1]-edges[grp],
            n_eig,
            n_eig
        );
    }
}


void mult_src_by_blocks(
    complex *blocks_H, complex *src_mat, complex *out, long *edges,
    int n_bl, int n_src, int n_eig, int n_grp
) {
    /*
     *  Block multiplication of small Cholesky decomp and source matrix.
     */
    for (int grp=0; grp<n_grp; grp++) {
        for (int src=0; src<n_src; src++) {
            mymatmul(
                blocks_H+edges[grp],
                src_mat+src+edges[grp]*n_src,
                out+src+grp*n_eig*n_src,
                n_bl,
                n_src,
                n_src,
                n_eig,
                1,
                edges[grp+1]-edges[grp]
            );
        }
    }
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


void many_tril_inv(complex *mat, complex *out, int n, int n_block) {
    /*
     *  void many_tril_inv(complex *mat, complex *out, int n, int n_block
     *
     *  Invert n_block lower-triangular matrices each with shape (n,n).
     */
    #pragma omp parallel for
    for (int i=0; i<n_block; i++) {
        tril_inv(mat+i*n*n, out+i*n*n, n);
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
}


void many_chol(complex *blocks, complex *out, int block_size, int n_blocks) {
    #pragma omp parallel for
    for (int i=0; i<n_blocks; i++) {
        int offset = i * block_size * block_size;
        cholesky(blocks+offset, out+offset, block_size);
    }
}


void many_chol_inplace(complex *blocks, int block_size, int n_blocks) {
    /*
     *  void many_chol(complex *blocks, int *block_sizes, int n_blocks)
     *
     *  Calculate the Cholesky decomposition of a list of matrices in-place.
     */
    #pragma omp parallel for
    for (int i=0; i<n_blocks; i++) {
        int offset = i * block_size * block_size;
        cholesky_inplace(blocks+offset, block_size);
    }
}


void make_small_block(
    complex *noise_diag, complex *diffuse_mat, complex *out, int n_eig, int start, int stop
) {
    for (int i=0; i<n_eig; i++) {
        for (int j=i; j<n_eig; j++) {
            complex sum = 0;
            for (int k=start; k<stop; k++) {
                complex tmp = (
                    conj(diffuse_mat[k*n_eig+i]) * diffuse_mat[k*n_eig+j]
                );
                sum += tmp / noise_diag[k];
            }
            out[i*n_eig+j] = sum;
            out[j*n_eig+i] = conj(sum);
        }
    }
}


void make_all_small_blocks(
    complex *noise_diag, complex *diffuse_mat, complex *out, long *edges, int n_eig, int n_block
) {
    for (int i=0; i<n_block; i++) {
        make_small_block(
            noise_diag,
            diffuse_mat,
            out+i*n_eig*n_eig,
            n_eig,
            edges[i],
            edges[i+1]
        );
    }   
}


void sparse_cov_times_vec(struct sparse_cov *cov, complex *vec, complex *out){
    // Iniitialize the output.
    memset(out, 0, sizeof(complex) * cov->n_bl);

    // Multiply by the noise variance.
    for (int i=0; i<cov->n_bl; i++){
        out[i] += cov->noise[i] * vec[i];
    }


    // Only do this check once.
    if (cov->isinv) {
        // Multiply by the diffuse covariance in blocks.
        for (int i=0; i<cov->n_grp; i++) {
            for (int j=0; j<cov->n_eig; j++) {
                complex tmp = 0;
                for (int k=cov->edges[i]; k<cov->edges[i+1]; k++) {
                    tmp += vec[k] * conj(cov->diff_mat[k*cov->n_eig+j]);
                }
                for (int k=cov->edges[i]; k<cov->edges[i+1]; k++) {
                    out[k] -= tmp * cov->diff_mat[k*cov->n_eig+j];
                }
            }
        }

        // Multiply by the source covariance.
        for (int i=0; i<cov->n_src; i++) {
            complex tmp = 0;
            for (int j=0; j<cov->n_bl; j++) {
                tmp += vec[j] * conj(cov->src_mat[j*cov->n_src+i]);
            }
            for (int j=0; j<cov->n_bl; j++) {
                out[j] -= tmp * cov->src_mat[j*cov->n_src+i];
            }
        }
    } else {
        for (int i=0; i<cov->n_grp; i++) {
            for (int j=0; j<cov->n_eig; j++) {
                complex tmp = 0;
                for (int k=cov->edges[i]; k<cov->edges[i+1]; k++) {
                    tmp += vec[k] * conj(cov->diff_mat[k*cov->n_eig+j]);
                }
                for (int k=cov->edges[i]; k<cov->edges[i+1]; k++) {
                    out[k] += tmp * cov->diff_mat[k*cov->n_eig+j];
                }
            }
        }

        for (int i=0; i<cov->n_src; i++) {
            complex tmp = 0;
            for (int j=0; j<cov->n_bl; j++) {
                tmp += vec[j] * conj(cov->src_mat[j*cov->n_src+i]);
            }
            for (int j=0; j<cov->n_bl; j++) {
                out[j] += tmp * cov->src_mat[j*cov->n_src+i];
            }
        }
    }
}


void sparse_cov_times_vec_wrapper(
    complex *noise,
    complex *diff_mat,
    complex *src_mat,
    int n_bl,
    int n_eig,
    int n_src,
    int n_grp,
    int *edges,
    int isinv,
    complex *vec,
    complex *out
) {
    struct sparse_cov *cov = init_cov(
        noise, diff_mat, src_mat, n_bl, n_eig, n_src, n_grp, edges, isinv
    );
    sparse_cov_times_vec(cov, vec, out);
    free(cov);
}
