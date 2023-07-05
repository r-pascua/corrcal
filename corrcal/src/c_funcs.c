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
    long *edges,
    int isinv
) {
    /*
     *  struct sparse_cov *init_cov(
     *      complex *noise,
     *      complex *diff_mat,
     *      complex *src_mat,
     *      int n_bl,
     *      int n_eig,
     *      int n_src,
     *      int n_grp,
     *      long *edges,
     *      int isinv
     *  )
     *
     *  Routine for initializing a sparse covariance structure.
     *
     *  Parameters
     *  ----------
     *  noise
     *      Diagonal of noise variance matrix.
     *  diff_mat
     *      Block-diagonal elements of the diffuse matrix, sorted by redundant
     *      groups. Should have shape (n_bl, n_eig). See discussion in Section
     *      ?? of Pascua+ 2023 for details.
     *  src_mat
     *      Source matrix, sorted by redundant groups. Should have shape
     *      (n_bl, n_src). See discussion in Section ?? of Pascua+ 2023 for
     *      details.
     *  n_bl
     *      Number of baselines in the data.
     *  n_eig
     *      Number of eigenmodes used for each redundant group in the diffuse
     *      matrix.
     *  n_src
     *      Number of sources used in the sky model.
     *  n_grp
     *      Number of redundant groups.
     *  edges
     *      Array indexing the edges of each redundant group. For example,
     *      edges[i] gives the starting index of redundant group i.
     *  isinv
     *      Whether the covariance has been inverted (i.e., whether the source
     *      and diffuse matrices are primed, as discussed in Section ?? of
     *      Pascua+ 2023).
     *
     *  Returns
     *  -------
     *  sparse_cov
     *      Structure containing all of the information necessary for working
     *      with the sparse representation of the covariance.
     */
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
    /*
     *  void mamtul(
     *      *complex left, complex *right, complex *out, int a, int b, int c
     *  )
     *
     *  Parallelized routine for performing out = left @ right.
     *
     *  Parameters
     *  ----------
     *  left, right
     *      Matrices to be multiplied together. Left has shape (a,b), and
     *      right has shape (b,c).
     *  out
     *      Where to write the output of the matrix product left @ right.
     *      Must have shape (a,c).
     *  a
     *      Number of rows in left.
     *  b
     *      Number of columns in left and rows in right.
     *  c
     *      Number of columns in right.
     */
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
     *  void matmul(
     *      complex *left, complex *right, complex *out,
     *      int stridea, int strideb, int stridec,
     *      int m, int n, int l
     *  )
     *
     *  Matrix multiplication that supports block-multiplication.
     *  For regular
     *  matrix multiplication, with shape(left) = (m,n), shape(right) = (n,l),
     *  set stridea = n, strideb = l, stridec = l.
     *
     *  Parameters
     *  ----------
     *  left, right
     *      Matrices to multiply together.
     *  out
     *      Where to write the output of the matrix product.
     *  stridea
     *      Number of columns in left.
     *  strideb
     *      Number of columns in right.
     *  stridec
     *      Number of columns in out.
     *  m
     *      Number of rows contained in this block of left.
     *  n
     *      Number of columns contained in this block of right.
     *  l
     *      Number of columns in this block of left and number of rows in
     *      this block of right.
     *
     *  Notes
     *  -----
     *  When used for performing block-multiplication, this function can only
     *  multiply one block with one other block. For regular matrix
     *  multiplication, where left has shape (m,n) and right has shape (n,l),
     *  set stridea=n, strideb=l, stridec=l.
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
    int n_eig, int n_grp
) {
    /*
     *  void block_multiply(
     *      complex *blocks, complex *diffuse_mat, complex *out,
     *      long *edges, int n_eig, int n_grp
     *  )
     *
     *  Multiply small blocks by diffuse matrix from the left.
     *
     *  Parameters
     *  ----------
     *  blocks
     *      Array of small square matrices sorted by redundant groups. The
     *      array should have shape (n_grp, n_eig, n_eig) (i.e., it contains
     *      n_grp square blocks each with shape (n_eig, n_eig)).
     *  diffuse_mat
     *      Diffuse matrix sorted into redundant groups, with shape (n, n_eig).
     *  out
     *      Where to write the product. Should have shape (n_bl, n_eig).
     *  edges
     *      Array specifying the edges of each redundant group. For example,
     *      edges[i] gives the start of group i.
     *  n_eig
     *      Number of eigenmodes used to describe each redundant group.
     *  n_grp
     *      Number of redundant groups.
     *
     *  Notes
     *  -----
     *  This function is meant to be used in the first application of the
     *  Woodbury identity in the inversion routine: it performs the operation
     *  out = diffuse_mat @ blocks block-by-block. See discussion in Section
     *  ?? of Pascua+ 2023 for details. (This is used for Step 1b in the
     *  prelim presentation, slide 45.)
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
     *  void mult_src_by_blocks(
     *      complex *blocks_H, complex *src_mat, complex *out, long *edges,
     *      int n_bl, int n_src, int n_eig, int n_grp
     *  )
     *
     *  Multiply source matrix by block-diagonal matrix from the left.
     *
     *  Parameters
     *  ----------
     *  blocks_H
     *      Hermitian conjugate of the block-diagonal entries in the "inverse"
     *      diffuse matrix. Should have shape (n_eig, n_bl).
     *  src_mat
     *      Source matrix with shape (n_bl, n_src).
     *  out
     *      Where to write the output of the matrix product.
     *  edges
     *      Array specifying the edges of each redundant group. For example,
     *      edges[i] gives the start of group i.
     *  n_bl
     *      Number of baselines in the data.
     *  n_src
     *      Number of sources used in the sky model.
     *  n_eig
     *      Number of eigenmodes used for each redundant group.
     *  n_grp
     *      Number of redundant groups.
     *
     *  Notes
     *  -----
     *  This function is meant to be used in the second application of the
     *  Woodbury identity in the inversion routine. It performs the matrix
     *  multiplication of the "inverse" diffuse matrix and the source matrix
     *  through repeated matrix-vector multiplications between the redundant
     *  blocks and the source vectors. See discussion in Section ?? of Pascua+
     *  2023 for details. (This is used in Step 2a in the prelim presentation,
     *  slide 49.)
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


void mult_src_blocks_by_diffuse(
    complex *inv_diff_mat, complex *src_blocks, complex *out, long *edges,
    int n_src, int n_eig, int n_grp
) {
    /*
     *  Compute the inverse diffuse covariance times the source matrix.
     *
     *  Parameters
     *  ----------
     *  inv_diff_mat
     *      "Inverse" of the diffuse matrix, with shape (n_bl, n_eig).
     *  src_blocks
     *      Product of the Hermitian conjugate of the "inverse" diffuse
     *      matrix and the source matrix, with shape (n_eig, n_src).
     *  out
     *      Where to write the output of the matrix product.
     *  edges
     *      Array specifying the edges of each redundant group. For example,
     *      edges[i] gives the start of group i.
     *  n_src
     *      Number of sources used in the sky model.
     *  n_eig
     *      Number of eigenmodes used for each redundant group.
     *  n_grp
     *      Number of redundant groups.
     */
    for (int grp=0; grp<n_grp; grp++) {
        for (int src=0; src<n_src; src++) {
            mymatmul(
                inv_diff_mat+edges[grp]*n_eig,
                src_blocks+src+grp*n_eig*n_src,
                out+edges[grp]*n_src+src,
                n_eig,
                n_src,
                n_src,
                edges[grp+1]-edges[grp],
                1,
                n_eig
            );
        }
    }
}


void tril_inv(complex *mat, complex *out, int n) {
    /*
     *  void tril_inv(complex *mat, complex *out, int n)
     *
     *  Invert a lower-triangular (n,n) matrix.
     *
     *  Parameters
     *  ----------
     *  mat
     *      Lower-triangular matrix to invert.
     *  out
     *      Where to write the inverse.
     *  n
     *      Number of rows/columns in the matrix.
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
     *  void many_tril_inv(complex *mat, complex *out, int n, int n_block)
     *
     *  Invert n_block lower-triangular matrices in parallel.
     *
     *  Parameters
     *  ----------
     *  mat
     *      Array of matrices to invert, with shape (n_block, n, n).
     *  out
     *      Where to write the output. Should have the same shape as mat.
     *  n
     *      Number of rows/columns in each block.
     *  n_block
     *      Number of matrices to invert.
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
     *
     *  Parameters
     *  ----------
     *  mat
     *      Matrix to perform Cholesky decomposition on.
     *  out
     *      Where to write the Cholesky decomposition.
     *  n
     *      Number of rows/columns in the input matrix.
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
     *
     *  Parameters
     *  ----------
     *  mat
     *      Matrix to perform in-place Cholesky decomposition on.
     *  n
     *      Number of rows/columns in the input matrix.
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
    /*
     *  void many_chol(
     *      complex *blocks, complex *out, int block_size, int n_blocks
     *  )
     *
     *  Perform Cholesky decomposition on many matrices in parallel.
     *
     *  Parameters
     *  ----------
     *  blocks
     *      Array of matrices to perform Cholesky decomposition on. Should
     *      have shape (n_blocks, block_size, block_size).
     *  out
     *      Where to write the output.
     *  block_size
     *      Number of rows/columns in each matrix.
     *  n_blocks
     *      Number of matrices to decompose.
     */
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
     *  Perform in-place Cholesky decomposition on many matrices in parallel.
     *
     *  Parameters
     *  ----------
     *  blocks
     *      Array of matrices to perform Cholesky decomposition on. Should
     *      have shape (n_blocks, block_size, block_size).
     *  block_size
     *      Number of rows/columns in each matrix.
     *  n_blocks
     *      Number of matrices to decompose.
     */
    #pragma omp parallel for
    for (int i=0; i<n_blocks; i++) {
        int offset = i * block_size * block_size;
        cholesky_inplace(blocks+offset, block_size);
    }
}


void make_small_block(
    complex *noise_diag, complex *diffuse_mat, complex *out,
    int n_eig, int start, int stop
) {
    /*
     *  void make_small_block(
     *      complex *noise_diag, complex *diffuse_mat, complex *out,
     *      int n_eig, int start, int stop
     *  )
     *
     *  Make one small block matrix as part of the inversion routine.
     *
     *  Parameters
     *  ----------
     *  noise_diag
     *      Diagonal of the noise variance matrix.
     *  diffuse_mat
     *      Diffuse matrix sorted into redundant groups.
     *  out
     *      Where to write the output.
     *  n_eig
     *      Number of eigenmodes used for each redundant group.
     *  start
     *      Starting index for this redundant group.
     *  stop
     *      Ending index for this redundant group.
     *
     *  Notes
     *  -----
     *  This function performs the operation out = diff_mat.H @ Ninv @ diff_mat
     *  for one redundant group as part of the first application of the
     *  Woodbury identity. See the discussion in Section ?? of Pascua+ 2023
     *  for details.
     */
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
    complex *noise_diag, complex *diffuse_mat, complex *out,
    long *edges, int n_eig, int n_block
) {
    /*
     *  void make_all_small_blocks(
     *      complex *noise_diag, complex *diffuse_mat, complex *out,
     *      long *edges, int n_eig, int n_block
     *  )
     *
     *  Make all small blocks for use in inversion routine.
     *
     *  Parameters
     *  ----------
     *  noise_diag
     *      Diagonal of the noise variance matrix.
     *  diffuse_mat
     *      Diffuse matrix sorted into redundant groups.
     *  out
     *      Where to write the output.
     *  edges
     *      Array specifying the edges of each redundant group. For example,
     *      edges[i] gives the start of group i.
     *  n_eig
     *      Number of eigenmodes used for each redundant group.
     *  n_block
     *      Number of redundant groups.
     *
     *  Notes
     *  -----
     *  This function performs the operation out = diff_mat.H @ Ninv @ diff_mat
     *  for all redundant groups as part of the first application of the
     *  Woodbury identity. See the discussion in Section ?? of Pascua+ 2023
     *  for details.
     */
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


void sparse_cov_times_vec(struct sparse_cov *cov, complex *vec, complex *out) {
    /*
     *  void sparse_cov_times_vec(
     *      struct sparse_cov *cov, complex *vec, complex *out
     *  )
     *
     *  Multiply a vector by the sparse covariance from the left.
     *
     *  Parameters
     *  ----------
     *  cov
     *      Structure containing the sparse covariance information.
     *  vec
     *      Vector to multiply by the sparse covariance.
     *  out
     *      Where to write the output.
     */
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
    long *edges,
    int isinv,
    complex *vec,
    complex *out
) {
    /*
     *  void sparse_cov_times_vec_wrapper(
     *      complex *noise,
     *      complex *diff_mat,
     *      complex *src_mat,
     *      int n_bl,
     *      int n_eig,
     *      int n_src,
     *      int n_grp,
     *      long *edges,
     *      int isinv,
     *      complex *vec,
     *      complex *out
     *  )
     *
     *  Thin wrapper for performing sparse matrix-vector multiplication.
     *
     *  Parameters
     *  ----------
     *  noise
     *      Diagonal of the noise variance matrix sorted by redundant groups.
     *  diff_mat
     *      Diffuse matrix sorted by redundant groups. Should have shape
     *      (n_bl, n_eig).
     *  src_mat
     *      Source matrix sorted by redundant groups. Should have shape
     *      (n_bl, n_src).
     *  n_bl
     *      Number of baselines.
     *  n_eig
     *      Number of eigenmodes used for each redundant group.
     *  n_src
     *      Number of sources used in the sky model.
     *  n_grp
     *      Number of redundant groups.
     *  edges
     *      Array specifying the edges of each redundant group. For example,
     *      edges[i] gives the start of group i.
     *  isinv
     *      Whether the covariance has been inverted or not.
     *  vec
     *      Vector to multiply by the sparse covariance.
     *  out
     *      Where to write the output.
     */
    struct sparse_cov *cov = init_cov(
        noise, diff_mat, src_mat, n_bl, n_eig, n_src, n_grp, edges, isinv
    );
    sparse_cov_times_vec(cov, vec, out);
    free(cov);
}
