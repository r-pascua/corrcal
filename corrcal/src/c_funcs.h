struct sparse_cov{
    complex *noise;
    complex *diff_mat;
    complex *src_mat;
    int n_bl;
    int n_eig;
    int n_src;
    int n_grp;
    int *edges;
    int isinv;
};

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
);
