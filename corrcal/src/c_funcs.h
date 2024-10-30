struct sparse_cov{
    double *noise;
    double *diff_mat;
    double *src_mat;
    int n_bl;
    int n_eig;
    int n_src;
    int n_grp;
    long *edges;
    int isinv;
};

struct sparse_cov *init_cov(
    double *noise,
    double *diff_mat,
    double *src_mat,
    int n_bl,
    int n_eig,
    int n_src,
    int n_grp,
    long *edges,
    int isinv
);
