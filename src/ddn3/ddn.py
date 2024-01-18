"""The DDN main functions

There are two top level DDN functions: the `ddn` and the `ddn_parallel`.
The former one is in serial, which is easier to debug, while the latter one is in parallel.
For smaller data sets (e.g., less than 500 nodes), the serial version is fast enough.
The two functions should have the same functionality.

Each function allow using several different methods.

- `org`: the method in DDN 2.0. This is slow for larger data sets.
- `resi`: the method in DDN 3.0 using residual update strategy. This is suitable for larger feature number.
- `corr`: the method in DDN 3.0 using correlation matrix update strategy. This is suitable for larger sample number.
- `strongrule`: using the strong rule to accelerate the condition when lambda=0.

We recommend using `resi` in general case. In case you have much more samples than features, consider `corr`.

The choice of two hyperparameters lambda1 and lambda2 is critical.
We recommend running DDN with a range of parameters, and using prior knowledge to select the suitable.
Alternatively, users may refer the parameter tuning tutorial for other methods of choosing the parameters.

These functions also support using warm start.
To use this function, input the coefficient matrix from the previous call of DDN.
However, we do not observe very significant speed up, and are not generally recommended.

"""

import numpy as np
import joblib
from joblib import Parallel, delayed
from ddn3 import tools, solver


def ddn_parallel(
    g1_data,
    g2_data,
    lambda1=0.30,
    lambda2=0.10,
    threshold=1e-6,
    mthd="resi",
    std_est="std",
    g_rec_in=(),
    n_process=1,
):
    """Run DDN in parallel.

    Denote P be the number features. N1 be the sample size for condition 1, and N2 for condition 2.

    Parameters
    ----------
    g1_data : array_like, shape N1 by P
        The data from condition 1
    g2_data : array_like, shape N2 by P
        The data from condition 2
    lambda1 : float
        DDN parameter lambda1.
    lambda2 : float
        Not used. Must be 0.
    threshold : float
        Convergence threshold.
    mthd : str
        The DDN solver to use.
    std_est : str
        The standardization method. The default value is fine in most cases.
    g_rec_in : ndarray
        The input coefficient matrix. If warm start is not needed, do not change it.
    n_process : int
        Number of cores to use. Do not exceed the number of cores in your computer.
        If set to 1, no parallelization is used.

    Returns
    -------
    g_res : ndarray
        The estimated coefficient array of shape (2, P, P).
        g_res[0] is for the first condition, and g_res[1] for the second condition.

    """
    if n_process <= 1:
        n_process = int(joblib.cpu_count() / 2)

    n_node = g1_data.shape[1]
    n1 = g1_data.shape[0]
    n2 = g2_data.shape[0]
    g1_data = tools.standardize_data(g1_data, scaler=std_est)
    g2_data = tools.standardize_data(g2_data, scaler=std_est)

    if len(g_rec_in) == 0:
        g_rec_in = np.zeros((2, n_node, n_node))
        use_warm = False
    else:
        use_warm = True

    if mthd == "corr":
        corr_matrix_1 = g1_data.T @ g1_data / n1
        corr_matrix_2 = g2_data.T @ g2_data / n2
    else:
        corr_matrix_1 = []
        corr_matrix_2 = []

    if mthd == "org":
        out = Parallel(n_jobs=n_process)(
            delayed(solver.run_org)(
                g1_data,
                g2_data,
                node,
                lambda1,
                lambda2,
                beta1_in=g_rec_in[0][node],
                beta2_in=g_rec_in[1][node],
                threshold=threshold,
                use_warm=use_warm,
            )
            for node in range(n_node)
        )
    elif mthd == "resi":
        out = Parallel(n_jobs=n_process)(
            delayed(solver.run_resi)(
                g1_data,
                g2_data,
                node,
                lambda1,
                lambda2,
                beta1_in=g_rec_in[0][node],
                beta2_in=g_rec_in[1][node],
                threshold=threshold,
                use_warm=use_warm,
            )
            for node in range(n_node)
        )
    elif mthd == "corr":
        out = Parallel(n_jobs=n_process)(
            delayed(solver.run_corr)(
                corr_matrix_1,
                corr_matrix_2,
                node,
                lambda1,
                lambda2,
                beta1_in=g_rec_in[0][node],
                beta2_in=g_rec_in[1][node],
                threshold=threshold,
                use_warm=use_warm,
            )
            for node in range(n_node)
        )
    elif mthd == "strongrule":
        out = Parallel(n_jobs=n_process)(
            delayed(solver.run_strongrule)(
                g1_data,
                g2_data,
                node,
                lambda1,
                lambda2,
                beta1_in=g_rec_in[0][node],
                beta2_in=g_rec_in[1][node],
                threshold=threshold,
                use_warm=use_warm,
            )
            for node in range(n_node)
        )
    else:
        raise ("Method not implemented")

    g_rec = np.zeros((2, n_node, n_node))
    for node in range(n_node):
        g_rec[0, node, :] = out[node][0]
        g_rec[1, node, :] = out[node][1]

    return g_rec


def ddn(
    g1_data,
    g2_data,
    lambda1=0.30,
    lambda2=0.10,
    threshold=1e-6,
    mthd="resi",
    std_est="std",
    g_rec_in=(),
    standard=True,
):
    """Run DDN.

    Denote P be the number features. N1 be the sample size for condition 1, and N2 for condition 2.

    Parameters
    ----------
    g1_data : array_like, shape N1 by P
        The data from condition 1
    g2_data : array_like, shape N2 by P
        The data from condition 2
    lambda1 : float
        DDN parameter lambda1.
    lambda2 : float
        Not used. Must be 0.
    threshold : float
        Convergence threshold.
    mthd : str
        The DDN solver to use.
    std_est : str
        The standardization method. The default value is fine in most cases.
    g_rec_in : ndarray
        The input coefficient matrix. If warm start is not needed, do not change it.

    Returns
    -------
    g_res : ndarray
        The estimated coefficient array of shape (2, P, P).
        g_res[0] is for the first condition, and g_res[1] for the second condition.

    """
    n_node = g1_data.shape[1]
    n1 = g1_data.shape[0]
    n2 = g2_data.shape[0]
    if standard:
        g1_data = tools.standardize_data(g1_data, scaler=std_est)
        g2_data = tools.standardize_data(g2_data, scaler=std_est)

    if len(g_rec_in) == 0:
        g_rec_in = np.zeros((2, n_node, n_node))
        use_warm = False
    else:
        use_warm = True

    if mthd == "corr":
        corr_matrix_1 = g1_data.T @ g1_data / n1
        corr_matrix_2 = g2_data.T @ g2_data / n2
    else:
        corr_matrix_1 = []
        corr_matrix_2 = []

    g_rec = np.zeros((2, n_node, n_node))
    for node in range(n_node):
        beta1_in = g_rec_in[0][node]
        beta2_in = g_rec_in[1][node]

        if mthd == "org":
            beta1, beta2 = solver.run_org(
                g1_data,
                g2_data,
                node,
                lambda1,
                lambda2,
                beta1_in,
                beta2_in,
                threshold,
                use_warm,
            )
        elif mthd == "resi":
            beta1, beta2 = solver.run_resi(
                g1_data,
                g2_data,
                node,
                lambda1,
                lambda2,
                beta1_in,
                beta2_in,
                threshold,
                use_warm,
            )
        elif mthd == "corr":
            beta1, beta2 = solver.run_corr(
                corr_matrix_1,
                corr_matrix_2,
                node,
                lambda1,
                lambda2,
                beta1_in,
                beta2_in,
                threshold,
                use_warm,
            )
        elif mthd == "strongrule":
            beta1, beta2 = solver.run_strongrule(
                g1_data,
                g2_data,
                node,
                lambda1,
                lambda2,
                beta1_in,
                beta2_in,
                threshold,
                use_warm,
            )
        else:
            print("Method not implemented")
            break

        g_rec[0, node, :] = beta1
        g_rec[1, node, :] = beta2

    return g_rec
