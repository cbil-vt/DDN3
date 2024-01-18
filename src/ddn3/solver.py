"""Wrapper functions for calling the BCD algorithms
"""

import numpy as np
from ddn3 import tools
from ddn3 import bcd, strong_rule


def run_org(
    g1_data,
    g2_data,
    node,
    lambda1,
    lambda2,
    beta1_in,
    beta2_in,
    threshold=1e-6,
    use_warm=False,
):
    """The wrapper that calls the DDN 2.0 BCD algorithm

    Denote P be the number features. N1 be the sample size for condition 1, and N2 for condition 2.

    Parameters
    ----------
    g1_data : array_like, shape N1 by P
        The data from condition 1
    g2_data : array_like, shape N2 by P
        The data from condition 2
    node : int
        Index of the current node that serve as the response variable.
    lambda1 : float
        DDN parameter lambda1.
    lambda2 : float
        DDN parameter lambda2.
    beta1_in : array_like, length P
        Initial beta for condition 1. If initialization is not needed, use an array of all zeros.
    beta2_in : array_like, length P
        Initial beta for condition 2. If initialization is not needed, use an array of all zeros.
    threshold : float
        Convergence threshold.
    use_warm : bool, optional
        Not used. To use warm start, simply input `beta1_in` and `beta2_in` from previous hyperparameters.

    Returns
    -------
    beta1 : ndarray, length P
        Estimated beta for `node` in condition 1.
    beta2 : ndarray, length P
        Estimated beta for `node` in condition 2.

    """
    beta_in = np.concatenate(
        (
            beta1_in[:node],
            beta1_in[node + 1 :],
            beta2_in[:node],
            beta2_in[node + 1 :],
        )
    )

    N_NODE = g1_data.shape[1]
    n1, n2 = g1_data.shape[0], g2_data.shape[0]

    y = tools.concatenate_data(g1_data[:, node], g2_data[:, node], method="row")
    node_fea = [i for i in range(N_NODE) if i != node]
    X = tools.concatenate_data(
        g1_data[:, node_fea], g2_data[:, node_fea], method="diag"
    )
    beta, r, betaerr = bcd.bcd_org(beta_in, y, X, lambda1, lambda2, n1, n2, threshold)

    # reindex the features
    beta1 = list(beta[0:node]) + [0] + list(beta[node : N_NODE - 1])
    beta1 = np.array(beta1)
    beta2 = (
        list(beta[N_NODE - 1 : node + N_NODE - 1])
        + [0]
        + list(beta[node + N_NODE - 1 : 2 * N_NODE - 2])
    )
    beta2 = np.array(beta2)
    return beta1, beta2


def run_resi(
    g1_data,
    g2_data,
    node,
    lambda1,
    lambda2,
    beta1_in,
    beta2_in,
    threshold,
    use_warm=False,
):
    """The wrapper that calls the DDN 3.0 residual update algorithm

    If warm start is used, the residual signal is updated accordingly to previously estimated beta.
    This takes some extra time, and may make warm start less appealing.

    Denote P be the number features. N1 be the sample size for condition 1, and N2 for condition 2.

    Parameters
    ----------
    g1_data : array_like, shape N1 by P
        The data from condition 1
    g2_data : array_like, shape N2 by P
        The data from condition 2
    node : int
        Index of the current node that serve as the response variable.
    lambda1 : float
        DDN parameter lambda1.
    lambda2 : float
        DDN parameter lambda2.
    beta1_in : array_like, length P
        Initial beta for condition 1. If initialization is not needed, use an array of all zeros.
    beta2_in : array_like, length P
        Initial beta for condition 2. If initialization is not needed, use an array of all zeros.
    threshold : float
        Convergence threshold.
    use_warm : bool, optional
        If true, use warm start.

    Returns
    -------
    beta1 : ndarray, length P
        Estimated beta for `node` in condition 1.
    beta2 : ndarray, length P
        Estimated beta for `node` in condition 2.

    """
    beta_in = np.concatenate((beta1_in, beta2_in))
    if use_warm:
        beta1_in_x = np.copy(beta1_in)
        beta2_in_x = np.copy(beta2_in)
        beta1_in_x[node] = 0
        beta2_in_x[node] = 0
        y1_resi = g1_data[:, node] - np.dot(g1_data, beta1_in_x)
        y2_resi = g2_data[:, node] - np.dot(g2_data, beta2_in_x)
    else:
        y1_resi = np.copy(g1_data[:, node])
        y2_resi = np.copy(g2_data[:, node])

    N_NODE = g1_data.shape[1]

    # beta = bcd_residual(g1_data, g2_data, node, lambda1, lambda2, threshold)
    beta, r, betaerr = bcd.bcd_residual(
        beta_in,
        g1_data,
        g2_data,
        y1_resi,
        y2_resi,
        node,
        lambda1,
        lambda2,
        threshold,
    )
    # n_iter.append(r)
    beta1 = np.array(beta[:N_NODE])
    beta2 = np.array(beta[N_NODE:])

    return beta1, beta2


def run_corr(
    corr_matrix_1,
    corr_matrix_2,
    node,
    lambda1,
    lambda2,
    beta1_in,
    beta2_in,
    threshold,
    use_warm=False,
):
    """The wrapper that calls the DDN 3.0 correlation matrix update algorithm

    Parameters
    ----------
    corr_matrix_1 : ndarray
        Input correlation matrix for condition 1.
    corr_matrix_2 : ndarray
        Input correlation matrix for condition 2.
    node : int
        Index of the current node that serve as the response variable.
    lambda1 : float
        DDN parameter lambda1.
    lambda2 : float
        DDN parameter lambda2.
    beta1_in : array_like, length P
        Initial beta for condition 1. If initialization is not needed, use an array of all zeros.
    beta2_in : array_like, length P
        Initial beta for condition 2. If initialization is not needed, use an array of all zeros.
    threshold : float
        Convergence threshold.
    use_warm : bool, optional
        If true, use warm start.

    Returns
    -------
    beta1 : ndarray, length P
        Estimated beta for `node` in condition 1.
    beta2 : ndarray, length P
        Estimated beta for `node` in condition 2.
    """
    beta_in = np.concatenate((beta1_in, beta2_in))

    beta, r, betaerr = bcd.bcd_corr(
        beta_in,
        node,
        lambda1,
        lambda2,
        corr_matrix_1,
        corr_matrix_2,
        threshold=threshold,
        max_iter=100000,
    )

    n_node = len(beta1_in)
    beta1 = np.array(beta[:n_node])
    beta2 = np.array(beta[n_node:])

    return beta1, beta2


def run_strongrule(
    g1_data,
    g2_data,
    node,
    lambda1,
    lambda2,
    beta1_in,
    beta2_in,
    threshold,
    use_warm=False,
):
    """The wrapper that calls the DDN 3.0 strong rule strategy.

    In this algorithm, `lambda2` is kept as zero. Then DDN becomes two separate lasso problem.
    The strong rule can then be utilized to eliminate part of the predictor variables for each `lambda1`.
    The lasso problem is also solved by the `lasso` function of scikit-learn.

    Warm start is generally preferred when using this strategy.

    Note that empirically this strategy is not fast.

    Denote P be the number features. N1 be the sample size for condition 1, and N2 for condition 2.

    Parameters
    ----------
    g1_data : array_like, shape N1 by P
        The data from condition 1
    g2_data : array_like, shape N2 by P
        The data from condition 2
    node : int
        Index of the current node that serve as the response variable.
    lambda1 : float
        DDN parameter lambda1.
    lambda2 : float
        Not used. Must be 0.
    beta1_in : array_like, length P
        Initial beta for condition 1. If initialization is not needed, use an array of all zeros.
    beta2_in : array_like, length P
        Initial beta for condition 2. If initialization is not needed, use an array of all zeros.
    threshold : float
        Convergence threshold.
    use_warm : bool, optional
        Use warm start or not.

    Returns
    -------
    beta1 : ndarray, length P
        Estimated beta for `node` in condition 1.
    beta2 : ndarray, length P
        Estimated beta for `node` in condition 2.

    """
    beta1_in_x = (np.concatenate([beta1_in[:node], beta1_in[node + 1 :]]),)
    beta2_in_x = (np.concatenate([beta2_in[:node], beta2_in[node + 1 :]]),)
    y1 = g1_data[:, node]
    y2 = g2_data[:, node]

    N_NODE = g1_data.shape[1]

    # choose other genes as feature
    idx = [i for i in range(N_NODE) if i != node]
    X1 = g1_data[:, idx]
    X2 = g2_data[:, idx]

    # perform bcd algorithm
    beta1 = strong_rule.lasso(
        y1,
        X1,
        lambda1,
        beta1_in_x,
        tol=threshold,
        use_strong_rule=True,
        use_warm=use_warm,
    )
    beta2 = strong_rule.lasso(
        y2,
        X2,
        lambda1,
        beta2_in_x,
        tol=threshold,
        use_strong_rule=True,
        use_warm=use_warm,
    )

    # reindex the features
    beta1 = list(beta1[0:node]) + [0] + list(beta1[node:])
    beta1 = np.array(beta1)
    beta2 = list(beta2[0:node]) + [0] + list(beta2[node:])
    beta2 = np.array(beta2)

    return beta1, beta2
