"""Utility functions of DDN
"""

import numpy as np


def standardize_data(data, scaler="std", zero_mean_already=False):
    """Standadize each column of the input data

    Parameters
    ----------
    data : ndarray
        Input data. Each column is a feature.
    scaler : str, optional
        Method to calculate the scale of each feature, by default "std"
    zero_mean_already : bool, optional
        The data is already with zero mean, by default False

    Returns
    -------
    standard_data : ndarray
        Standardized data
    """

    # sample standardization : z = (x - u) / s

    standard_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        # mean value
        u = np.mean(data[:, i]) if not zero_mean_already else 0

        if scaler == "std":
            # standard deviation
            s = np.std(data[:, i])
        elif scaler == "rms":
            # root-mean-square
            s = np.sqrt(np.mean(np.square(data[:, i])))
        else:
            s = 1

        standard_data[:, i] = (data[:, i] - u) / s

    return standard_data


def gen_mv(cov_mat, n_sample):
    """Generate zero mean multivariate data and standardize it.

    Parameters
    ----------
    cov_mat : ndarray
        Covariance matrix
    n_sample : int
        Sample size

    Returns
    -------
    ndarray
        Generated samples
    """
    x = np.random.multivariate_normal(np.zeros(len(cov_mat)), cov_mat, n_sample)
    # x = standardize_data(x, scaler="std")
    return x


def concatenate_data(controldata, casedata, method="diag"):
    """Several method to concatenate two arrays"""
    if method == "row":
        return np.concatenate((controldata, casedata), axis=0)
    elif method == "col":
        return np.concatenate((controldata, casedata), axis=1)
    elif method == "diag":
        return np.concatenate(
            (
                np.concatenate((controldata, casedata * 0), axis=0),
                np.concatenate((controldata * 0, casedata), axis=0),
            ),
            axis=1,
        )
    else:
        return []


def get_net_topo_from_mat(mat_prec, thr=1e-4):
    """Threshold the input matrxi to get adjacency matrix

    The input is also made symmetry.
    As we do not use self loop, the diagonal elements are always 0.

    Parameters
    ----------
    mat_prec : ndarray
        Input matrix. It could be a precision matrix, or a coefficient matrix.
    thr : float, optional
        Threshold to remove too small items in `mat_prec`, by default 1e-4

    Returns
    -------
    ndarray
        Adjacency matrix
    """
    N = len(mat_prec)
    x = np.copy(mat_prec)
    x[np.arange(N), np.arange(N)] = 0.0
    x = 1.0 * (np.abs(x) > thr)
    x = 1.0 * ((x + x.T) > 0)
    return x


def get_common_diff_net_topo(g_beta, thr=1e-4):
    """Calcualte common and differential network from the output of DDN

    g_beta[0] is the coefficient matrxi for condition 1
    g_beta[0] is the coefficient matrxi for condition 2
    """
    g1 = get_net_topo_from_mat(g_beta[0], thr=thr)
    g2 = get_net_topo_from_mat(g_beta[1], thr=thr)
    g_net_comm = 1.0 * ((g1 + g2) == 2)
    g_net_dif = 1.0 * (g1 != g2)
    return g_net_comm, g_net_dif


def ddn_obj_fun(y, X, lambda1, lambda2, n1, n2, beta):
    """Objective function for DDN, assuming two conditions having the same sample size"""
    p = X.shape[1] // 2
    beta1 = beta[:p]
    beta2 = beta[p:]
    d0 = y - X @ beta
    res = (
        np.sum(d0 * d0) / n1 / 2
        + np.sum(np.abs(beta)) * lambda1
        + np.sum(np.abs(beta1 - beta2)) * lambda2
    )
    return res
