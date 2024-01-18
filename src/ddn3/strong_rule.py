"""Functions that implements the strong rule strategy in DDN 3.0
"""

import numpy as np
from sklearn.linear_model import Lasso


def strong_rule(X, y, lambda1):
    """Find the variables to keep using the strong rule

    This rule eliminates a set of variables that will generally not be selected by Lasso.
    Let N be the sample size. Let P be the feature size.

    Parameters
    ----------
    X : array_like
        The predictors. Shape N by P-1.
    y : array_like
        The response variables. Shape N by 1.
    lambda1 : float
        DDN parameter lambda1.

    Returns
    -------
    idx : ndarray
        List of indices to keep

    """
    n = len(y)
    lambdas = np.abs(np.matmul(X.T, y)) / n
    lambdamax = max(lambdas)
    idx = np.array(
        [i for i in range(X.shape[1]) if lambdas[i] >= 2 * lambda1 - lambdamax]
    )
    # drop = X.shape[1] - len(idx)
    return idx


def lasso(y, X, lambda1, beta_in, tol=1e-6, use_strong_rule=True, use_warm=False):
    """Compute lasso with strong rule that eliminates variables

    Let N be the sample size. Let P be the feature size.

    Parameters
    ----------
    y : array-like
        The response variable. Size N by 1.
    X : array_like
        The predictor. Size N by P-1
    lambda1 : float
        DDN parameter lambda1.
    beta_in : array_like
        Initial lasso coefficient. Size P-1 by 1.
    tol : float
        Lasso tolerance.
    use_strong_rule : bool
        Apply strong rule or not. Strong rule will eliminate some predictors.
    use_warm : bool
        Apply warm start or not.

    Returns
    -------
    beta : ndarray
        The estimated lasso coefficients. Size P-1.

    """
    # select feature index
    n_pred = X.shape[1]
    idx = np.arange(n_pred)

    if use_strong_rule:
        idx = strong_rule(X, y, lambda1)
        if len(idx) == 0:
            return np.zeros(n_pred)
        X = X[:, idx]

    # calculate beta with sklearn lasso
    if use_warm:
        clf = Lasso(
            alpha=lambda1,
            # max_iter=1000,
            tol=tol,
            fit_intercept=False,
            warm_start=True,
        )
        clf.coef_ = beta_in
    else:
        clf = Lasso(
            alpha=lambda1,
            # max_iter=1000,
            tol=tol,
            fit_intercept=False,
            warm_start=False,
        )
        clf.coef_ = np.zeros((n_pred,))
    clf.fit(X, y)

    beta = np.zeros(n_pred)
    # beta = np.zeros((len(idx),))
    beta[idx] = clf.coef_

    return beta
