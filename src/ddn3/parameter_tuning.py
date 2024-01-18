"""The module for tuning hyperparameters in DDN

Find strategies are implemented.

- `cv_joint`: grid search CV for lambda1 and lambda2. This is time-consuming for larger data.
- `cv_sequential`: CV for lambda1 first, then use the determined lambda1 to do CV on lambda2.
- `cv_bai`:  CV for lambda1 first, then use the method in [1] to directly calculate lambda2.
- `mb_cv`: use theorem 3 in [2] to directly calculate lambda1, and use CV to get lambda2.
- `mb_bai`: use theorem 3 in [2] to directly calculate lambda1, and the method in [1] to get lambda2.

If the data set is not too large, `cv_joint` is a better choice.
If the data is larger, it is better to use the `cv_sequential` option.
If the data is really large, consider `cv_bai`.

A bette approach is to manually choose a set of lambda1 and select the one that leads to a reasonable network.
By utilizing the prior knowledge, it is more likely to obtain the network that is usable.

[1] "Learning structural changes of Gaussian graphical models in controlled experiments." UAI (2012).
[2] "High-dimensional graphs and variable selection with the lasso." Ann. Statist. (2006): 1436-1462.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from ddn3 import ddn
from ddn3 import tools


class DDNParameterSearch:
    def __init__(
        self,
        dat1,
        dat2,
        lambda1_list=np.arange(0.05, 1.05, 0.05),
        lambda2_list=np.arange(0.025, 0.525, 0.025),
        n_cv=5,
        ratio_validation=0.2,
        alpha1=0.05,
        alpha2=0.01,
    ) -> None:
        """Initialize the parameter search.

        A number of parameters are set here. However, some of them are only used for certain parameter tuning methods.

        Parameters
        ----------
        dat1 : array_like
            The first data.
        dat2 : array_like
            The second data.
        lambda1_list : array_like
            A list of lambda1 values to search
        lambda2_list : array_like
            A list of lambda2 values to search
        n_cv : int
            The number of repeats. As we use repeated K-fold CV, `n_cv` can be as large as you like.
            Usually, a value of 10 or 20 is sufficient.
        ratio_validation : float
            The ratio of data that we used for validation. The remaining data is for training.
            As we use repeated CV, each time the training data is different.
        alpha1 : float
            The alpha used in theorem 3 of MB algorithm [2].
        alpha2 : float
            The alpha used in the method in [1]
        """
        # parameters
        self.dat1 = dat1
        self.dat2 = dat2
        self.l1_lst = lambda1_list
        self.l2_lst = lambda2_list
        self.n_cv = n_cv
        self.ratio_val = ratio_validation
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        # derived and outputs
        self.n = int((dat1.shape[0] + dat2.shape[0]) / 2)
        self.p = dat1.shape[1]

    def fit(self, method="cv_sequential"):
        """Estimate lambda1 and lambda2.

        The validation error is defined as the ratio of the signal in the validation set that cannot be
         explained by the network estimated in the training set.

        Parameters
        ----------
        method : str
            The method used for estimation.

        Returns
        -------
        out : tuple
            The validation error, estimated lambda1, and lambda2

        """
        if method == "cv_joint":
            out = self.run_cv_joint()
        elif method == "cv_sequential":
            out = self.run_cv_sequential()
        elif method == "cv_bai":
            out = self.run_cv_bai()
        elif method == "mb_cv":
            out = self.run_mb_cv()
        elif method == "mb_bai":
            out = self.run_mb_bai()
        else:
            out = self.run_cv_bai()
        return out

    def run_cv_joint(self):
        """Estimation lambda1 and lambda2 with grid search CV"""
        # can be slow
        val_err, _, _ = cv_two_lambda(
            self.dat1,
            self.dat2,
            n_cv=self.n_cv,
            ratio_val=self.ratio_val,
            lambda1_lst=self.l1_lst,
            lambda2_lst=self.l2_lst,
        )
        l1_est, l2_est = get_lambdas_one_se_2d(val_err, self.l1_lst, self.l2_lst)
        return val_err, l1_est, l2_est

    def run_cv_sequential(self):
        """Use CV to estimate lambda1, then use CV for lambda2"""
        val_err1, _, _ = cv_two_lambda(
            self.dat1,
            self.dat2,
            lambda1_lst=self.l1_lst,
            lambda2_lst=[0.0],
            n_cv=self.n_cv,
            ratio_val=self.ratio_val,
        )
        val_err1 = np.squeeze(val_err1)
        l1_est = get_lambda_one_se_1d(val_err1, self.l1_lst)

        val_err2, _, _ = cv_two_lambda(
            self.dat1,
            self.dat2,
            lambda1_lst=[l1_est],
            lambda2_lst=self.l2_lst,
            n_cv=self.n_cv,
            ratio_val=self.ratio_val,
        )
        val_err2 = np.squeeze(val_err2)
        l2_est = get_lambda_one_se_1d(val_err2, self.l2_lst)

        return [val_err1, val_err2], l1_est, l2_est

    def run_cv_bai(self):
        """Use CV for lambda1, use the method in [1] for lambda2"""
        val_err, _, _ = cv_two_lambda(
            self.dat1,
            self.dat2,
            lambda1_lst=self.l1_lst,
            lambda2_lst=[0.0],
            n_cv=self.n_cv,
            ratio_val=self.ratio_val,
        )
        val_err = np.squeeze(val_err)
        l1_est = get_lambda_one_se_1d(val_err, self.l1_lst)
        l2_est = get_lambda2_bai(self.dat1, self.dat2, alpha=self.alpha2)

        return val_err, l1_est, l2_est

    def run_mb_cv(self):
        """Use the theorem 3 in [2] for lambda1, and use CV for lambda2"""
        # mb not reliable with small sample size
        l1_est = get_lambda1_mb(self.alpha1, self.n, self.p)
        val_err, _, _ = cv_two_lambda(
            self.dat1,
            self.dat2,
            lambda1_lst=[l1_est],
            lambda2_lst=self.l2_lst,
            n_cv=self.n_cv,
            ratio_val=self.ratio_val,
        )
        val_err = np.squeeze(val_err)
        l2_est = get_lambda_one_se_1d(val_err, self.l2_lst)

        return val_err, l1_est, l2_est

    def run_mb_bai(self):
        """Use theorem 3 in [2] for lambda1, and use method in [1] for lambda2"""
        # mb not reliable with small sample size
        l1_est = get_lambda1_mb(self.alpha1, self.n, self.p, mthd=0)
        l2_est = get_lambda2_bai(self.dat1, self.dat2, alpha=self.alpha2)
        return [], l1_est, l2_est


def get_lambda1_mb(alpha, n, p, mthd=0):
    """Estimate lambda1 using theorem 3 in [2]

    Parameters
    ----------
    alpha : float
        The parameter controlling false positives
    n : int
        The number of samples
    p : int
        The number of features
    mthd : int
        Use 0 for methods in [2], with a correction term of 2.
        Use 1 for methods used in https://pubmed.ncbi.nlm.nih.gov/25273109/

    Returns
    -------
    lmb1 : float
        The estimated lambda1

    """
    lmb1 = 0.5
    if mthd == 0:
        # The MB paper do not have 1/2 factor for data term
        # To make things consistent, we further divide by 2
        lmb1 = 2 / np.sqrt(n) * norm.ppf(1 - alpha / (2 * p * n * n)) / 2
        # lmb1 = 2 / np.sqrt(n) * norm.ppf(1 - alpha / (2 * p * n * n))
    if mthd == 1:
        # NOTE: this is from KDDN Java code, which is not the same as MB paper
        lmb1 = 2 / n * norm.ppf(1 - alpha / (2 * p * n * n))
    return lmb1


def get_lambda2_bai(
    x1,
    x2,
    alpha=0.01,
):
    """Estimate lambda2 using the method in [1]

    Parameters
    ----------
    x1 : array_like
        The data for condition 1
    x2 : array_like
        The data for condition 2
    alpha : float
        The parameter controlling false positives

    Returns
    -------
    lmb2 : float
        Estimated labmda2

    """
    x1 = tools.standardize_data(x1)
    x2 = tools.standardize_data(x2)

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    N = (n1 + n2) / 2
    p = x1.shape[1]

    s = norm.ppf(1 - alpha / 2) * np.sqrt(2 / (N - 3))

    # pd = (x1.T @ x1) * (x2.T @ x2)
    pd = (x1.T @ x1 / n1) * (x2.T @ x2 / n2)
    pd[np.arange(p), np.arange(p)] = 0
    rho1rho2 = np.sum(pd) / p / (p - 1)

    lmb2 = (np.exp(2 * s) - 1) / (np.exp(2 * s) + 1) / 2 * (1 - rho1rho2)
    print("Avg rho is ", rho1rho2)
    print("lambda2 is ", lmb2)

    return lmb2


def get_lambda_one_se_1d(val_err, lambda_lst):
    """Choose a lambda from a list of lambda values using the one standard error rule

    let K be the number of CV repeats, L the number of lambda values.

    Parameters
    ----------
    val_err : array_like
        Validation errors corresponding to the list of lambda. Shape K by L.
    lambda_lst : array_like
        The lambda values. Shape L.

    Returns
    -------
    float
        Chosen lambda value

    """
    val_err_mean = np.mean(val_err, axis=0)
    val_err_std = np.std(val_err, axis=0)
    n_cv = val_err.shape[0]

    idx = np.argmin(val_err_mean)
    val_err_mean[:idx] = -10000

    cut_thr = val_err_mean[idx] + val_err_std[idx] / np.sqrt(n_cv)  # standard error
    if np.max(val_err_mean) < cut_thr:
        print("One SE rule failed.")
        return lambda_lst[-1]
    else:
        # idx_thr = np.max(np.where(val_err_mean <= cut_thr)[0])
        idx_thr = np.min(np.where(val_err_mean >= cut_thr)[0])
        return lambda_lst[idx_thr]


def get_lambdas_one_se_2d(val_err, lambda1_lst, lambda2_lst):
    """Choose a lambda from a list of lambda values using the one standard error rule

    After finding the minimum value, we find all the combinations of lambda1 and lambda2 values that are at least one
    standard error larger than this minimum value.
    From these combinations, we choose the one whose lambda1 and lambda2 values are closest to those of the minimum.

    let K be the number of CV repeats, L1 the number of lambda1 values, L2 the number of lambda2 values.

    Parameters
    ----------
    val_err : array_like
        Validation errors corresponding to the list of lambda. Shape K by L1 by L2.
    lambda1_lst : array_like
        The lambda1 values, shape L1.
    lambda1_lst : array_like
        The lambda2 values, shape L2.

    Returns
    -------
    float
        Chosen lambda value

    """
    gap1 = np.abs(lambda1_lst[-1] - lambda1_lst[0]) / len(lambda1_lst)
    gap2 = np.abs(lambda2_lst[-1] - lambda2_lst[0]) / len(lambda2_lst)
    lmb1_scale = gap1 / gap2

    val_err_mean = np.mean(val_err, axis=0)
    val_err_std = np.std(val_err, axis=0)
    n_cv = val_err.shape[0]

    idx1, idx2 = np.unravel_index(val_err_mean.argmin(), val_err_mean.shape)
    print(idx1, idx2)
    print("l1_org, l2_org ", lambda1_lst[idx1], lambda2_lst[idx2])
    msk = np.zeros_like(val_err_mean)
    msk[idx1:, idx2:] = 1.0
    se = val_err_std[idx1, idx2] / np.sqrt(n_cv)
    z = (val_err_mean - np.min(val_err_mean)) / se
    z = z * msk

    if np.max(z) == 0:
        print("One SE rule failed.")
        return np.max(lambda1_lst), np.max(lambda2_lst)
    else:
        m1, m2 = val_err_mean.shape
        cord1, cord2 = np.meshgrid(np.arange(m1), np.arange(m2), indexing="ij")
        d = (cord1 - idx1) ** 2 * lmb1_scale + (cord2 - idx2) ** 2
        d[z < 1] = 100000
        idx1a, idx2a = np.unravel_index(d.argmin(), d.shape)
        print(idx1a, idx2a)
        print("l1, l2 ", lambda1_lst[idx1a], lambda2_lst[idx2a])

        return lambda1_lst[idx1a], lambda2_lst[idx2a]


def calculate_regression(data, topo_est):
    """Linear regression

    For each variable, use all its neighbors as  predictors and find the regression coefficients.

    This is an example of regression operation

    >>> x = np.array([[-1,-1,1,1.0], [1,1,-1,-1]]).T
    >>> y = np.array([1,1,-1,-1.0])
    >>> out = np.linalg.lstsq(x, y, rcond=None)
    >>> out[0]

    Parameters
    ----------
    data : array_like
        All data
    topo_est : array_like
        Estimated adjacency matrix

    Returns
    -------

    """
    n_fea = data.shape[1]
    g_asso = np.eye(n_fea, dtype=np.double)
    for i in range(n_fea):
        pred_idx = np.where(topo_est[i] > 0)[0]
        if len(pred_idx) == 0:
            continue
        y = data[:, i]
        x = data[:, pred_idx]
        out = np.linalg.lstsq(x, y, rcond=None)
        g_asso[i, pred_idx] = out[0]
    return g_asso


def cv_two_lambda(
    dat1,
    dat2,
    n_cv=5,
    ratio_val=0.2,
    lambda1_lst=np.arange(0.05, 1.05, 0.05),
    lambda2_lst=np.arange(0.025, 0.525, 0.025),
):
    """Cross validation by grid search lambda1 and lambda2

    To estimate the validation error, we estimate the coefficient of each node on the training set based on the
    estimated network topology. Then for each node in the validation set, we try to use its neighbors to explain the
    signal in that node. The portion of unexplained signal in all nodes is defined as the validation error.

    Let K be the number of CV repeats, L1 the number of lambda1 values, L2 the number of lambda2 values.

    Parameters
    ----------
    dat1 : array_like
        Data for condition 1
    dat2 : array_like
        Data for condition 1
    n_cv : int
        Number of repeats. Can be as large as you like, as we re-sample each time.
    ratio_val : float
        Ratio of data for validation. The remaining is used for training.
    lambda1_lst : array_like
        Values of lambda1 for searching
    lambda2_lst : array_like
        Values of lambda2 for searching

    Returns
    -------
    val_err : array_like
        The validation error for each lambda1 and lambda2 combination. Shape K by L1 by L2
    lambda1_lst : array_like
    lambda2_lst : array_like

    """
    val_err = np.zeros((n_cv, len(lambda1_lst), len(lambda2_lst)))
    n_node = dat1.shape[1]
    n1 = dat1.shape[0]
    n2 = dat2.shape[0]
    n1_val = int(n1 * ratio_val)
    n1_train = n1 - n1_val
    n2_val = int(n2 * ratio_val)
    n2_train = n2 - n2_val

    mthd = "resi"
    # if len(lambda2_lst) == 1 and lambda2_lst[0] == 0:
    #     mthd = 'strongrule'
    print(mthd)

    for n in range(n_cv):
        print("Repeat ============", n)
        msk1 = np.zeros(n1)
        msk1[np.random.choice(n1, n1_train, replace=False)] = 1
        msk2 = np.zeros(n2)
        msk2[np.random.choice(n2, n2_train, replace=False)] = 1
        g1_train = tools.standardize_data(dat1[msk1 > 0])
        g1_val = tools.standardize_data(dat1[msk1 == 0])
        g2_train = tools.standardize_data(dat2[msk2 > 0])
        g2_val = tools.standardize_data(dat2[msk2 == 0])

        for i, lambda1 in enumerate(lambda1_lst):
            # print(n, i)
            for j, lambda2 in enumerate(lambda2_lst):
                g_beta_est = ddn.ddn(
                    g1_train,
                    g2_train,
                    lambda1=lambda1,
                    lambda2=lambda2,
                    mthd=mthd,
                )
                g1_net_est = tools.get_net_topo_from_mat(g_beta_est[0])
                g2_net_est = tools.get_net_topo_from_mat(g_beta_est[1])
                g1_coef = calculate_regression(g1_train, g1_net_est)
                g1_coef[np.arange(n_node), np.arange(n_node)] = 0
                g2_coef = calculate_regression(g2_train, g2_net_est)
                g2_coef[np.arange(n_node), np.arange(n_node)] = 0
                rec_ratio1 = np.linalg.norm(
                    g1_val @ g1_coef.T - g1_val
                ) / np.linalg.norm(g1_val)
                rec_ratio2 = np.linalg.norm(
                    g2_val @ g2_coef.T - g2_val
                ) / np.linalg.norm(g2_val)
                val_err[n, i, j] = (rec_ratio1 + rec_ratio2) / 2

    return val_err, lambda1_lst, lambda2_lst


def plot_error_1d(val_err, lambda_lst=(), ymin=None, ymax=None):
    """Plot the curve of cross validation with one lambda"""
    val_err_mean = np.mean(val_err, axis=0)
    val_err_std = np.std(val_err, axis=0)
    n_cv = val_err.shape[0]
    if not ymin:
        ymin = max(np.min(val_err_mean), 0.0)
    if not ymax:
        ymax = min(np.max(val_err_mean), 1.0)
    gap1 = (ymax - ymin) / 10
    fig, ax = plt.subplots()
    ax.errorbar(
        lambda_lst,
        val_err_mean,
        yerr=val_err_std / np.sqrt(n_cv),
        ecolor="red",
        elinewidth=0.5,
    )
    ax.set_ylim([ymin - gap1, ymax + gap1])


def plot_error_2d(val_err, cmin=None, cmax=None):
    """Plot the image of cross validation with both lambda1 and lambda2"""
    val_err_mean = np.mean(val_err, axis=0)
    val_err_std = np.std(val_err, axis=0)
    n_cv = val_err.shape[0]
    if not cmin:
        cmin = np.min(val_err_mean)
    if not cmax:
        cmax = np.max(val_err_mean)
    fig, ax = plt.subplots()
    pos = ax.imshow(val_err_mean, origin="lower", vmin=cmin, vmax=cmax)
    color_bar = fig.colorbar(pos, ax=ax)
    ax.set_xlabel("$\lambda_2$")
    ax.set_ylabel("$\lambda_1$")
