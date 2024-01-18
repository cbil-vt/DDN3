"""The convenience wrappers for running DDN
"""

from ddn3 import ddn, tools_export
from ddn3.simulation import simple_data


def ddn_pipeline(dat1, dat2, gene_names, lambda1=0.3, lambda2=0.1):
    """A wrapper around DDN

    Reads two data sets and a list of genes, then runs DDN and reports common and differential networks.

    Parameters
    ----------
    dat1 : array_like
        The first data for DDN
    dat2 : array_like
        The second data for DDN
    gene_names : list of str
        List of gene names for reporting and visualization
    lambda1 : float, optional
        The parameter controlling the overall sparsity. Usually should be between 0 and 1
    lambda2 : float, optional
        The parameter controlling the differences between two networks. Usually should be between 0 and 0.3

    Returns
    -------
    comm_edge : pandas.DataFrame
        The common edges between the two networks
    diff_edge : pandas.DataFrame
        The differential edges between the two networks
    node_non_isolated : list of str
        List of nodes that have at least one neighboring nodes

    """
    omega1, omega2 = ddn.ddn(dat1, dat2, lambda1, lambda2)
    (
        comm_edge,
        _,
        _,
        diff_edge,
        node_non_isolated,
    ) = tools_export.get_diff_comm_net_for_plot(omega1, omega2, gene_names)
    return comm_edge, diff_edge, node_non_isolated
