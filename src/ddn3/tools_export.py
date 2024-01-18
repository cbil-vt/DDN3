"""Tools for extrating information from DDN outputs
"""

import numpy as np
import pandas as pd
from ddn3 import tools


def get_diff_comm_net_for_plot(omega1, omega2, gene_names):
    """Generate common and differential networks from DDN outputs

    Parameters
    ----------
    omega1 : ndarray
        DDN estimated coefficient matrix for condition 1
    omega2 : ndarray
        DDN estimated coefficient matrix for condition 2
    gene_names : list of str
        Gene names

    Returns
    -------
    comm_edge : pandas.DataFrame
        Data frame containing information of edges in common networks.
        The first two columns means the two genes in each edge.
        The fourth column gives the weights of that edge. 
        The weights is the absolute value of the DDN coefficient.
    dif1_edge : pandas.DataFrame
        Data frame containing information of differential edges from condition 1.
        The contents are similart to `comm_edge`.
    dif2_edge : pandas.DataFrame
        Data frame containing information of differential edges from condition 2.
        The contents are similart to `comm_edge`.
    diff_edge : pandas.DataFrame
        Data frame containing information of edges in the differential networks.
        The contents are similart to `comm_edge`, except that the third column 
        labels whether this edge come from condition 1 or 2.
    node_non_isolated : list of str
        Node names that are involved in edges in either common network or differential networks.
    """
    
    # make the coefficient matrices symmetric
    omega1 = (omega1 + omega1.T) / 2
    omega2 = (omega2 + omega2.T) / 2
    omega_comm = (omega1 + omega2) / 2

    # apply a small threshold to obtain adjacency matrices
    g1 = tools.get_net_topo_from_mat(omega1)
    g2 = tools.get_net_topo_from_mat(omega2)
    gene_names = np.array(gene_names)

    # find common and differential networks
    comm_adj_mat = (g1 + g2) == 2
    dif1_adj_mat = g1 != comm_adj_mat
    dif2_adj_mat = g2 != comm_adj_mat

    # Get edges in format `gene1`, `gene2`, `condition`, `beta`
    comm_edge = _get_edge_list(comm_adj_mat, omega_comm, gene_names, group_idx=0)
    dif1_edge = _get_edge_list(dif1_adj_mat, omega1, gene_names, group_idx=1)
    dif2_edge = _get_edge_list(dif2_adj_mat, omega2, gene_names, group_idx=2)

    # find nodes that are involved in at least one edge
    node_degree_nz = (np.sum(g1, axis=0) > 0) + (np.sum(g2, axis=0) > 0)
    node_idx_non_isolated = np.sort(np.where(node_degree_nz > 0)[0])
    node_non_isolated = gene_names[node_idx_non_isolated]
    diff_edge = pd.concat((dif1_edge, dif2_edge))

    return comm_edge, dif1_edge, dif2_edge, diff_edge, node_non_isolated


def _get_edge_list(conn_mat, beta_mat, gene_names, group_idx=0):
    """Create a pandas data frame reporting the edges

    Parameters
    ----------
    conn_mat : ndarray
        Adjacency matrix for a graph
    beta_mat : ndarray
        The coefficient matrix of a graph
    gene_names : list of str
        Gane names
    group_idx : int, optional
        The group labeling, by default 0
        0 for common edges, 1 for condition 1, 2 for condition 2

    Returns
    -------
    pandas.DataFrame
        Data frame containing information of edges.
        The first two columns give names of genes for each edge.
        The third column is for the group label.
        The fourth column is the absolute value of the coefficient.
    """
    conn_mat1 = np.tril(conn_mat, -1)
    n1, n2 = np.where(conn_mat1 > 0)
    edge_weight = np.abs(beta_mat[n1, n2])
    out = dict(
        gene1=gene_names[n1],
        gene2=gene_names[n2],
        condition=group_idx,
        weight=edge_weight,
    )
    return pd.DataFrame(data=out)


def get_node_type_and_label_two_parts(
    nodes_show, part1_id="SP", part2_id="TF", ofst1=2, ofst2=1
):
    """Given a list of features, divide them to to groups, and clean their names.

    The first several characters in the name of each feature is related to their group information.
    For example, for feature "SP_AAAA", it means a gene in the "SP" group and the gene name is "AAAA".
    For feature "TF_BBBB", it means a gene in "TF" group with name "BBBB".
    We want to put thing starting with "TF" and "SP" to two separate groups.
    We also want to create a simplifed label that do not contain "SP" and "TP" to make plotting easier.
    
    Parameters
    ----------
    nodes_show : list of str
        The list of feature names
    part1_id : str, optional
        The label for group 1, by default "SP"
    part2_id : str, optional
        The label for group 2, by default "TF"
    ofst1 : int, optional
        The number of extra characters to remove the prefix before the gene name, by default 2
    ofst2 : int, optional
        The number of extra characters to remove the prefix before the gene name, by default 1

    Returns
    -------
    nodes_type : dict
        For each name in the feature, give its group index.
    labels : dict
        For each name in the feature, give its simplified name.
    """
    # TODO: support more parts
    x_len1 = len(part1_id)
    x_len2 = len(part2_id)

    # make labels shorter
    nodes_type = dict()
    labels = dict()
    for i, node in enumerate(nodes_show):
        if node[:x_len1] == part1_id:
            labels[node] = node[x_len1 + ofst1 :]
            nodes_type[node] = 0
        if node[:x_len2] == part2_id:
            labels[node] = node[x_len2 + ofst2 :]
            nodes_type[node] = 1

    return nodes_type, labels


def get_edge_subset_by_name(edge_df, name_match="TF"):
    """Extract subset of edges in the data frame.

    Parameters
    ----------
    edge_df : pandas.DataFrame
        The data frame containing edges.
    name_match : str, optional
        The pattern of the name to match, by default "TF"
        Must be at the beginning of the name of each feature.

    Returns
    -------
    pandas.DataFrame
        Extracted edges
    """
    gene_sel = np.zeros(len(edge_df))
    nn = len(name_match)
    for i in range(len(edge_df)):
        gene1, gene2, _, _ = edge_df.iloc[i]
        if gene1[:nn] == name_match and gene2[:nn] == name_match:
            gene_sel[i] = 1
    return edge_df[gene_sel > 0]


def get_node_subset_by_name(nodes, name_match="TF"):
    """Get a subset of the nodes according to the given pattern"""
    nodes_sel = []
    nn = len(name_match)
    for node in nodes:
        if node[:nn] == name_match:
            nodes_sel.append(node)
    return nodes_sel
