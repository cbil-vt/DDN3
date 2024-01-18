"""Visualization functions of DDN

These functions are quite basic. 
For advanced plotting of networks, consider using specialized tools.
"""

import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist
import networkx as nx
import matplotlib.pyplot as plt


def draw_network_for_ddn(
    edges_df,
    nodes_to_draw,
    fig_size=18,
    font_size_scale=1,
    node_size_scale=2,
    part_number=1,
    nodes_type=None,
    labels=None,
    mode="common",
    export_pdf=True,
    pdf_name="",
):
    """Draw networks for DDN

    By default, we draw all nodes to a circle.
    It is possible for names of features to overlap with each other,
    especially at the top or bottom of the circle.
    If this happens, consider setting a smaller `font_size_scale`.

    For common network, if two nodes in an edge have same type, draw grey line.
    If  two nodes in an edge have different type, draw green line.

    For differential network, if an edge comes from condition 1, draw blue line.
    If an edge comes from condition 2, draw red line.

    We support drawing nodes in one or two types (like Gene vs. TF).
    If two types of nodes are present, we draw two ellipses.
    See `part_number` and `nodes_type` parameters.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Edge information. 
        First two columns for the two feature names. 
        Third column for edge type (common=0, diff1=1, diff2=2)
        Fourth column for weight.
    nodes_to_draw : list of str
        Name of nodes to draw
    fig_size : int, optional
        Size of figure, by default 18
        We draw in a square figure, so width=height=`fig_size`.
    font_size_scale : int, optional
        Scale of fonts, by default 1
        To make the font larger, set larger value.
        The default value is scaled according to the node number.
    node_size_scale : int, optional
        Scale of node sizes, by default 2
    part_number : int, optional
        Number of ellipse to draw, by default 1
        If set to 1, all nodes are in the same condition, draw a single circle.
        If set to 2, nodes are in two conditions (like Gene and TF), we draw two ellipses.
        If set to 2, `nodes_type` is needed.
    nodes_type : None or dict, optional
        Node type (e.g., Gene=0, TF=1), by default None
    labels : dict
        Alternative (e.g., simplfied) names for nodes
    mode : str, optional
        Draw common graph or differential graph, by default "common"
    export_pdf : bool, optional
        Set to true want to export the graph as a PDF file, by default True
    pdf_name : str, optional
        Name of the PDF file to export, by default ""

    Returns
    -------
    nx.Graph
        Graph object
    """
    # create networkx graph
    G = _create_nx_graph(
        edges_df,
        nodes_to_draw,
        mode=mode,
        nodes_type=nodes_type,
    )

    # nodes positions
    # must provide nodes_type dictionary for two parts
    if part_number == 1:
        pos, d_min = _get_pos_one_part(nodes_to_draw)
    elif part_number == 2:
        pos, d_min = _get_pos_two_parts(nodes_to_draw, nodes_type)
    else:
        raise ("Not implemented")

    # plot the network
    _plot_network_helper(
        G,
        pos,
        d_min=d_min,
        labels=labels,
        fig_size=fig_size,
        font_size_scale=font_size_scale,
        node_size_scale=node_size_scale,
    )

    # export figure
    if export_pdf:
        plt.savefig(f"{pdf_name}_{mode}.pdf", format="pdf", bbox_inches="tight")

    return G


def _add_node_to_a_circle(pos, nodes, cen, rad, angles):
    """Find positions in an ellipse, and calculate the minimum distances between points

    Parameters
    ----------
    pos : dict
        For saving the position results for NetworkX
    nodes : list of str
        Name of nodes
    cen : array_like
        Central position of this circle, shape (2,)
    rad : array_like
        Length of two axes of the ellipse, shape (2,)
    angles : array_like
        Angles of the points

    Returns
    -------
    float
        Minimum distances between points
    """
    pos_lst = []
    for i, node in enumerate(nodes):
        theta = angles[i]
        pos0 = np.array(
            [cen[0] + np.cos(theta) * rad[0], cen[1] + np.sin(theta) * rad[1]]
        )
        pos[node] = pos0
        pos_lst.append(pos0)
    pos_lst = np.array(pos_lst)
    if len(pos_lst) > 1:
        d = pdist(pos_lst)
        return np.min(d)
    else:
        return 0.5


def _angles_in_ellipse(num, a, b):
    """Calculate angles of evenly spaced points in an ellipse

    Based on https://stackoverflow.com/a/52062369, 
    which is from https://pypi.org/project/flyingcircus/

    Parameters
    ----------
    num : int
        Sample number to get
    a : float
        Length of shorter axis
    b : float
        Length of longer axis

    Returns
    -------
    angles : ndarray
        Angles of sampled points
    """
    assert num > 0
    assert a < b
    angles = 2 * np.pi * np.arange(num) / num
    if a != b:
        e2 = 1.0 - a**2.0 / b**2.0
        tot_size = sp.special.ellipeinc(2.0 * np.pi, e2)
        arc_size = tot_size / num
        arcs = np.arange(num) * arc_size
        res = sp.optimize.root(lambda x: (sp.special.ellipeinc(x, e2) - arcs), angles)
        angles = res.x
    return angles


def _get_pos_one_part(nodes_show):
    """Find positions of nodes on a circle

    Also provides the minimum distances between nodes.

    Parameters
    ----------
    nodes_show : list of str
        Node names

    Returns
    -------
    pos : dict
        The positions for each node
    d_min : float
        Minimum distances between nodes
    """
    # positions of nodes
    n = len(nodes_show)

    # generate positions
    angle = _angles_in_ellipse(n, 0.999, 1.0)
    cen = [0.0, 0]
    rad = [1.0, 1.0]

    pos = dict()
    d_min = _add_node_to_a_circle(pos, nodes_show, cen, rad, angle)
    return pos, d_min


def _get_pos_two_parts(nodes_show, nodes_type):
    """Find positions of nodes with two groups.

    Group 1 on the left ellipse, group 2 on the right.
    Also provides the minimum distances between nodes.

    Parameters
    ----------
    nodes_show : list of str
        Node names
    nodes_type : dict
        Group ID for each node

    Returns
    -------
    pos : dict
        The positions for each node
    d_min : float
        Minimum distances between nodes
    """
    # positions of nodes
    nodes_sp = []
    nodes_tf = []
    for node in nodes_show:
        if nodes_type[node] == 0:
            nodes_sp.append(node)
        if nodes_type[node] == 1:
            nodes_tf.append(node)

    n_sp = len(nodes_sp)
    n_tf = len(nodes_tf)

    # generate positions
    angle_sp = _angles_in_ellipse(n_sp, 0.4, 1.0)
    angle_tf = _angles_in_ellipse(n_tf, 0.4, 1.0)

    cen_sp = [-0.6, 0]
    cen_tf = [0.6, 0]
    rad_sp = [0.4, 1]
    rad_tf = [0.4, 1]

    pos = dict()
    d_min_sp = _add_node_to_a_circle(pos, nodes_sp, cen_sp, rad_sp, angle_sp)
    d_min_tf = _add_node_to_a_circle(pos, nodes_tf, cen_tf, rad_tf, angle_tf)
    d_min = min(d_min_sp, d_min_tf)
    return pos, d_min


def _create_nx_graph(
    edges_df,
    nodes_show,
    min_alpha=0.2,
    max_alpha=1.0,
    mode="common",
    nodes_type=None,
):
    """Create NetworkX graph based on edge data frame.

    Add nodes and edges. Provide visualization related properties to the nodes.

    For common network, if two nodes in an edge have same type, draw grey line.
    If  two nodes in an edge have different type, draw green line.

    For differential network, if an edge comes from condition 1, draw blue line.
    If an edge comes from condition 2, draw red line.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Edge information. 
        First two columns for the two feature names. 
        Third column for edge type (common=0, diff1=1, diff2=2)
        Fourth column for weight.
    nodes_show : list of str
        Name of nodes to draw
    min_alpha : float, optional
        Minimum alpha value of edges, by default 0.2
        This is for the most light edges.
    max_alpha : float, optional
        Maximum alpha value of edges, by default 1.0
    mode : str, optional
        Draw common graph or differential graph, by default "common"
    nodes_type : None or dict, optional
        Node type (e.g., Gene=0, TF=1), by default None

    Returns
    -------
    nx.Graph
        Generated graph
    """
    # create the overall graph
    color_condition = {0: [0.7, 0.7, 0.7], 1: [0, 0, 1], 2: [1, 0, 0], 3: [0, 0.6, 0.3]}
    beta_max = np.max(edges_df["weight"])

    if nodes_type is None:
        nodes_type = dict()
        for node in nodes_show:
            nodes_type[node] = 0

    G = nx.Graph()
    for node in nodes_show:
        G.add_node(node)

    for i in range(len(edges_df)):
        gene1, gene2, condition, beta = edges_df.iloc[i]
        if condition in color_condition:
            alpha = np.abs(beta) / beta_max * (max_alpha - min_alpha) + min_alpha
            weight = np.abs(beta) / beta_max * 3.0 + 0.5
            if mode != "common":
                color = list(1 - (1 - np.array(color_condition[condition])) * alpha)
            else:
                if nodes_type[gene1] == nodes_type[gene2]:
                    color = list(1 - (1 - np.array(color_condition[0])) * alpha)
                else:
                    color = list(1 - (1 - np.array(color_condition[3])) * alpha)
            G.add_edge(gene1, gene2, color=color, weight=weight)

    return G


def _plot_network_helper(
    G,
    pos,
    d_min,
    labels,
    fig_size=18,
    font_size_scale=1,
    node_size_scale=2,
):
    """Draw the network

    The graph elements are optimized for figure size 18 x 18.
    For smaller figures, users may need to change `font_size_scale` and `node_size_scale`.

    Parameters
    ----------
    G : nx.Graph
        Graph to draw
    pos : dict
        Position of each node
    d_min : float
        Minimum distance between nodes.
        We use this to adjust node size, font size, etc.
    labels : dict
        Alternative names for nodes
    fig_size : int, optional
        Size of figure, by default 18
        We draw in a square figure, so width=height=`fig_size`.
    font_size_scale : int, optional
        Scale of fonts, by default 1
        To make the font larger, set larger value.
        The default value is scaled according to the node number.
    node_size_scale : int, optional
        Scale of node sizes, by default 2
    """
    # scale according to distances between nodes
    s_min = (d_min * 100) ** 2
    s_min = min(s_min, 500)

    # node size
    node_size = np.array([d for n, d in G.degree()])
    node_size = node_size / (np.max(node_size) + 1)
    node_size = node_size * s_min * node_size_scale * 10

    # font size
    font_size = d_min * font_size_scale * 100
    font_size = min(font_size, 10)

    # edges properties
    edges = G.edges()
    edge_color = [G[u][v]["color"] for u, v in edges]
    edge_weight = np.array([G[u][v]["weight"] for u, v in edges])

    # when there are too many edges, make the edge thin
    if len(edge_weight) > 400:
        edge_weight = edge_weight / len(edge_weight) * 400

    # draw
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_color="lightblue",
        node_size=node_size,
        alpha=0.5,
    )

    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax,
        edgelist=edges,
        edge_color=edge_color,
        width=edge_weight,
    )

    nx.draw_networkx_labels(
        G,
        pos=pos,
        ax=ax,
        labels=labels,
        font_size=font_size,
        font_color="blueviolet",
    )

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim(ax.get_xlim())
