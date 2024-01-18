"""Visualization functions of DDN, improved v2 version

These functions are still quite basic.
For advanced plotting of networks, consider using specialized tools.
"""

import math
import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist
import networkx as nx
import matplotlib.pyplot as plt


def draw_network_for_ddn(
    edges_df,
    nodes_to_draw,
    mode="common",
    nodes_type=None,
    cen_lst=None,
    rad_lst=None,
    labels=None,
    fig_size=(16, 9),
    font_size_scale=1,
    node_size_scale=1,
    pdf_name="",
):
    """Draw networks for DDN

    Support drawing any number of ellipses according to `nodes_type`.
    The positions and shapes are given in `cen_lst` and `rad_lst`.
    This makes the layout of the graphs more flexible.

    The direction of the labels now points to the center of each ellipse.
    This allows showing larger fonts.

    The node size, font size, edge weight are now automatically adjusted according to figure size and node number.

    For common network, if two nodes in an edge have same type, draw grey line.
    If two nodes in an edge have different type, draw green line.

    For differential network, if an edge comes from condition 1, draw blue line.
    If an edge comes from condition 2, draw red line.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Edge information.
        First two columns for the two feature names.
        Third column for edge type (common=0, diff1=1, diff2=2)
        Fourth column for weight.
    nodes_to_draw : list of str
        Name of nodes to draw
    fig_size : tuple, optional
        Size of figure.
    font_size_scale : int, optional
        Scale of fonts, by default 1
        To make the font larger, set larger value.
        The default value is scaled according to the node number.
    node_size_scale : int, optional
        Scale of node sizes, by default 2
    nodes_type : None or dict, optional
        Node type (e.g., Gene=0, TF=1), by default None
    cen_lst : None or ndarray, optional
        The center of ellipse for each type of node. For node type i, use cen_lst[i]
        Shape (k, 2), k is the number of types. 2 for (x, y)
        Assume the y-axis of the figure is in range [-1,1], so do not set too large values.
    rad_lst : None or ndarray, optional
        The radius of ellipse for each type of node. For node type i, use rad_lst[i]
        Shape (k, 2), k is the number of types. 2 for shorter and longer axis length.
        Assume the y-axis of the figure is in range [-1,1], so do not set too large values.
    labels : dict
        Alternative (e.g., simplified) names for nodes
    mode : str, optional
        Draw common graph or differential graph, by default "common"
    pdf_name : str, optional
        Name of the PDF file to export, by default "". If set as "", no pdf will be output.

    Returns
    -------
    nx.Graph
        Graph object
    """
    # by default, assume there is one node type, and we draw circle
    if nodes_type is None:
        nodes_type = dict()
        for node in nodes_to_draw:
            nodes_type[node] = 0

    if cen_lst is None:
        cen_lst = np.array([[0.0, 0.0]])

    if rad_lst is None:
        rad_lst = np.array([[1.0, 1.0]])

    # create networkx graph
    G = create_nx_graph(
        edges_df,
        nodes_to_draw,
        mode=mode,
        nodes_type=nodes_type,
    )

    if labels is None:
        labels = dict((n, n) for n in G.nodes())

    # nodes positions
    pos, d_min = get_pos_multi_parts(
        nodes_to_draw, nodes_type, cen_lst=cen_lst, rad_lst=rad_lst
    )

    # plot the network
    fig, ax = plot_network(
        G,
        pos,
        d_min=d_min,
        labels=labels,
        node_type=nodes_type,
        cen_lst=cen_lst,
        fig_size=fig_size,
        font_size_scale=font_size_scale,
        node_size_scale=node_size_scale,
    )

    # export figure
    if len(pdf_name) > 0:
        plt.savefig(f"{pdf_name}_{mode}.pdf", format="pdf", bbox_inches="tight")

    return G, fig, ax


def create_nx_graph(
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


def plot_network(
    G,
    pos,
    d_min,
    labels,
    node_type,
    cen_lst,
    fig_size,
    font_size_scale=1,
    node_size_scale=2,
    font_alpha_min=0.4,
):
    """Draw the network

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
    node_type : None or dict, optional
        Node type (e.g., Gene=0, TF=1), by default None
    cen_lst : None or ndarray, optional
        The center of ellipse for each type of node. For node type i, use cen_lst[i]
        Shape (k, 2), k is the number of types. 2 for (x, y)
        Assume the y-axis of the figure is in range [-1,1], so do not set too large values.
    fig_size : tuple
        Size of figure. The unit is inch.
    font_size_scale : int, optional
        Scale of fonts, by default 1
        To make the font larger, set larger value.
        The default value is scaled according to the node number.
    node_size_scale : int, optional
        Scale of node sizes, by default 1
    font_alpha_min : float, optional
        The smallest alpha value for fonts in labels, between 0 and 1
    """
    # The positions are given in a [-a,a]x[-1,1] region
    # Re-scale it to figure size, but leave some margin for text (here 0.8)
    fig_half_size = fig_size[1] * 0.9 / 2
    for x in pos:
        pos[x] = pos[x] * fig_half_size
    cen_lst = cen_lst * fig_half_size
    d_min = d_min * fig_half_size

    # node size
    # Node size in unit points^2. 1 inch = 72 points.
    # in case all nodes have degree zero
    # too large nodes are ugly
    s_min = (d_min * 72) ** 2
    s_min = min(s_min, 36*36)
    node_size = np.array([d for n, d in G.degree()]) + 0.1
    node_size = node_size / (np.max(node_size) + 1)
    node_size = node_size * s_min * node_size_scale

    # font size
    # In points. 1 inch = 72 points. Font size about the height of a character.
    # too large font may go outside the figure
    font_size = d_min * 72 * 0.8 * font_size_scale
    font_size = min(font_size, fig_half_size * 0.1 * 20)
    font_size_lst = font_size + node_size*0
    # font_size_lst = font_size * (
    #     np.abs(node_size) / np.max(node_size) * (1.0 - 0.5) + 0.5
    # )
    font_alpha_lst = np.abs(node_size) / np.max(node_size) * (1.0 - font_alpha_min) + font_alpha_min

    # draw
    fig, ax = plt.subplots(figsize=fig_size)

    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_color="lightblue",
        node_size=node_size,
        alpha=0.5,
    )

    # edges properties
    # in case there are no edges
    # too thick edges are ugly
    edges = G.edges()
    if len(edges) > 0:
        edge_color = [G[u][v]["color"] for u, v in edges]
        edge_weight = np.array([G[u][v]["weight"] for u, v in edges])

        # when there are too many edges, make the edge thin
        # edge weight also in points, 1 inch = 72 points
        d_min1 = min(d_min, 0.25)
        edge_weight = edge_weight / np.max(edge_weight) * d_min1 * 72 / 6
        if len(edge_weight) > 200:
            edge_weight = edge_weight / len(edge_weight) * 200

        nx.draw_networkx_edges(
            G,
            pos=pos,
            ax=ax,
            edgelist=edges,
            edge_color=edge_color,
            width=edge_weight,
        )

    _draw_network_labels(
        ax, pos, d_min, node_type, cen_lst, labels, font_size_lst, font_alpha_lst
    )

    ax.set_xlim((-fig_size[0] / 2, fig_size[0] / 2))
    ax.set_ylim((-fig_size[1] / 2, fig_size[1] / 2))
    return fig, ax


def get_pos_multi_parts(
    nodes_all,
    nodes_type,
    cen_lst=np.array([[-0.6, 0], [0.6, 0]]),
    rad_lst=np.array([[0.4, 1], [0.4, 1]]),
):
    """Find positions of nodes with multiple isolated subgraphs.

    Each subgraph is an ellipse.
    Also provides the minimum distances between nodes.

    Parameters
    ----------
    nodes_all : list of str
        Node names
    nodes_type : dict
        feature type ID for each node
    cen_lst : None or ndarray, optional
        The center of ellipse for each type of node. For node type i, use cen_lst[i]
        Shape (k, 2), k is the number of types. 2 for (x, y)
        Assume the y-axis of the figure is in range [-1,1], so do not set too large values.
    rad_lst : None or ndarray, optional
        The radius of ellipse for each type of node. For node type i, use rad_lst[i]
        Shape (k, 2), k is the number of types. 2 for shorter and longer axis length.
        Assume the y-axis of the figure is in range [-1,1], so do not set too large values.

    Returns
    -------
    pos : dict
        The positions for each node
    d_min : float
        Minimum distances between nodes
    """
    # types of features
    fea_type_lst = []
    for node in nodes_type:
        fea_type_lst.append(nodes_type[node])
    fea_type_enu = np.unique(np.array(fea_type_lst))

    # create positions for each type of features
    pos = dict()
    d_min_lst = []
    cntt = 0
    for fea_id in fea_type_enu:
        nodes = []
        for node in nodes_all:
            if nodes_type[node] == fea_id:
                nodes.append(node)
        pos, d_min = _get_pos_one_part(
            nodes, cen=cen_lst[cntt], rad=rad_lst[cntt], pos=pos
        )
        d_min_lst.append(d_min)
        cntt += 1
    d_min = np.min(np.array(d_min_lst))

    return pos, d_min


def _get_pos_one_part(
    nodes_show,
    cen=(0.0, 0.0),
    rad=(1.0, 1.0),
    pos=None,
):
    """Find positions of nodes on a circle

    Also provides the minimum distances between nodes.

    Parameters
    ----------
    nodes_show : tuple of str
        Node names
    cen : ndarray, optional
        The center of ellipse for each type of node. Shape (2, )
        Assume the y-axis of the figure is in range [-1,1], so do not set too large values.
    rad : ndarray, optional
        The radius of ellipse for each type of node. Shape (2, )
        Assume the y-axis of the figure is in range [-1,1], so do not set too large values.
    pos : dict
        NetworkX position dictionary

    Returns
    -------
    pos : dict
        The positions for each node
    d_min : float
        Minimum distances between nodes
    """
    n = len(nodes_show)
    r0 = float(rad[0])
    r1 = float(rad[1])
    if r0 == r1:
        r0 = r0 * 0.99999
    if r0 > r1:
        r0, r1 = r1, r0
    angle = _angles_in_ellipse(n, r0, r1)

    if pos is None:
        pos = dict()
    d_min = _add_node_to_a_circle(pos, nodes_show, cen, rad, angle)
    return pos, d_min


def _draw_network_labels(
    ax,
    pos,
    d_min,
    node_type,
    cen_lst,
    labels,
    font_size_lst,
    font_alpha_lst,
):
    """Add labels to a graph

    Modified from Networkx add label function

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes of the figure
    pos : dict
        NetworkX position dictionary
    d_min : float
        Minimum distance between nodes
    node_type : dict
        Feature type index of each node
    cen_lst : None or ndarray, optional
        The center of ellipse for each type of node. For node type i, use cen_lst[i]
        Shape (k, 2), k is the number of types. 2 for (x, y)
        Assume the y-axis of the figure is in range [-1,1], so do not set too large values.
    labels : dict
        Alternative names for each node, to be shown in the graph
    font_size_lst : array_like
        Font size for each node, in points
    font_alpha_lst : array_like
        Font alpha for each node, value from 0 to 1
    """
    cnt = 0
    for node, label in labels.items():
        (x, y) = pos[node]
        fea_idx = node_type[node]
        x1 = x - cen_lst[fea_idx][0]
        y1 = y - cen_lst[fea_idx][1]

        # rotate labels to make it point to the center
        # this allows larger fonts
        # treat labels on the right and those on the left differently
        angle = math.atan2(y1, np.abs(x1)) / math.pi * 180
        if x1 < 0:
            angle = -angle

        # this makes "1" and 1 labeled the same
        if not isinstance(label, str):
            label = str(label)

        nn = np.sqrt(x1**2 + y1**2)
        x1n = x1 / nn
        y1n = y1 / nn

        x_ext = x + x1n * (len(label) / 2 * font_size_lst[cnt] / 72 + d_min / 4)
        y_ext = y + y1n * (len(label) / 2 * font_size_lst[cnt] / 72 + d_min / 4)

        _ = ax.text(
            x_ext,
            y_ext,
            label,
            size=font_size_lst[cnt],
            color="k",
            family="sans-serif",
            weight="normal",
            alpha=font_alpha_lst[cnt],
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transData,
            rotation=angle,
            bbox=None,
            clip_on=True,
        )
        cnt += 1

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )


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
    cnt = 0
    for node in nodes:
        theta = angles[cnt]
        pos0 = np.array(
            [cen[0] + np.cos(theta) * rad[0], cen[1] + np.sin(theta) * rad[1]]
        )
        pos[node] = pos0
        pos_lst.append(pos0)
        cnt += 1
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
