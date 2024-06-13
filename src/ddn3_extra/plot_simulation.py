import numpy as np
import matplotlib.pyplot as plt
from ddn3 import performance, simulation
from ddn3_extra import simulation_r


def plot_curve_with_noise(x_in, y_in, style=0, noise_scale=0.01):
    x = x_in + np.random.randn(len(x_in)) * noise_scale
    y = y_in + np.random.randn(len(y_in)) * noise_scale
    if style == 0:
        plt.scatter(x, y, s=1.0)
    else:
        plt.plot(x, y, s=1.0)


def draw1(
    ax,
    res,
    idx1,
    idx2,
    remove_zero=False,
    color="blue",
    alpha=1.0,
    ratio_max=0.9,
    label="",
):
    x = res[:, idx1]
    y = res[:, idx2]
    if remove_zero:
        idx = (x > 0) * (y > 0)
        x = x[idx]
        y = y[idx]

    ax.plot(x, y, color=color, alpha=alpha, label=label)

    # find x index where y larger enough
    x_idx = np.where(y > ratio_max * np.max(y))[0]
    if len(x_idx) == 0:
        return 0

    # from these x, find the one with smallest y
    idx1 = x_idx[np.argmin(y[x_idx])]

    return x[idx1]


def plot_curve(res_comm, res_diff, idx1=2, idx2=4):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    draw1(ax[0], res_comm, idx1, idx2)
    draw1(ax[1], res_diff, idx1, idx2)


def get_range_large(x, y):
    y_sel = np.where(y > 0.5 * np.max(y))[0]
    x_left = x[np.min(y_sel)]
    x_right = x[np.max(y_sel)]
    return x_left, x_right


def plot_average_f1(ax, res_ddn, res_jgl, l1_lst, hightlight=False):
    ddn_f1 = get_f1_mat(res_ddn)
    jgl_f1 = get_f1_mat(res_jgl)
    f1_avg_ddn = (ddn_f1[:, 0] + ddn_f1[:, 1]) / 2
    f1_avg_jgl = (jgl_f1[:, 0] + jgl_f1[:, 1]) / 2
    f1_max = max(np.max(f1_avg_ddn), np.max(f1_avg_jgl))

    n_l2 = len(f1_avg_ddn)
    c0 = "tab:blue"
    c1 = "tab:orange"
    x_left_lst = []
    x_right_lst = []
    for i in range(n_l2):
        m0 = f1_avg_ddn[i]
        m1 = f1_avg_jgl[i]

        xl0, xr0 = get_range_large(l1_lst, m0)
        xl1, xr1 = get_range_large(l1_lst, m1)
        x_left_lst.extend([xl0, xl1])
        x_right_lst.extend([xr0, xr1])

        alpha = 0.2 + 0.78 * i / (n_l2 - 1)
        ax.plot(l1_lst, m0, color=c0, linewidth=i * 0.2 + 1, alpha=alpha)
        ax.plot(l1_lst, m1, color=c1, linewidth=i * 0.2 + 1, alpha=alpha)

    if hightlight:
        ax.set_xlim([np.mean(x_left_lst), np.mean(x_right_lst)])
        ax.set_ylim([f1_max * 0.3, f1_max * 1.05])


def plot_ddn_jgl_res(res, l1_lst, l2_lst):
    res_mean = np.mean(res, axis=0)
    res_ddn_mean = res_mean[0]
    res_jgl_mean = res_mean[1]

    # For each method
    # l2 number, l1 number, comm/diff, metric number

    fig, ax = plt.subplots(3, 2, figsize=(11, 15), layout="constrained")

    n_l2idx = len(res_ddn_mean)
    xcut_comm = []
    xcut_diff = []
    for l2idx in range(n_l2idx):
        alpha = 0.2 + 0.78 * l2idx / (n_l2idx - 1)
        # alpha = 1.0
        col_ddn = "tab:blue"
        col_jgl = "tab:orange"

        # common
        xddn = res_ddn_mean[l2idx, :, 0]
        xjgl = res_jgl_mean[l2idx, :, 0]
        label = f"DDN {l2_lst[l2idx]:.3f}"
        _ = draw1(
            ax[0, 0],
            xddn,
            2,
            4,
            remove_zero=True,
            color=col_ddn,
            alpha=alpha,
            label=label,
        )
        label = f"JGL {l2_lst[l2idx]:.3f}"
        _ = draw1(
            ax[0, 0],
            xjgl,
            2,
            4,
            remove_zero=True,
            color=col_jgl,
            alpha=alpha,
            label=label,
        )
        c0 = draw1(ax[0, 1], xddn, 1, 0, color=col_ddn, alpha=alpha)
        c1 = draw1(ax[0, 1], xjgl, 1, 0, color=col_jgl, alpha=alpha)
        xcut_comm.extend([c0, c1])

        # diff
        xddn = res_ddn_mean[l2idx, :, 1]
        xjgl = res_jgl_mean[l2idx, :, 1]
        _ = draw1(ax[1, 0], xddn, 2, 4, remove_zero=True, color=col_ddn, alpha=alpha)
        _ = draw1(ax[1, 0], xjgl, 2, 4, remove_zero=True, color=col_jgl, alpha=alpha)
        c0 = draw1(ax[1, 1], xddn, 1, 0, color=col_ddn, alpha=alpha)
        c1 = draw1(ax[1, 1], xjgl, 1, 0, color=col_jgl, alpha=alpha)
        xcut_diff.extend([c0, c1])

    # print(xcut_comm)
    # print(xcut_diff)

    ax[0, 1].set_xlim([0, np.mean(xcut_comm)])
    ax[1, 1].set_xlim([0, np.mean(xcut_diff)])

    ax[0, 0].set_title("Common network")
    ax[0, 0].set_xlabel("Recall")
    ax[0, 0].set_ylabel("Precision")

    ax[0, 1].set_title("Common network")
    ax[0, 1].set_xlabel("FP")
    ax[0, 1].set_ylabel("TP")

    ax[1, 0].set_title("Differential network")
    ax[1, 0].set_xlabel("Recall")
    ax[1, 0].set_ylabel("Precision")

    ax[1, 1].set_title("Differential network")
    ax[1, 1].set_xlabel("FP")
    ax[1, 1].set_ylabel("TP")

    plot_average_f1(ax[2, 0], res_ddn_mean, res_jgl_mean, l1_lst, hightlight=False)
    ax[2, 0].set_title("Overall performance")
    ax[2, 0].set_xlabel("$\lambda_1$")
    ax[2, 0].set_ylabel("Average $F_1$ score")

    plot_average_f1(ax[2, 1], res_ddn_mean, res_jgl_mean, l1_lst, hightlight=True)
    ax[2, 1].set_title("Overall performance")
    ax[2, 1].set_xlabel("$\lambda_1$")
    ax[2, 1].set_ylabel("Average $F_1$ score")

    # re-order the legend
    handles, labels = ax[0, 0].get_legend_handles_labels()
    order = list(range(0, len(labels), 2)) + list(range(1, len(labels), 2))
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    fig.legend(handles, labels, loc="outside right center")

    fig.get_layout_engine().set(
        hspace=0.1,
    )

    return fig, ax


def plot_ddn_jgl_simple(
    ax,
    res,
    l2_lst,
):
    """For comparison with DINGO"""
    res_mean = np.mean(res, axis=0)
    res_ddn_mean = res_mean[0]
    res_jgl_mean = res_mean[1]

    # For each method
    # l2 number, l1 number, comm/diff, metric number   

    n_l2idx = len(res_ddn_mean)
    xcut_comm = []
    # xcut_diff = []
    for l2idx in range(n_l2idx):
        alpha = 0.2 + 0.78 * l2idx / (n_l2idx - 1)
        # alpha = 1.0
        col_ddn = "tab:blue"
        col_jgl = "tab:orange"

        # common
        xddn = res_ddn_mean[l2idx, :, 0]
        xjgl = res_jgl_mean[l2idx, :, 0]
        label = f"DDN {l2_lst[l2idx]:.3f}"
        _ = draw1(
            ax[0],
            xddn,
            2,
            4,
            remove_zero=True,
            color=col_ddn,
            alpha=alpha,
            label=label,
        )
        label = f"JGL {l2_lst[l2idx]:.3f}"
        _ = draw1(
            ax[0],
            xjgl,
            2,
            4,
            remove_zero=True,
            color=col_jgl,
            alpha=alpha,
            label=label,
        )
        c0 = draw1(ax[1], xddn, 1, 0, color=col_ddn, alpha=alpha)
        c1 = draw1(ax[1], xjgl, 1, 0, color=col_jgl, alpha=alpha)
        xcut_comm.extend([c0, c1])

        # # diff
        # xddn = res_ddn_mean[l2idx, :, 1]
        # xjgl = res_jgl_mean[l2idx, :, 1]
        # _ = draw1(ax[1, 0], xddn, 2, 4, remove_zero=True, color=col_ddn, alpha=alpha)
        # _ = draw1(ax[1, 0], xjgl, 2, 4, remove_zero=True, color=col_jgl, alpha=alpha)
        # c0 = draw1(ax[1, 1], xddn, 1, 0, color=col_ddn, alpha=alpha)
        # c1 = draw1(ax[1, 1], xjgl, 1, 0, color=col_jgl, alpha=alpha)
        # xcut_diff.extend([c0, c1])

    ax[1].set_xlim([0, np.mean(xcut_comm)*4])
    # ax[1, 1].set_xlim([0, np.mean(xcut_diff)])

    ax[0].set_title("Common network")
    ax[0].set_xlabel("Recall")
    ax[0].set_ylabel("Precision")

    ax[1].set_title("Common network")
    ax[1].set_xlabel("FP")
    ax[1].set_ylabel("TP")

    # ax[1, 0].set_title("Differential network")
    # ax[1, 0].set_xlabel("Recall")
    # ax[1, 0].set_ylabel("Precision")

    # ax[1, 1].set_title("Differential network")
    # ax[1, 1].set_xlabel("FP")
    # ax[1, 1].set_ylabel("TP")



def plot_ddn_jgl_simple1(
    ax,
    res,
    l2_lst,
):
    """For comparison with DINGO"""
    res_mean = np.mean(res, axis=0)
    res_ddn_mean = res_mean[0]
    res_jgl_mean = res_mean[1]

    # For each method
    # l2 number, l1 number, comm/diff, metric number   

    n_l2idx = len(res_ddn_mean)
    xcut_comm = []
    xcut_diff = []
    for l2idx in range(n_l2idx):
        alpha = 0.2 + 0.78 * l2idx / (n_l2idx - 1)
        # alpha = 1.0
        col_ddn = "tab:blue"
        col_jgl = "tab:orange"

        # common
        xddn = res_ddn_mean[l2idx, :, 0]
        xjgl = res_jgl_mean[l2idx, :, 0]
        label = f"DDN {l2_lst[l2idx]:.3f}"
        _ = draw1(
            ax[0, 0],
            xddn,
            2,
            4,
            remove_zero=True,
            color=col_ddn,
            alpha=alpha,
            label=label,
        )
        label = f"JGL {l2_lst[l2idx]:.3f}"
        _ = draw1(
            ax[0, 0],
            xjgl,
            2,
            4,
            remove_zero=True,
            color=col_jgl,
            alpha=alpha,
            label=label,
        )
        c0 = draw1(ax[0, 1], xddn, 1, 0, color=col_ddn, alpha=alpha)
        c1 = draw1(ax[0, 1], xjgl, 1, 0, color=col_jgl, alpha=alpha)
        xcut_comm.extend([c0, c1])

        # diff
        xddn = res_ddn_mean[l2idx, :, 1]
        xjgl = res_jgl_mean[l2idx, :, 1]
        _ = draw1(ax[1, 0], xddn, 2, 4, remove_zero=True, color=col_ddn, alpha=alpha)
        _ = draw1(ax[1, 0], xjgl, 2, 4, remove_zero=True, color=col_jgl, alpha=alpha)
        c0 = draw1(ax[1, 1], xddn, 1, 0, color=col_ddn, alpha=alpha)
        c1 = draw1(ax[1, 1], xjgl, 1, 0, color=col_jgl, alpha=alpha)
        xcut_diff.extend([c0, c1])

    ax[0, 1].set_xlim([0, np.mean(xcut_comm)*4])
    ax[1, 1].set_xlim([0, np.mean(xcut_diff)])

    ax[0, 0].set_title("Common network")
    ax[0, 0].set_xlabel("Recall")
    ax[0, 0].set_ylabel("Precision")

    ax[0, 1].set_title("Common network")
    ax[0, 1].set_xlabel("FP")
    ax[0, 1].set_ylabel("TP")

    ax[1, 0].set_title("Differential network")
    ax[1, 0].set_xlabel("Recall")
    ax[1, 0].set_ylabel("Precision")

    ax[1, 1].set_title("Differential network")
    ax[1, 1].set_xlabel("FP")
    ax[1, 1].set_ylabel("TP")



def get_f1_mat(a_in):
    """Collect the F1 score for different lambda1 and lambda2"""
    n_l2 = len(a_in)
    n_l1 = a_in.shape[1]
    f1_mat = np.zeros((n_l2, 2, n_l1))
    for i in range(n_l2):
        f1_comm = performance.get_f1(a_in[i][:, 0, 2], a_in[i][:, 0, 4])
        f1_diff = performance.get_f1(a_in[i][:, 1, 2], a_in[i][:, 1, 4])
        f1_mat[i, 0] = f1_comm
        f1_mat[i, 1] = f1_diff
    return f1_mat


def get_best_f2(res_now):
    f1_comm_lst = []
    f1_diff_lst = []
    f1_mean_lst = []
    f1_max_lst = []

    for res0 in res_now:
        f1_comm = performance.get_f1(res0[:,0,2], res0[:,0,4])
        f1_diff = performance.get_f1(res0[:,1,2], res0[:,1,4])
        f1_mean = (f1_comm + f1_diff)/2
        f1_comm_lst.append(f1_comm)
        f1_diff_lst.append(f1_diff)
        f1_mean_lst.append(f1_mean)
        f1_max = np.max(f1_mean)
        f1_max_lst.append(f1_max)

    idx = np.argmax(f1_max_lst)
    return idx, f1_comm_lst[idx], f1_diff_lst[idx], f1_mean_lst[idx]


def get_ground_truth_edge_count(n_node, graph_type, n_group):
    _, omega1, omega2 = simulation_r.huge_omega(
        n_node=n_node, ratio_diff=0.25, graph_type=graph_type, n_group=n_group
    )
    _, _, comm_gt, diff_gt = simulation.prep_sim_from_two_omega(
        omega1, omega2
    )
    gt_comm_edge_num = np.sum(comm_gt)/2
    gt_diff_edge_num = np.sum(diff_gt)/2
    gt_comm_non_edge = (n_node*n_node - n_node)/2 - gt_comm_edge_num
    gt_diff_non_edge = (n_node*n_node - n_node)/2 - gt_diff_edge_num
    return gt_comm_edge_num, gt_comm_non_edge, gt_diff_edge_num, gt_diff_non_edge


def count_edges_P_N():
    ct = dict()
    ct["random"] = get_ground_truth_edge_count(100, "random", 1)
    ct["hub"] = get_ground_truth_edge_count(100, "hub", 5)
    ct["cluster"] = get_ground_truth_edge_count(100, "cluster", 5)
    ct["scale_free_4"] = get_ground_truth_edge_count(100, "scale-free-multi", 4)
    ct["scale_free_8_200"] = get_ground_truth_edge_count(200, "scale-free-multi", 8)
    ct["scale_free_16_400"] = get_ground_truth_edge_count(400, "scale-free-multi", 16)
    ct["imbalance"] = get_ground_truth_edge_count(100, "scale-free-multi", 2)
    return ct


def collect_curves(res_name_dict, dat_dir):
    curve_dict = dict()
    for graph_type, res_file in res_name_dict.items():
        temp = np.load(f"{dat_dir}/{graph_type}/{res_file}.npz")
        res = temp['res']
        l1_lst = temp['l1_lst']
        l2_lst = temp['l2_lst']
        res_mean = np.mean(res, axis=0)
        methods = ["ddn", "jgl"]
        for i, res_now in enumerate(res_mean):
            prefix = f"{graph_type}_{methods[i]}"
            idx, f1_comm, f1_diff, f1_mean = get_best_f2(res_now)
            curve_dict[f"{prefix}_f1_comm"] = f1_comm
            curve_dict[f"{prefix}_f1_diff"] = f1_diff
            curve_dict[f"{prefix}_f1_mean"] = f1_mean
            curve_dict[f"{prefix}_lambda1"] = l1_lst[np.argmax(f1_mean)]
            curve_dict[f"{prefix}_lambda2"] = l2_lst[idx]
            curve_dict[f"{prefix}_tpr_comm"] = res_now[idx, :, 0, 2]
            curve_dict[f"{prefix}_fpr_comm"] = res_now[idx, :, 0, 3]        
            curve_dict[f"{prefix}_tpr_diff"] = res_now[idx, :, 1, 2]
            curve_dict[f"{prefix}_fpr_diff"] = res_now[idx, :, 1, 3]
            curve_dict["l1_lst"] = l1_lst
            curve_dict["l2_lst"] = l2_lst
    
    return curve_dict


def draw_f1(curve_dict, x_type="f1_diff", scale_free=2):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    l1_lst = curve_dict["l1_lst"]
    ax.plot(l1_lst, curve_dict[f"random_ddn_{x_type}"], color="tab:blue", label="Random, DDN")
    ax.plot(l1_lst, curve_dict[f"random_jgl_{x_type}"], color="tab:blue", linestyle='--', label="Random, JGL")
    ax.plot(l1_lst, curve_dict[f"cluster_ddn_{x_type}"], color="tab:orange", label="Cluster, DDN")
    ax.plot(l1_lst, curve_dict[f"cluster_jgl_{x_type}"], color="tab:orange", linestyle='--', label="Cluster, JGL")
    ax.plot(l1_lst, curve_dict[f"hub_ddn_{x_type}"], color="#3A3A3A", label="Hub, DDN")
    ax.plot(l1_lst, curve_dict[f"hub_jgl_{x_type}"], color="#3A3A3A", linestyle='--', label="Hub, JGL")
    ax.plot(l1_lst, curve_dict[f"scale_free_{scale_free}_ddn_{x_type}"], color="#5CCEB3", label="Scale-free, DDN")
    ax.plot(l1_lst, curve_dict[f"scale_free_{scale_free}_jgl_{x_type}"], color="#5CCEB3", linestyle='--', label="Scale-free, JGL")
    ax.set_xlabel("$\lambda_1$")
    ax.set_ylabel("F1 score")
    fig.legend(loc="outside right center")
    return fig, ax


def draw_roc(curve_dict, x_type="fpr_comm", y_type="tpr_comm", scale_free=2):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(curve_dict[f"random_ddn_{x_type}"], curve_dict[f"random_ddn_{y_type}"], color="tab:blue", label="Random, DDN")
    ax.plot(curve_dict[f"random_jgl_{x_type}"], curve_dict[f"random_jgl_{y_type}"], color="tab:blue", linestyle='--', label="Random, JGL")
    ax.plot(curve_dict[f"cluster_ddn_{x_type}"], curve_dict[f"cluster_ddn_{y_type}"], color="tab:orange", label="Cluster, DDN")
    ax.plot(curve_dict[f"cluster_jgl_{x_type}"], curve_dict[f"cluster_jgl_{y_type}"], color="tab:orange", linestyle='--', label="Cluster, JGL")
    ax.plot(curve_dict[f"hub_ddn_{x_type}"], curve_dict[f"hub_ddn_{y_type}"], color="#3A3A3A", label="Hub, DDN")
    ax.plot(curve_dict[f"hub_jgl_{x_type}"], curve_dict[f"hub_jgl_{y_type}"], color="#3A3A3A", linestyle='--', label="Hub, JGL")
    ax.plot(curve_dict[f"scale_free_{scale_free}_ddn_{x_type}"], curve_dict[f"scale_free_{scale_free}_ddn_{y_type}"], color="#5CCEB3", label="Scale-free, DDN")
    ax.plot(curve_dict[f"scale_free_{scale_free}_jgl_{x_type}"], curve_dict[f"scale_free_{scale_free}_jgl_{y_type}"], color="#5CCEB3", linestyle='--', label="Scale-free, JGL")
    ax.set_xlim([-0.001,0.02])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    fig.legend(loc="outside right center")
    return fig, ax


def draw_f1_feature_num(curve_dict, x_type="f1_diff"):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    l1_lst = curve_dict["l1_lst"]
    ax.plot(l1_lst, curve_dict[f"scale_free_4_ddn_{x_type}"], color="tab:blue", label="100 nodes, DDN")
    ax.plot(l1_lst, curve_dict[f"scale_free_4_jgl_{x_type}"], color="tab:blue", linestyle='--', label="100 nodes, JGL")
    ax.plot(l1_lst, curve_dict[f"scale_free_8_200_ddn_{x_type}"], color="tab:orange", label="200 nodes, DDN")
    ax.plot(l1_lst, curve_dict[f"scale_free_8_200_jgl_{x_type}"], color="tab:orange", linestyle='--', label="200 nodes, JGL")
    ax.plot(l1_lst, curve_dict[f"scale_free_16_400_ddn_{x_type}"], color="#3A3A3A", label="400 nodes, DDN")
    ax.plot(l1_lst, curve_dict[f"scale_free_16_400_jgl_{x_type}"], color="#3A3A3A", linestyle='--', label="400 nodes, JGL")
    ax.set_xlabel("$\lambda_1$")
    ax.set_ylabel("F1 score")
    fig.legend(loc="outside right center")
    return fig, ax


def draw_roc_feature_num(curve_dict, x_type="fpr_comm", y_type="tpr_comm"):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(curve_dict[f"scale_free_4_ddn_{x_type}"], curve_dict[f"scale_free_4_ddn_{y_type}"], color="tab:blue", label="100 nodes, DDN")
    ax.plot(curve_dict[f"scale_free_4_jgl_{x_type}"], curve_dict[f"scale_free_4_jgl_{y_type}"], color="tab:blue", linestyle='--', label="100 nodes, JGL")
    ax.plot(curve_dict[f"scale_free_8_200_ddn_{x_type}"], curve_dict[f"scale_free_8_200_ddn_{y_type}"], color="tab:orange", label="200 nodes, DDN")
    ax.plot(curve_dict[f"scale_free_8_200_jgl_{x_type}"], curve_dict[f"scale_free_8_200_jgl_{y_type}"], color="tab:orange", linestyle='--', label="200 nodes, JGL")
    ax.plot(curve_dict[f"scale_free_16_400_ddn_{x_type}"], curve_dict[f"scale_free_16_400_ddn_{y_type}"], color="#3A3A3A", label="400 nodes, DDN")
    ax.plot(curve_dict[f"scale_free_16_400_jgl_{x_type}"], curve_dict[f"scale_free_16_400_jgl_{y_type}"], color="#3A3A3A", linestyle='--', label="400 nodes, JGL")
    ax.set_xlim([-0.001,0.02])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    fig.legend(loc="outside right center")
    return fig, ax


def draw_f1_balance(curve_dict, x_type="f1_diff"):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    l1_lst = curve_dict["l1_lst"]
    ax.plot(l1_lst, curve_dict[f"balanced_ddn_{x_type}"], color="tab:blue", label="Balanced, DDN")
    ax.plot(l1_lst, curve_dict[f"balanced_jgl_{x_type}"], color="tab:blue", linestyle='--', label="Balanced, JGL")
    ax.plot(l1_lst, curve_dict[f"balanced_not_ddn_{x_type}"], color="tab:orange", label="Unbalanced, DDN")
    ax.plot(l1_lst, curve_dict[f"balanced_not_jgl_{x_type}"], color="tab:orange", linestyle='--', label="Unbalanced, JGL")
    ax.set_xlabel("$\lambda_1$")
    ax.set_ylabel("F1 score")
    fig.legend(loc="outside right center")
    return fig, ax


def draw_roc_balance(curve_dict, x_type="fpr_comm", y_type="tpr_comm"):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(curve_dict[f"balanced_ddn_{x_type}"], curve_dict[f"balanced_ddn_{y_type}"], color="tab:blue", label="Balanced, DDN")
    ax.plot(curve_dict[f"balanced_jgl_{x_type}"], curve_dict[f"balanced_jgl_{y_type}"], color="tab:blue", linestyle='--', label="Balanced, JGL")
    ax.plot(curve_dict[f"balanced_not_ddn_{x_type}"], curve_dict[f"balanced_not_ddn_{y_type}"], color="tab:orange", label="Unbalanced, DDN")
    ax.plot(curve_dict[f"balanced_not_jgl_{x_type}"], curve_dict[f"balanced_not_jgl_{y_type}"], color="tab:orange", linestyle='--', label="Unbalanced, JGL")
    ax.set_xlim([-0.001,0.02])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    fig.legend(loc="outside right center")
    return fig, ax

