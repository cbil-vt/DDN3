import numpy as np
from ddn3_extra import tools_r as rr


def make_multi_scale_free(n_node_lst, verbose=False):
    n_node_lst = np.array(n_node_lst).astype(int)
    n_node_total = int(np.sum(n_node_lst))
    omega_all = np.zeros((n_node_total, n_node_total))
    idx = 0
    for n_node in n_node_lst:
        huge_graph = rr.huge.huge_generator(
            n=10,
            d=int(n_node),
            graph="scale-free",
            v=0.5,
            u=0.2,
            verbose=verbose,
        )
        omega = np.array(huge_graph.rx2("omega"))
        idx1 = idx + n_node
        omega_all[idx:idx1, idx:idx1] = omega
        idx = idx + n_node
    return omega_all


def huge_omega(
    n_node,
    ratio_diff=0.25,
    graph_type="random",
    thr=0.0001,
    ratio=0.9,
    verbose=False,
    n_group=5,
):
    if graph_type == "random":
        huge_graph = rr.huge.huge_generator(
            n=10,
            d=n_node,
            graph=graph_type,
            v=0.5,
            u=0.2,
            prob=2 / n_node,
            verbose=verbose,
        )
        omega = np.array(huge_graph.rx2("omega"))
    elif graph_type == "scale-free":
        huge_graph = rr.huge.huge_generator(
            n=10,
            d=n_node,
            graph=graph_type,
            v=0.5,
            u=0.2,
            verbose=verbose,
        )
        omega = np.array(huge_graph.rx2("omega"))
    elif graph_type == "scale-free-multi":
        node0 = int(n_node / n_group)
        n_node_lst = np.zeros(n_group) + node0
        omega = make_multi_scale_free(n_node_lst)
    elif (graph_type == "hub") or (graph_type == "cluster"):
        huge_graph = rr.huge.huge_generator(
            n=10,
            d=n_node,
            graph=graph_type,
            g=n_group,
            v=0.5,
            u=0.2,
            verbose=verbose,
        )
        omega = np.array(huge_graph.rx2("omega"))
    else:
        raise ("Not implemented.")

    omega1, omega2 = make_two_from_one(
        omega, ratio_diff=ratio_diff, ratio=ratio, thr=thr, verbose=verbose
    )

    return omega, omega1, omega2


def make_two_from_one(
    omega, dep_allow=None, ratio_diff=0.25, ratio=0.9, thr=1e-4, verbose=False
):
    n_node = len(omega)

    n_edge = round((np.sum(np.abs(omega) > thr) - n_node) / 2)

    msk = np.tril(np.ones(n_node), k=-1)
    omega_lower = np.copy(omega)
    omega_lower[msk == 0] = 100.0
    if dep_allow is not None:
        dep_in = dep_allow + dep_allow.T
        omega_lower[dep_in == 0] = 100.0
    idx_zero = np.where(np.abs(omega_lower) < thr)

    omega_ng = np.copy(np.abs(omega))
    omega_ng[np.arange(n_node), np.arange(n_node)] = 0
    fill_value = np.mean(omega_ng[np.abs(omega_ng) > thr]) * ratio
    if verbose:
        print("Non-diagonal values ", fill_value)

    n_diff = round(n_edge * ratio_diff)
    if verbose:
        print("Common edegs and diff edges", n_edge, n_diff)

    omega1 = np.copy(omega)
    omega2 = np.copy(omega)

    # try to generate two conditions
    if n_diff > 0:
        # eig_min = -1
        nn = 0
        for nn in range(100):
            idx_chg = np.random.choice(len(idx_zero[0]), n_diff * 2, replace=False)
            idx_chg1 = idx_chg[:n_diff]
            idx_chg2 = idx_chg[n_diff:]

            n1 = len(idx_chg1)
            fill1 = fill_value * np.sign((np.random.rand(n1) - 0.5))
            n2 = len(idx_chg2)
            fill2 = fill_value * np.sign((np.random.rand(n2) - 0.5))

            diff1 = np.zeros((n_node, n_node))
            diff1[idx_zero[0][idx_chg1], idx_zero[1][idx_chg1]] = fill1
            diff1 = diff1 + diff1.T
            omega1 = omega + diff1

            diff2 = np.zeros((n_node, n_node))
            diff2[idx_zero[0][idx_chg2], idx_zero[1][idx_chg2]] = fill2
            diff2 = diff2 + diff2.T
            omega2 = omega + diff2

            eig1, _ = np.linalg.eig(omega1)
            eig2, _ = np.linalg.eig(omega2)
            if verbose:
                print("Smallest eigen values ", np.min(eig1), np.min(eig2))

            eig_min = np.min([np.min(eig1), np.min(eig2)])
            if eig_min > 0:
                break

        if nn >= 99:
            raise ("Failed")

    return omega1, omega2


def make_two_from_one_by_removing(
    omega_in,
    ratio_diff=0.25,
    ratio_diag=1.0,
    thr=1e-4,
    verbose=False,
):
    n_node = len(omega_in)

    omega = np.copy(omega_in)
    omega[np.arange(n_node), np.arange(n_node)] *= ratio_diag

    omega_ng = np.copy(omega)
    omega_ng[np.arange(n_node), np.arange(n_node)] = 0.0

    n_edge = round((np.sum(np.abs(omega_ng) > thr)) / 2)
    idx_nonzero = np.where(np.abs(omega_ng) > thr)
    n_diff = round(n_edge * ratio_diff)
    omega1 = np.copy(omega)
    omega2 = np.copy(omega)

    # try to generate two conditions
    if n_diff > 0:
        for nn in range(100):
            idx_chg = np.random.choice(len(idx_nonzero[0]), n_diff * 2, replace=False)
            idx_chg1 = idx_chg[:n_diff]
            idx_chg2 = idx_chg[n_diff:]

            omega1 = np.copy(omega)
            omega1[idx_nonzero[0][idx_chg1], idx_nonzero[1][idx_chg1]] = 0
            omega1[idx_nonzero[1][idx_chg1], idx_nonzero[0][idx_chg1]] = 0

            omega2 = np.copy(omega)
            omega2[idx_nonzero[0][idx_chg2], idx_nonzero[1][idx_chg2]] = 0
            omega2[idx_nonzero[1][idx_chg2], idx_nonzero[0][idx_chg2]] = 0

            eig1, _ = np.linalg.eig(omega1)
            eig2, _ = np.linalg.eig(omega2)

            if verbose:
                print("Smallest eigen values ", np.min(eig1), np.min(eig2))

            eig_min = np.min([np.min(eig1), np.min(eig2)])
            if eig_min > 0:
                break
            if nn >= 99:
                raise ("Failed")

    return omega1, omega2
