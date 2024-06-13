import numpy as np
from ddn3 import ddn, performance, simulation
from ddn3_extra import tools_r as rr
from ddn3_extra import simulation_r


def scan_ddn(
    dat1, dat2, lambda1_rg=np.arange(0.02, 1.0, 0.02), lambda2=0.1, mthd="resi"
):
    t1_lst = []
    t2_lst = []
    for i, lamb in enumerate(lambda1_rg):
        out_ddn = ddn.ddn(
            dat1, dat2, lambda1=lamb, lambda2=lambda2, threshold=1e-5, mthd=mthd
        )
        t1_lst.append(np.copy(out_ddn[0]))
        t2_lst.append(np.copy(out_ddn[1]))
    return t1_lst, t2_lst


def scan_jgl(
    dat1, dat2, lambda1_rg=np.arange(0.02, 1.0, 0.02), lambda2=0.1, weights="equal"
):
    dat1x = rr.np2r2d(dat1)
    dat2x = rr.np2r2d(dat2)
    t1_lst = []
    t2_lst = []
    lamb2x = lambda2
    for _, lamb in enumerate(lambda1_rg):
        lamb1x = rr.robjects.FloatVector([lamb])[0]
        out_jgl = rr.jgl.JGL(
            [dat1x, dat2x],
            lambda1=lamb1x,
            lambda2=lamb2x,
            return_whole_theta=True,
            weights=weights,
        )
        theta = out_jgl.rx2("theta")
        t1_lst.append(np.copy(np.array(theta[0])))
        t2_lst.append(np.copy(np.array(theta[1])))
    return t1_lst, t2_lst


def scan_dingo(dat1, dat2, B=5, cores=5, rho=-1):
    n_node = dat1.shape[1]
    dat12x = rr.np2r2d(np.vstack((dat1, dat2)))
    x = rr.np2r1d(np.concatenate((np.zeros(len(dat1)) + 1, np.zeros(len(dat2)) - 1)))
    n = [str(i) for i in range(n_node)]
    dat12x.colnames = rr.robjects.r["c"](*n)

    if rho < 0:
        out = rr.dingo.dingo(dat12x, x, diff_score=True, B=B, cores=cores)
    else:
        out = rr.dingo.dingo(
            dat12x, x, diff_score=True, B=B, cores=cores, rhoarray=float(rho)
        )
    rho1 = out.rx2("rho")
    print("rho is ", rho1)

    comm_theta = np.array(out.rx2("P"))
    theta1 = np.array(out.rx2("R1"))
    theta2 = np.array(out.rx2("R2"))
    gene_pairs = np.array(out.rx2("genepair")).astype(int)
    diff_score = np.array(out.rx2("diff.score"))

    diff_est = np.zeros((n_node, n_node))
    diff_est[gene_pairs[0], gene_pairs[1]] = diff_score
    diff_est = diff_est + diff_est.T

    r1mat = np.zeros((n_node, n_node))
    r1mat[gene_pairs[0], gene_pairs[1]] = theta1
    r1mat = r1mat + r1mat.T

    r2mat = np.zeros((n_node, n_node))
    r2mat[gene_pairs[0], gene_pairs[1]] = theta2
    r2mat = r2mat + r2mat.T

    return r1mat, r2mat, comm_theta, diff_est, out


def scan_wrap_ddn_jgl(
    l1_lst,
    l2_lst,
    n1,
    n2,
    n_node=100,
    ratio_diff=0.25,
    graph_type="scale-free",
    n_group=5,
    weights_jgl="equal",
):
    # method number, l2 number, l1 number, comm/diff, metric number
    res = np.zeros((2, len(l2_lst), len(l1_lst), 2, 5))

    # new graph for each repeat
    _, omega1, omega2 = simulation_r.huge_omega(
        n_node=n_node, ratio_diff=ratio_diff, graph_type=graph_type, n_group=n_group
    )
    g1_cov, g2_cov, comm_gt, diff_gt = simulation.prep_sim_from_two_omega(
        omega1, omega2
    )
    dat1, dat2 = simulation.gen_sample_two_conditions(g1_cov, g2_cov, n1, n2)

    for j, l2 in enumerate(l2_lst):
        # DDN
        t1_lst_ddn, t2_lst_ddn = scan_ddn(dat1, dat2, lambda1_rg=l1_lst, lambda2=l2)
        res_comm_ddn, res_diff_ddn = performance.scan_error_measure(
            t1_lst_ddn, t2_lst_ddn, comm_gt, diff_gt
        )
        res[0, j, :, 0, :] = res_comm_ddn
        res[0, j, :, 1, :] = res_diff_ddn

        # JGL
        t1_lst_jgl, t2_lst_jgl = scan_jgl(
            dat1, dat2, lambda1_rg=l1_lst, lambda2=float(l2), weights=weights_jgl
        )
        res_comm_jgl, res_diff_jgl = performance.scan_error_measure(
            t1_lst_jgl, t2_lst_jgl, comm_gt, diff_gt
        )
        res[1, j, :, 0, :] = res_comm_jgl
        res[1, j, :, 1, :] = res_diff_jgl

    print("Finished this repeat")

    return res
