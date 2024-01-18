import numpy as np
from ddn3 import ddn


temp = np.load("./tests/two_part_network.npz")
dat1 = temp['dat1']
dat2 = temp['dat2']
lambda1 = temp['lambda1']
lambda2 = temp['lambda2']
omega1_ref = temp['omega1']
omega2_ref = temp['omega2']

ERR_THRESHOLD = 1e-8


def get_err(omega1, omega2, omega1_ref, omega2_ref):
    d1 = np.linalg.norm(omega1 - omega1_ref)
    d2 = np.linalg.norm(omega2 - omega2_ref)
    return d1, d2


def test_ddn_org():
    omega1, omega2 = ddn.ddn(dat1, dat2, lambda1=lambda1, lambda2=lambda2, mthd='org')
    d1, d2 = get_err(omega1, omega2, omega1_ref, omega2_ref)
    assert(d1 < ERR_THRESHOLD)
    assert(d2 < ERR_THRESHOLD)


def test_ddn_resi():
    omega1, omega2 = ddn.ddn(dat1, dat2, lambda1=lambda1, lambda2=lambda2, mthd='resi')
    d1, d2 = get_err(omega1, omega2, omega1_ref, omega2_ref)
    assert(d1 < ERR_THRESHOLD)
    assert(d2 < ERR_THRESHOLD)


def test_ddn_corr():
    omega1, omega2 = ddn.ddn(dat1, dat2, lambda1=lambda1, lambda2=lambda2, mthd='corr')
    d1, d2 = get_err(omega1, omega2, omega1_ref, omega2_ref)
    assert(d1 < ERR_THRESHOLD)
    assert(d2 < ERR_THRESHOLD)


def test_ddn_org_parallel():
    omega1, omega2 = ddn.ddn_parallel(dat1, dat2, lambda1=lambda1, lambda2=lambda2, mthd='org', n_process=2)
    d1, d2 = get_err(omega1, omega2, omega1_ref, omega2_ref)
    assert(d1 < ERR_THRESHOLD)
    assert(d2 < ERR_THRESHOLD)


def test_ddn_resi_parallel():
    omega1, omega2 = ddn.ddn_parallel(dat1, dat2, lambda1=lambda1, lambda2=lambda2, mthd='resi', n_process=2)
    d1, d2 = get_err(omega1, omega2, omega1_ref, omega2_ref)
    assert(d1 < ERR_THRESHOLD)
    assert(d2 < ERR_THRESHOLD)


def test_ddn_corr_parallel():
    omega1, omega2 = ddn.ddn_parallel(dat1, dat2, lambda1=lambda1, lambda2=lambda2, mthd='corr', n_process=2)
    d1, d2 = get_err(omega1, omega2, omega1_ref, omega2_ref)
    assert(d1 < ERR_THRESHOLD)
    assert(d2 < ERR_THRESHOLD)


if __name__ == "__main__": 
    test_ddn_org()
