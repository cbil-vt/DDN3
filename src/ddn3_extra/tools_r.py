import os

os.environ["R_HOME"] = "C:\\Program Files\\R\\R-4.4.0"

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

utils = importr("utils")
base = importr("base")
jgl = importr("JGL")
dingo = importr("iDINGO")
# dingo = importr("iDINGODBG")
glasso = importr("glasso")
huge = importr("huge")


def np2r2d(dat):
    datx = robjects.FloatVector(dat.flatten())
    nrow, ncol = dat.shape
    return robjects.r["matrix"](datx, nrow=nrow, ncol=ncol, byrow=True)


def np2r1d(dat):
    return robjects.FloatVector(dat.flatten())
