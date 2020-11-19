import torch
from Utils.projector_helpers import l1_from_simplex, list_project

def _michelot_list(y, a=1):
    N = y.shape[0]
    v = y
    p = (y.sum() - a) / N

    while (v>p).sum() != v.shape[0]:
        v = v[v>p]
        p = (v.sum() - a) / v.shape[0]

    tau = p
    K = v.shape[0]

    return (y - tau).clamp(min=0)

michelot = l1_from_simplex(list_project(_michelot_list))