import torch
from Utils import projector_helpers

from Utils.quicksort import quickSort as quick_sort

def descent_simplex_batch(y, s=1):
    tau = (y.max(dim=1)[0] )
    tau = tau.unsqueeze(-1)
    tau = torch.where(tau - s > 0, tau - s, tau)
    step_size = torch.ones_like(tau)

    i=0
    norms = y.norm(dim=1, p=1)
    norms_diff = norms - s
    while ((norms_diff).abs()>1e-7).any():
        y_ = (y-tau).clamp(min=0)
        norms = y_.norm(dim=1, p=1)
        norms_diff_ = norms - s

        slower = torch.sign(norms_diff_) != torch.sign(norms_diff)    
        step_size = torch.where(slower[:, None], step_size * 0.5, step_size)
        step = norms_diff_.unsqueeze(-1)

        step *= step_size
        tau += step

        norms_diff = norms_diff_

    return y_


def descent_simplex_single(y, s=1):
    return descent_simplex_batch(y[None])[0]

descent_l1 = \
projector_helpers.l1_from_simplex(projector_helpers.list_project(descent_simplex_single))