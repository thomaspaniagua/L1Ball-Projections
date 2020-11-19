import torch
from Utils import projector_helpers

from Utils.quicksort import quickSort as quick_sort

def slow_sort(x):
    sorted_list = []
    for i in range(x.shape[0]):
        row_list = x[i].tolist()
        #row_list.sort(reverse=True)
        quick_sort(row_list, 0, len(row_list)-1)
        row_list = list(reversed(row_list))
        sorted_list.append(row_list)

    return torch.tensor(sorted_list)

def project_simplex(x: torch.Tensor, s=1, use_slow_sort=True):
    # Adapted from https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
    # Only top N numbers will contribute to final L1 norm
    n = x.shape[1]
    
    if use_slow_sort:
        u = slow_sort(x)
    else:
        u, _ = x.sort(dim=1, descending=True)
    
    cssv = u.cumsum(dim=1)

    rho = u * torch.arange(1, n+1)[None]
    rho = rho > (cssv - s)

    rho = rho.split(dim=0, split_size=1)
    rho = list(map(lambda x: x[0].nonzero()[-1][0], rho))

    offset = list(map(lambda p: cssv[p[0], p[1]], enumerate(rho)))
    offset = torch.stack(offset)

    rho = torch.stack(rho)

    theta = (offset - s)/(rho+1)
    
    w = x - theta[:, None]
    w = w.clamp_(min=0)

    return w

def project_simplex_1(x):
    return project_simplex(x[None])[0]

def project_l1_ball(x: torch.Tensor, s=1):
    u = torch.abs(x)
    u_proj = project_simplex(u)

    x_proj = torch.sign(x)*u_proj

    return x_proj

project_l1_ball_serial = \
projector_helpers.l1_from_simplex(projector_helpers.list_project(project_simplex_1))