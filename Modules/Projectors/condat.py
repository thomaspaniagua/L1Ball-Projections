import torch
from Utils import projector_helpers

def condat_simplex(y, s=1):
    # Proposed algorithm form http://www.optimization-online.org/DB_FILE/2014/08/4498.pdf 
    # Pg 4
    
    # 1.
    v = [y[0]]
    v_hat = []
    p = y[0] - s
    N = y.shape[0]

    # 2.
    for n in range(1, N):
        if y[n] > p:
            p = p + (y[n] - p) / (len(v)+1) # 2.1

            if p > y[n] - s: # 2.2
                v.append(y[n])
            else:
                # 2.3
                v_hat.extend(v)
                v = [y[n]]
                p = y[n] - 1

    # 3
    if len(v_hat) > 0:
        for y_ in v_hat:
            #3.1
            if y_ > p:
                v.append(y_)
                p = p + (y_ - p)/(len(v))

    first = False            
    v_len = len(v)
    # 4
    while v_len != len(v) or not first:
        v_len = len(v)
        to_remove = []
        for i, y_ in enumerate(v):
            if y_ < p:
                to_remove.append(i)
                p = p + (p - y_)/(len(v)-len(to_remove))

        for index in sorted(to_remove, reverse=True):
            del v[index]

        first = True

    # 5
    tau = p
    K = len(v)
    return (y-tau).clamp(min=0)

condat_l1 = \
projector_helpers.l1_from_simplex(projector_helpers.list_project(condat_simplex))