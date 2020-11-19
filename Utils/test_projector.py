import torch

def test_projector(projector, sensitivity=2e-2):
    torch.manual_seed(0) # Generate same samples for all projectors
    for n in range(2,10):
        x = torch.randn(20, n)
        x_proj = projector(x)
        x_proj_l1_norms = x_proj.norm(dim=1, p=1)
        if (x_proj_l1_norms > 1 + sensitivity).any():
            print ("Failed", n)
            return False

    return True
