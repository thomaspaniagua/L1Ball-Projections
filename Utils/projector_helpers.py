import torch

def l1_from_simplex(project_simplex):
    def project_l1_ball(x: torch.Tensor, s=1):
        u = torch.abs(x)
        u_proj = project_simplex(u)

        x_proj = torch.sign(x)*u_proj

        x_norms = x.norm(dim=1, p=1)
        in_ball = x_norms <= s

        in_ball = in_ball[:, None].expand(-1, x.shape[1])

        x_proj = torch.where(in_ball, x, x_proj)

        return x_proj

    return project_l1_ball

def list_project(projector):
    def project(x_batch):
        x_proj = []
        for y in x_batch:
            x_proj.append(projector(y))
        
        return torch.stack(x_proj)

    return project