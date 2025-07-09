import torch


def positional_encoding(pts: torch.Tensor, L: int | list[int]) -> torch.Tensor:
    """Positional encoding trick used to enable MLPs to fit high-frequency signals.

    Args:
        p: Points to encode.
        L: Number of frequencies in the position encoding.

    Returns:
        pts_enc: Position-encoded points.
    """
    if isinstance(L, int):
        pts = torch.reshape(pts, (-1, pts.shape[-1]))[..., None, None]
        L_ls = torch.linspace(0, L - 1, steps=L, device=pts.device)
        L_ls = torch.stack([L_ls, L_ls], dim=1)
        pts = (2**L_ls * torch.pi)[None, None] * pts
        pts = torch.stack([torch.sin(pts[..., 0]), torch.cos(pts[..., 1])], dim=-1)
        pts_enc = torch.reshape(pts, (pts.shape[0], pts.shape[1], -1))
    elif isinstance(L, list):
        pts_enc_list = []
        for i, num_freqs in enumerate(L):
            l_ls = torch.linspace(0, num_freqs - 1, steps=num_freqs, device=pts.device)
            x = (2**l_ls * torch.pi)[..., None, :] * pts[..., i, None]
            pts_enc_list.append(torch.cat([torch.sin(x), torch.cos(x)], dim=-1))
        pts_enc = torch.cat(pts_enc_list, dim=-1)
    return pts_enc
