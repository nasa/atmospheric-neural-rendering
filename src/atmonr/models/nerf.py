import torch
import torch.nn as nn
import torch.nn.functional as F


class AtmoNeRF(nn.Module):
    """Atmospheric Neural Radiance Field."""

    def __init__(
        self,
        pos_channels: int,
        dir_channels: int,
        out_channels: int,
        volume_channels: int,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize an AtmoNeRF.

        Args:
            pos_channels: Number of positional channels in the input.
            dir_channels: Number of directional channels in the input.
            out_channels: Number of spectral channels in the output.
            volume_channels: Number of volume densities to model.
            hidden_dim: Size of the hidden layers.
        """
        super().__init__()
        self.pos_channels = pos_channels
        self.dir_channels = dir_channels
        self.out_channels = out_channels
        self.volume_channels = volume_channels
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.pos_channels, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc5 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc6 = nn.Linear(self.hidden_dim + self.pos_channels, self.hidden_dim)
        self.fc7 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc8 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc9 = nn.Linear(self.hidden_dim, self.hidden_dim + self.volume_channels)
        self.fc10 = nn.Linear(self.hidden_dim + self.dir_channels, self.hidden_dim // 2)
        self.fc11 = nn.Linear(self.hidden_dim // 2, self.out_channels)

        for i in range(1, 12):
            nn.init.kaiming_normal_(getattr(self, "fc" + str(i)).weight, mode="fan_out")

    def forward_pos_only(self, x_pos: torch.Tensor):
        """The first part of the forward pass, up until volume density is computed.

        Args:
            x_pos: Input, containing only the position information.

        Returns:
            x: Intermediate tensor after the first part of the forward pass.
            sigma: The volume density predicted by the model.
        """
        x = F.relu(self.fc1(x_pos))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = torch.cat([x, x_pos], dim=1)  # skip connection
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        sigma = x[:, self.hidden_dim :]
        # add Gaussian noise to volume density
        if self.training:
            sigma += torch.randn(sigma.shape, device=sigma.device)
        sigma = F.relu(sigma)
        return x, sigma

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input to the NeRF, containing both position and direction.

        Returns:
            color: Color output.
            sigma: Volume density output.
        """
        x_pos, d = (
            x[:, : self.pos_channels],
            x[:, self.pos_channels :],
        )  # split up into x (position) and d (direction)
        x, sigma = self.forward_pos_only(x_pos)
        x = torch.cat([x[:, : self.hidden_dim], d], dim=1)
        x = F.relu(self.fc10(x))
        color = F.sigmoid(self.fc11(x))
        return color, sigma


def get_model(
    hidden_dim: int,
    N_lambda: int,
    L_x: int | list[int],
    L_d: int,
    include_height: bool,
) -> tuple[AtmoNeRF, AtmoNeRF]:
    """Get the appropriate model, given script arguments.

    Args:
        hidden_dim: Number of
        N_lambda: Number of spectral channels.
        L_x: Number of encoding frequencies describing positions.
        L_d: Number of encoding frequencies describing directions.
        include_height: Whether heights are included as input to the NeRF.

    Returns:
        nerf_c: Coarse neural radiance field.
        nerf_f: Fine neural radiance field.
    """
    # number of positional channels
    if isinstance(L_x, int):
        pos_channels = L_x * 6
        if include_height:
            pos_channels += L_x * 2
    elif isinstance(L_x, list):
        assert (include_height and len(L_x) == 4) or (
            not include_height and len(L_x) == 3
        )
        pos_channels = sum(L_x) * 2
    # number of directional channels
    dir_channels = L_d * 6
    # initialize two models: coarse model provides sampling bias for the fine model
    # (see NeRF paper 5.2)
    nerf_c = AtmoNeRF(
        pos_channels=pos_channels,
        dir_channels=dir_channels,
        out_channels=N_lambda,
        volume_channels=1,
        hidden_dim=hidden_dim,
    )
    nerf_f = AtmoNeRF(
        pos_channels=pos_channels,
        dir_channels=dir_channels,
        out_channels=N_lambda,
        volume_channels=N_lambda,
        hidden_dim=hidden_dim,
    )
    return nerf_c, nerf_f
