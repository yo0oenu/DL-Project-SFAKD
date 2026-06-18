import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Proposed local branch: spatial attention (CBAM-style)
# ---------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    """
    Computes a spatial attention map from channel-wise max and mean statistics,
    then re-weights the input feature map accordingly.

    This replaces FAM-KD's original local branch (a single 1x1 convolution),
    which has no spatial receptive field and therefore cannot emphasize
    spatially important regions.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
        )

    def spatial_attn(self, x: torch.Tensor) -> torch.Tensor:
        # channel-wise max and mean, concatenated along the channel dim -> (N, 2, H, W)
        max_pool = torch.max(x, dim=1)[0].unsqueeze(1)
        mean_pool = torch.mean(x, dim=1).unsqueeze(1)
        return torch.cat((max_pool, mean_pool), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.spatial_attn(x)
        attn = torch.sigmoid(self.spatial_conv(attn))
        return attn * x


# ---------------------------------------------------------------------------
# FFT shift helpers (needed for the frequency / global branch)
# ---------------------------------------------------------------------------
def roll_n(x: torch.Tensor, axis: int, n: int) -> torch.Tensor:
    f_idx = tuple(
        slice(None, None, None) if i != axis else slice(0, n, None)
        for i in range(x.dim())
    )
    b_idx = tuple(
        slice(None, None, None) if i != axis else slice(n, None, None)
        for i in range(x.dim())
    )
    front = x[f_idx]
    back = x[b_idx]
    return torch.cat([back, front], axis)


def batch_fftshift2d(x: torch.Tensor) -> torch.Tensor:
    real, imag = x.real, x.imag
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim) // 2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim = 2 (real & imag)


def batch_ifftshift2d(x: torch.Tensor) -> torch.Tensor:
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)
    return torch.stack((real, imag), -1)


def init_rate_half(tensor: torch.Tensor) -> None:
    if tensor is not None:
        tensor.data.fill_(0.5)


# ---------------------------------------------------------------------------
# Proposed fused module: global (frequency) branch + local (spatial) branch
# ---------------------------------------------------------------------------
class FAM_Module(nn.Module):
    """
    Two-branch attention module that fuses:
      - a global branch: learnable filtering in the Fourier domain (frequency
        attention, following FAM-KD), and
      - a local branch: the proposed spatial-attention block above,

    using two learnable scalars (rate1, rate2) so the network can adaptively
    balance each branch's contribution and avoid representational collapse
    into a single domain.
    """

    def __init__(self, in_channels: int, out_channels: int, shapes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapes = shapes  # spatial size (H == W assumed) of the feature map

        self.rate1 = nn.Parameter(torch.Tensor(1))
        self.rate2 = nn.Parameter(torch.Tensor(1))

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, shapes, shapes, dtype=torch.cfloat)
        )

        self.spatial_attention = SpatialAttention(kernel_size=3)
        self.spatial_non_linear = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        init_rate_half(self.rate1)
        init_rate_half(self.rate2)

    def compl_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x, cuton: float = 0.1) -> torch.Tensor:
        # allow (x, cuton) tuples for convenience, mirroring the original API
        if isinstance(x, tuple):
            x, cuton = x

        # --- global branch: frequency-domain attention ---
        x_ft = torch.fft.fft2(x, norm="ortho")
        out_ft = self.compl_mul2d(x_ft, self.weights1)
        batch_shift = batch_fftshift2d(out_ft)

        h, w = batch_shift.shape[2:4]
        cy, cx = h // 2, w // 2
        rh, rw = int(cuton * cy), int(cuton * cx)
        # high-pass filter: zero out the low-frequency (center) region
        batch_shift[:, :, cy - rh:cy + rh, cx - rw:cx + rw, :] = 0

        out_ft = batch_ifftshift2d(batch_shift)
        out_ft = torch.view_as_complex(out_ft)
        global_out = torch.fft.ifft2(
            out_ft, s=(x.size(-2), x.size(-1)), norm="ortho"
        ).real

        # --- local branch: proposed spatial attention ---
        local_out = self.spatial_non_linear(self.spatial_attention(x))

        # --- adaptive fusion via learnable scalars ---
        return self.rate1 * global_out + self.rate2 * local_out


# ---------------------------------------------------------------------------
# Forward-pass sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    batch_size, channels, spatial_size = 2, 64, 32

    dummy_input = torch.randn(batch_size, channels, spatial_size, spatial_size)

    print("Testing SpatialAttention (proposed local branch)...")
    spatial_attn = SpatialAttention(kernel_size=3)
    spatial_out = spatial_attn(dummy_input)
    print(f"  input  shape: {tuple(dummy_input.shape)}")
    print(f"  output shape: {tuple(spatial_out.shape)}")
    assert spatial_out.shape == dummy_input.shape

    print("\nTesting FAM_Module (global frequency branch + proposed local branch)...")
    fam_module = FAM_Module(
        in_channels=channels, out_channels=channels, shapes=spatial_size
    )
    fam_out = fam_module(dummy_input)
    print(f"  input  shape: {tuple(dummy_input.shape)}")
    print(f"  output shape: {tuple(fam_out.shape)}")
    assert fam_out.shape == (batch_size, channels, spatial_size, spatial_size)

    print("\nAll forward passes succeeded.")
