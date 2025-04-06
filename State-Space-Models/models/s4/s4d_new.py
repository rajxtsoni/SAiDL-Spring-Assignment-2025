"""
s4d_new.py
A minimal version of S4D that incorporates options for four discretization methods:
  - 'zoh'      : zero-order hold discretization (default)
  - 'bilinear' : Tustin (bilinear) discretization
  - 'dirac'    : discretization with Dirac delta input spikes
  - 'async'    : asynchronous discretization (here, equivalent to zoh)
based on modifications from s4.py and additional techniques.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.nn import DropoutNd  # Make sure this module is available in your repo

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters with configurable discretization."""
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None, disc='zoh'):
        """
        Args:
            d_model (int): Number of independent kernels (H)
            N (int): Full state size (the kernel internally uses N//2 complex conjugate pairs)
            dt_min, dt_max (float): Range for time step initialization.
            lr: Learning rate (optional)
            disc (str): Discretization method to use. Options:
                        'zoh'      -- zero-order hold discretization (default)
                        'bilinear' -- bilinear (Tustin) discretization
                        'dirac'    -- discretization with Dirac delta input spikes
                        'async'    -- asynchronous discretization (here, equivalent to zoh)
        """
        super().__init__()
        self.disc = disc
        H = d_model
        # Initialize log_dt uniformly in log-space between dt_min and dt_max
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        # Initialize complex parameter C with shape (H, N//2)
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        # Initialize A: real part is fixed to 0.5, imag part is scaled by pi times index.
        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        Compute the convolution kernel of length L.
        Returns a tensor of shape (H, L), where H is the number of channels.
        """
        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (H, N//2)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N//2)
        # Compute dtA = A * dt (broadcast dt over last dimension)
        dtA = A * dt.unsqueeze(-1)  # (H, N//2)
        # Create time indices tensor (assume unit time steps)
        t = torch.arange(L, device=A.device)  # (L)

        if self.disc == 'zoh':
            # Zero-Order Hold discretization (default)
            T = dtA.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0)  # (H, N//2, L)
            C_scaled = C * (torch.exp(dtA) - 1.) / A
            K = 2 * torch.einsum('hn, hnl -> hl', C_scaled, torch.exp(T)).real
        elif self.disc == 'bilinear':
            # Bilinear (Tustin) discretization
            C_scaled = C * (1. - dtA/2).reciprocal() * dt.unsqueeze(-1)
            dA = (1. + dtA/2) / (1. - dtA/2)
            E = torch.exp(dA.log().unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0))
            K = 2 * torch.einsum('hn, hnl -> hl', C_scaled, E).real
        elif self.disc == 'dirac':
            # Dirac discretization: use unit scaling (gamma_bar = 1)
            T = dtA.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0)
            # Do not apply the scaling factor (exp(dtA)-1)/A
            K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(T)).real
        elif self.disc == 'async':
            # Asynchronous discretization: here kept equivalent to 'zoh'
            T = dtA.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0)
            C_scaled = C * (torch.exp(dtA) - 1.) / A
            K = 2 * torch.einsum('hn, hnl -> hl', C_scaled, torch.exp(T)).real
        else:
            raise ValueError(f"Unknown discretization method: {self.disc}")

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with an optional learning rate and 0 weight decay."""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        """
        S4D model that uses the S4DKernel for convolution.
        Args:
            d_model: Feature dimension (number of channels H).
            d_state: State size for the SSM kernel.
            dropout: Dropout probability.
            transposed: If True, input and output are assumed to be (B, H, L).
            kernel_args: Extra arguments passed to S4DKernel (e.g. lr, disc).
        """
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        # Learnable D term (skip connection)
        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel with configurable discretization (disc option forwarded via kernel_args)
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Activation and dropout
        self.activation = nn.GELU()
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # Position-wise output transform: 1x1 convolution followed by GLU activation
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):
        """
        Forward pass for S4D.
        Input shape: (B, H, L) if transposed=True; otherwise (B, L, H)
        Returns:
            Tuple (y, None) where y is the output tensor of shape (B, H, L).
        """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM convolution kernel using the chosen discretization method
        k = self.kernel(L=L)  # (H, L)

        # Convolution in frequency domain
        k_f = torch.fft.rfft(k, n=2 * L)  # (H, ?)
        u_f = torch.fft.rfft(u, n=2 * L)   # (B, H, ?)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B, H, L)

        # Add D-term (skip connection)
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, None  # Return dummy state (can be modified as needed)

# Example usage:
if __name__ == "__main__":
    # Create a sample input tensor of shape (batch, channels, length)
    x = torch.randn(8, 64, 128)
    # Instantiate S4D with a selected discretization method: choose among 'zoh', 'bilinear', 'dirac', 'async'
    model = S4D(d_model=64, d_state=64, dropout=0.1, disc='bilinear')
    y, _ = model(x)
    print("Output shape:", y.shape)
