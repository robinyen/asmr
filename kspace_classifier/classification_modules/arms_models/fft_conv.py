from functools import partial
from typing import Tuple, Union, Iterable

from functools import partial
from typing import Tuple, Union, Iterable

import torch
from torch import nn, Tensor
import torch.nn.functional as f

from torch.fft import fftn, ifftn


def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    scalar_matmul = partial(torch.einsum, "agc..., gbc... -> agb...")
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    # Compute the real and imaginary parts independently, then manually insert them
    # into the output Tensor.  This is fairly hacky but necessary for PyTorch 1.7.0,
    # because Autograd is not enabled for complex matrix operations yet.  Not exactly
    # idiomatic PyTorch code, but it should work for all future versions (>= 1.7.0).
    real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
    imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])


def fft_conv(signal: Tensor, kernel: Tensor, bias: Tensor = None,
             groups: int = 1) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.

    Returns:
        (Tensor) Convolved tensor
    """

    signal_ = signal

    kernel_padding = [
        pad
        for i in reversed(range(2, signal_.ndim))
        for pad in [0, signal_.size(i) - kernel.size(i)]
    ]
    padded_kernel = f.pad(kernel, kernel_padding)
    kernel_fr = fftn(padded_kernel, dim=tuple(range(2, signal.ndim)))

    kernel_fr.imag *= -1

    output_fr = complex_matmul(signal_, kernel_fr, groups=groups)

    output = ifftn(output_fr, dim=(-2, -1))


    return output, kernel_fr


class _FFTConv(nn.Module):

    """Base class for PyTorch FFT convolution layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Iterable[int]],
        padding: Union[int, Iterable[int]] = None,
        stride: Union[int, Iterable[int]] = None,
        groups: int = 1,
        bias: bool = False,
        ndim: int = 2,
    ):
        """
        Args:
            in_channels: (int) Number of channels in input tensors
            out_channels: (int) Number of channels in output tensors
            kernel_size: (Union[int, Iterable[int]) Square radius of the kernel
            padding: (Union[int, Iterable[int]) Number of zero samples to pad the
                input on the last dimension.
            stride: (Union[int, Iterable[int]) Stride size for computing output values.
            bias: (bool) If True, includes bias, which is added after convolution
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.use_bias = bias

        w = torch.randn(out_channels, in_channels // groups,
                        kernel_size, kernel_size).to(torch.complex64)
        w = w / (out_channels * in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.view_as_real(w))

        bias_w = torch.randn(out_channels,).to(torch.complex64)
        self.bias = nn.Parameter(torch.view_as_real(bias_w)) if bias else None

    def forward(self, signal):
        weights = torch.view_as_complex(self.weight)
        bias = torch.view_as_complex(self.bias) if self.use_bias else None
        output, kernel_fr = fft_conv(
            signal,
            weights,
            bias=bias,
            groups=self.groups)

        self.kernel_fr = kernel_fr

        return output


FFTConv1d = partial(_FFTConv, ndim=1)
FFTConv2d = partial(_FFTConv, ndim=2)
FFTConv3d = partial(_FFTConv, ndim=3)