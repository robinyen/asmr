from typing import Tuple, Union
import torch
import numpy as np
from torch.functional import norm


def rss(inv_fft_coil, coil_dim=1):
    """
    Root sum of squares for multi-coil data
    """
    return torch.sqrt((torch.abs(inv_fft_coil) ** 2).sum(coil_dim))


def ifftc(kspace, norm="ortho"):
    """
    Inverse Fourier transform for complex data
    """
    centered_kspace = torch.fft.ifftshift(kspace, dim=(-2, -1))
    data = torch.fft.ifftn(centered_kspace, dim=(-2, -1), norm=norm)

    return torch.fft.fftshift(data, dim=(-2, -1))


def fftc(data, norm="ortho"):
    """
    Fourier transform for complex data
    """
    center_data = torch.fft.ifftshift(data, dim=(-2, -1))
    kspace = torch.fft.fftn(center_data, dim=(-2, -1), norm=norm)

    return torch.fft.fftshift(kspace, dim=(-2, -1))


def crop(data, shape_rows, shape_cols):
    return data[..., shape_rows[0] : shape_rows[1], shape_cols[0] : shape_cols[1]]


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Center crop of data. 
    Args:
        data: torch.Tensor
        shape: Tuple[int, int]
    Returns:
        torch.Tensor        
    """    
    if data.shape[-2:] == shape:
        return data

    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def normalize(
    data: torch.Tensor,
    eps=1e-11,
):
    mean = data.mean().item()
    std = data.std().item()

    return (data - mean) / (std + eps)
