from typing import Sequence, Optional, Tuple
from torch import nn
import torch
import numpy as np


class MaskFunction(nn.Module):
    
    def __init__(
        self,
        target_shape: Tuple[int, int],
        k_fraction: float,
        batch_size: int,
        sampled_indices: Tuple[int, int] = (0, -1),
        center_fraction: float = 0.0,
        is_complex: bool = False,
        device: torch.device = torch.device("cpu"),
        coil_type: str = "sc",
    ):
        
        super().__init__()
        self.target_shape = target_shape
        self.center_fraction = center_fraction
        self.k_fraction = k_fraction
        self.is_complex = is_complex
        self.device = device
        self.coil_type = coil_type
        self.batch_size = batch_size

        sampled_indices = (
            (sampled_indices[0], target_shape[1] + sampled_indices[1])
            if sampled_indices[1] < 0
            else sampled_indices
        )
        self.sampled_indices = sampled_indices

    def get_acceleration_mask(
        self, center_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        raise NotImplementedError

    def get_mask(self) -> torch.Tensor:
        
        center_mask = self.get_center_mask()
        acceleration_mask = self.get_acceleration_mask(center_mask)
        center_mask = center_mask.to(acceleration_mask)
        mask = torch.max(center_mask, acceleration_mask).unsqueeze(-1)
        
        return mask

    def forward(self, x: torch.Tensor, batch_size: int = None) -> torch.Tensor:
        
        mask = self.get_mask().squeeze(-1)
        if self.device != x.device:
            mask = mask.to(x.device)
        assert mask.shape == (self.target_shape[1],), mask.shape
        
        
        return x * mask

    def get_center_mask(self) -> torch.Tensor:

        start, end = self.sampled_indices

        len_sampled_indices = end - start + 1
        num_low_frequencies = round(len_sampled_indices * self.center_fraction)

        _, num_cols = self.target_shape
        mask = torch.zeros(num_cols).float()
        if num_low_frequencies == 0:
            return mask

        pad = (len_sampled_indices - num_low_frequencies + 1) // 2
        mask[start + pad : start + pad + num_low_frequencies] = 1
        assert mask.sum() == num_low_frequencies

        return mask


if __name__ == "__main__":
    mask = MaskFunction((5, 7), 0.5, center_fraction=1 / 3, sampled_indices=(4, 6), batch_size=32)
    x = torch.ones(5 * 7).reshape(5, 7)

    cm = mask(x)
    print(cm)
