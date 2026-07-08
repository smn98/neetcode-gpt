import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    mean = round(x.mean().item(), 4)
                    std = round(x.std().item(), 4)
                    dead_frac = round(((x <= 0).all(dim=0)).float().mean().item(), 4)
                    stats.append({'mean': mean, 'std':std, 'dead_fraction': dead_frac})
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        loss_fn = nn.MSELoss()
        model.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        stats = []
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                mean = round(grad.mean().item(), 4)
                std = round(grad.std().item(), 4)
                norm = round(torch.norm(grad).item(), 4)
                stats.append({'mean':mean, 'std':std, 'norm':norm})
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for stat in activation_stats:
            if stat['dead_fraction'] > 0.5:
                return 'dead_neurons'
        
        for stat in gradient_stats:
            if stat['norm'] > 1000:
                return 'exploding_gradients'
        
        if gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'

        for stat in activation_stats:
            if stat['std'] < 0.1:
                return 'vanishing_gradients'
            elif stat['std'] > 10.0:
                return 'exploding_gradients'

        return 'healthy'
