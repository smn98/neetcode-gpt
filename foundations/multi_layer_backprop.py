import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        x = np.array(x, dtype=float).reshape(1, -1)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float).reshape(1, -1)
        
        z1 = x @ W1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2
        # Loss: MSE = mean((predictions - y_true)^2)
        loss = np.mean((z2 - y_true) ** 2)
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        n = z2.size
        dz2 = (2 / n) * (z2 - y_true)
        dW2 = dz2.T @ a1
        db2 = dz2.squeeze()
        da1 = dz2 @ W2
        dz1 = da1 * (z1 > 0)
        dW1 = dz1.T @ x
        db1 = dz1.squeeze()
        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1,4).tolist(),
            'db1': np.round(db1.reshape(-1),4).tolist(),
            'dW2': np.round(dW2,4).tolist(),
            'db2': np.round(db2.reshape(-1),4).tolist(),
        }
        