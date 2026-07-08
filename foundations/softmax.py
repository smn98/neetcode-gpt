import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        res = z - np.max(z)
        res = np.exp(res)
        return np.round(res / np.sum(res), 4)
