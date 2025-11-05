import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class PreprocConfig:
    dt: float = 1.0
    current_sign: str = "discharge_positive"  # enforce discharge-positive rule
    temp_standardize: bool = True

class Standardizer:
    def __init__(self) -> None:
        self.mu = None
        self.sigma = None
    def fit(self, x: np.ndarray):
        self.mu = float(np.mean(x))
        self.sigma = float(np.std(x) + 1e-12)
    def transform(self, x:np.ndarray):
        return (x - self.mu) / self.sigma
    def inverse(self, z:np.ndarray):
        return z*self.sigma + self.mu

def enforce_sign_convention():
    pass

def preprocess():
    pass