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

def enforce_sign_convention(df: pd.DataFrame, current_sign: str) -> pd.DataFrame:
    # project rule: discharge-positive, charge-negative
    # If your data is charge-positive, flip sign
    # In your earlier note: PEC_Measured_Current + = charge; - = discharge → we flip
    if current_sign == "discharge_positive":
        # If we detect majority of positive I co-occurs with V rising (charge), flip
        # Heuristic: if mean(I) > 0.0 on rest? We'll just provide a toggle here:
        # The user can set it in config; here we assume AVL file is charge-positive → flip
        df = df.copy()
        df["I"] = -df["I"]
    return df

def preprocess(df: pd.DataFrame, cfg: PreprocConfig, temp_std: Standardizer | None = None):
    out = df.copy()
    out = enforce_sign_convention(out, cfg.current_sign)
    out = out.sort_values("t").reset_index(drop=True)
    
    # Ensure equidistant t if needed (optional)
    if cfg.temp_standardize:
        if temp_std is None:
            temp_std = Standardizer()
            temp_std.fit(out["T"].values)
        out["Tz"] = temp_std.transform(out["T"].values)
    else:
        out["Tz"] = out["T"].values
    return out, temp_std