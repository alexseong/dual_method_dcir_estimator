import numpy as np
import pandas as pd

def sliding_windows(df: pd.DataFrame, H: int, stride: int, min_dyn_frac: float=0.2, I_eps=0.2):
    """
    Yield windows (start_idx, end_idx) where fraction of |I|>I_eps >= min_dyn_frac.
    """
    N = len(df)
    out = []

    for s in range(0, max(0, N - H + 1), stride):
        e = s + H
        Iwin = df["I"].values[s:e]
        if len(Iwin) < H: break
        dyn = np.mean(np.abs(Iwin) > I_eps)
        if dyn >= min_dyn_frac:
            out.append((s, e))

    return out
