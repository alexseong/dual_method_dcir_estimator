import numpy as np
import pandas as pd

def ocv_analytic(soc):
    # Simple smooth surrogate; replace with lookup if needed
    # concave-ish shape typical for automotive cells
    return 3.0 + 1.2*soc - 0.1*np.sin(8*np.pi*soc)

def simulate_2rc_sequence( T=4000, dt=1.0, Imax=25.0, seed=17, 
                           R0=3.0e-3, R1=2.0e-3, C1=1200.0, R2=3.0e-3, C2=2000.0, 
                           T_C=25.0, soc0=0.8):
    rng = np.random.default_rng(seed)
    t = np.arange(T)*dt

    # PRBS + pulses
    I = np.zeros(T)
    for k in range(200, T, 400):
        amp = rng.uniform(0.2, 1.0)*Imax
        sign = rng.choice([-1,1])
        width = rng.integers(40, 120)
        I[k:k+width] = sign*amp
    I += rng.normal(0, 0.15, size=T)

    v1 = 0.0
    v2 = 0.0
    soc = soc0
    V = []
    SOC_list = []
    T_list = []
    Q = 3600.0
    eta = 1.0

    for k in range(T):
        # ECM forward Euler (simulation only)
        dv1 = (-v1/(R1*C1) + I[k]/C1 ) * dt
        dv2 = (-v2/(R2*C2) + I[k]/C2 ) * dt
        v1 += dv1
        v2 += dv2
        soc += (-eta/Q) * I[k] * dt
        soc = np.clip(soc, 0.0, 1.0)
        V.append(ocv_analytic(soc) - R0*I[k] - v1 - v2)
        SOC_list.append(soc)
        T_list.append(T_C)
    
    df = pd.DataFrame({
        "t": t,
        "V": np.array(V) + rng.normal(0, 0.002, size=T), # measurement noise
        "I": I,
        "T": np.array(T_list),
        "SOC_sim": np.array(SOC_list)   # for reference (not used in training)
    })

    return df

def save_parquet(df: pd.DataFrame, path: str):
    df.to_parquet(path, index=False)