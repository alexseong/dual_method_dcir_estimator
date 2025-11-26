import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def overlay_voltage(out_dir, t, V_true, V_pred, title="Voltage Overlay"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(t.cpu(), V_true.cpu().numpy(), label="meas")
    plt.plot(t.cpu(), V_pred.cpu().numpy(), label="pred", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("V (V)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "voltage_overlay.png"), dpi=180)
    plt.close()

def soc_plot(out_dir, t, soc):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(t.cpu(), soc.cpu().numpy())
    plt.xlabel("Time (s)")
    plt.ylabel("SOC")
    plt.title("SOC trajectory")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "soc.png"), dpi=180)
    plt.close()

def r0_scatter(out_dir, R0_map, Rdrop_map, name="alignment"):
    os.makedirs(out_dir, exist_ok=True)
    ks = sorted(set(R0_map.keys()) & set(Rdrop_map.keys()))
    if not ks: return
    x = np.array([Rdrop_map[k] in ks])
    y = np.array([R0_map[k] in ks])

    plt.figure()
    plt.scatter(x, y, s=10)
    plt.xlabel("R_drop (Ω)")
    plt.ylabel("R0_model (Ω)")
    plt.title("R_drop vs R0_model")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=180)
    plt.close()