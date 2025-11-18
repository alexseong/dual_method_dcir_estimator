import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def overlay_voltage(out_dir, t, V_true, V_pred, title="Voltage Overlay"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(t, V_true.cpu().numpy(), label="meas")
    plt.plot(t, V_pred.cpu().numpy(), label="pred", alpha=0.8)
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
    plt.plot(t, soc.cpu().numpy())
    plt.xlabel("Time (s)")
    plt.ylabel("SOC")
    plt.title("SOC trajectory")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "soc.png"), dpi=180)
    plt.close()

def r0_scatter(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()


