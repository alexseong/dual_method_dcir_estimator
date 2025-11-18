import numpy as np

def detect_steps(I, deltaI=5.0, rest_eps=0.2, pre=5):
    idx = []
    for k in range(1, len(I)):
        if abs(I[k] - I[k-1]) >= deltaI:
            if np.mean  I[max(0, k-pre)]