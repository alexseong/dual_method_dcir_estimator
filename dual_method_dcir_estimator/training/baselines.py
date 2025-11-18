import numpy as np

def detect_steps(I, deltaI=5.0, rest_eps=0.2, pre=5):
    idx = []
    for k in range(1, len(I)):
        if abs(I[k] - I[k-1]) >= deltaI:
            if np.mean(np.abs(I[max(0, k-pre):k])) <= rest_eps:
                idx.append(k)
    return idx

def vdrop_dcirs(V, I, k_list, tau=1, W=3):
    out = []
    for k in k_list:
        pre = V[max(0, k-W):k].mean()
        post = V[min(len(V), k+tau):min(len(V), k+tau+W)].mean()
        preI = I[max(0, k-W):k].mean()
        postI = I[min(len(I), k+tau):min(len(I), k+tau+W)].mean()
        dI = postI - preI
        if abs(dI) < 1e-6: continue
        R = (post - pre)/dI
        out.append((k, R))
    return out