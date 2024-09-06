import numpy as np
from sklearn.decomposition import NMF

def nmf_approx_two_sources(S_mix_gt, beta_loss='frobenius', solver='cd'):
    if len(S_mix_gt.shape) == 3:
        S_mix_gt = np.mean(S_mix_gt, axis=2)

    nmf = NMF(n_components=2, random_state=0, beta_loss=beta_loss, solver=solver)
    W = nmf.fit_transform(S_mix_gt.cpu())
    H = nmf.components_

    # Reconstruct sources
    S1_approx = np.dot(W[:, 0:1], H[0:1, :])
    S2_approx = np.dot(W[:, 1:2], H[1:2, :])

    return S1_approx, S2_approx
