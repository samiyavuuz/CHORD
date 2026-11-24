"""The ten post-hoc OOD detectors of the paper, implemented in numpy following the original papers.

Every detector works on feature-space vectors u (any correction) and the frozen source head z = W u + b
(the logistic regression). Statistics are fitted once on the CPTAC-train features (model.bank). Hyperparameters are
the original papers' defaults. Output: OOD score (higher = more OOD).

- Output-based: MSP (Hendrycks & Gimpel, 2017), Energy (Liu et al., 2020).
- Feature-based: kNN (Sun et al., 2022: Euclidean distance of the L2-normalised feature to its k-th nearest source-train
  feature, k = 10), KernelPCA (Fang et al., 2024; cosine kernel, >= 85% of the spectrum), CARef (Ling et al., 2025).
- Feature + head: ViM (Wang et al., 2022), fDBD (Liu & Qin, 2024), NNGuide (Park et al., 2023; original formulation:
  confidence x mean top-k cosine to the confidence-scaled bank), NCI (Liu et al., 2025), CADRef (Ling et al., 2025).
"""

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.covariance import EmpiricalCovariance

from chord.source import l2

DETECTORS = {  # name -> family (for tables)
    "MSP": "output", "Energy": "output",
    "kNN": "feature", "KernelPCA": "feature", "CARef": "feature",
    "ViM": "feature+head", "fDBD": "feature+head", "NNGuide": "feature+head", "NCI": "feature+head",
    "CADRef": "feature+head",
}
KNN_K = 10
VIM_DIM, KPCA_VAR = 512, 0.85
NCI_ALPHA, NNGUIDE_K = 1e-4, 10


class Detectors:
    def __init__(self, model):
        """model: fitted SourceModel."""
        self.W, self.b = model.clf.coef_, model.clf.intercept_
        F = model.bank
        K = len(model.classes)
        self.K, self.bank = K, F
        z_tr = self.logits(F)
        self.mu_tr = F.mean(0)
        # KernelPCA (cosine kernel = linear kernel on unit features): centred unit features, keep >= 85% of the spectrum
        Fc = l2(F) - l2(F).mean(0)
        self.kpca_mu = l2(F).mean(0)
        uu, s, _ = np.linalg.svd(Fc.T @ Fc)
        q = int(np.searchsorted(np.cumsum(s) / s.sum(), KPCA_VAR) + 1)
        self.kpca_U = uu[:, :q]
        # ViM: principal space of train features around the origin u0 of the head
        self.u0 = -np.linalg.pinv(self.W) @ self.b
        cov = EmpiricalCovariance(assume_centered=True).fit(F - self.u0).covariance_
        w, v = np.linalg.eigh(cov)
        self.NS = np.ascontiguousarray(v[:, np.argsort(-w)[VIM_DIM:]])
        self.vim_alpha = z_tr.max(1).mean() / np.linalg.norm((F - self.u0) @ self.NS, axis=1).mean()
        # CARef / CADRef: train feature means by predicted class
        pred = z_tr.argmax(1)
        self.pred_means = np.stack([F[pred == c].mean(0) for c in range(K)])
        self.cad_global_msp = softmax(z_tr, 1).max(1).mean()
        # fDBD denominators ||w_c - w_p||
        self.fdbd_den = np.linalg.norm(self.W[:, None] - self.W[None], axis=2) + np.eye(K)
        # NNGuide: unit train features scaled by their energy confidence
        self.nng_bank = l2(F) * logsumexp(z_tr, 1)[:, None]

    def logits(self, u: np.ndarray) -> np.ndarray:
        return u @ self.W.T + self.b

    def id_scores(self, u: np.ndarray) -> dict:
        """Higher = more in-distribution (detector convention)."""
        z = self.logits(u)
        p = softmax(z, 1)
        pred = z.argmax(1)
        s = {"MSP": p.max(1), "Energy": logsumexp(z, 1)}
        un = l2(u)
        sim = u @ self.bank.T
        s["kNN"] = -np.sqrt(np.maximum(2 - 2 * np.partition(sim, -KNN_K, 1)[:, -KNN_K], 0))
        xc = un - self.kpca_mu
        s["KernelPCA"] = -np.linalg.norm(xc - xc @ self.kpca_U @ self.kpca_U.T, axis=1)
        tm = self.pred_means[pred]
        s["CARef"] = -np.abs(tm - u).sum(1) / np.abs(u).sum(1)
        res = np.linalg.norm((u - self.u0) @ self.NS, axis=1)
        s["ViM"] = logsumexp(z, 1) - self.vim_alpha * res
        zmax = z.max(1, keepdims=True)
        s["fDBD"] = (np.abs(z - zmax) / self.fdbd_den[pred]).sum(1) / np.linalg.norm(u - self.mu_tr, axis=1)
        conf = logsumexp(z, 1)
        gsim = un @ self.nng_bank.T
        s["NNGuide"] = conf * np.partition(gsim, -NNGUIDE_K, 1)[:, -NNGUIDE_K:].mean(1)
        dcen = u - self.mu_tr
        s["NCI"] = (self.W[pred] * dcen).sum(1) / np.linalg.norm(dcen, axis=1) + NCI_ALPHA * np.abs(u).sum(1)
        sg = np.sign(self.W[pred])
        dist = u - tm
        l1 = np.abs(u).sum(1)
        ep, en = np.clip(dist * sg, 0, None).sum(1) / l1, np.clip(-dist * sg, 0, None).sum(1) / l1
        s["CADRef"] = -(ep / p.max(1) + en / self.cad_global_msp)
        assert set(s) == set(DETECTORS)
        return s

    def scores(self, u: np.ndarray) -> dict:
        """OOD scores (higher = more OOD), in DETECTORS order."""
        s = self.id_scores(u)
        return {k: -s[k] for k in DETECTORS}
