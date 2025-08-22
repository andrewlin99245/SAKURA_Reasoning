import os
import sys

# ---------------- Cache setup ----------------
SHARED_CACHE_DIR = os.path.expanduser("~/.cache/sakura_reasoning")
os.makedirs(SHARED_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = SHARED_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = SHARED_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = SHARED_CACHE_DIR
os.environ["HF_HUB_CACHE"] = SHARED_CACHE_DIR
os.environ["XDG_CACHE_HOME"] = os.path.expanduser("~/.cache")
os.environ["TORCH_HOME"] = SHARED_CACHE_DIR
if "PYTORCH_CACHE_HOME" in os.environ:
    del os.environ["PYTORCH_CACHE_HOME"]
print(f"Cache configured: {SHARED_CACHE_DIR}")

# ---------------- Imports ----------------
import csv
import argparse
import json
import numpy as np
from tqdm import tqdm

# (Torch / HF imports are kept to stay drop-in; not used in this script's flow)
import torch
import torch.nn.functional as F
import librosa
from datasets import load_dataset
from transformers import AutoProcessor
from scipy.stats import pearsonr, spearmanr, ttest_ind

# ---------------- Local project imports (kept for drop-in) ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, project_dir)

from utils.Qwen2Audio_patch import Qwen2AudioSLAForCausalLM  # noqa: F401
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig  # noqa: F401

try:
    from steering_vector import obtain_vsv  # noqa: F401
    from ..layers.llm_layer import add_vsv_layers, remove_vsv_layers, get_layers  # noqa: F401
    VSV_AVAILABLE = True
except Exception:
    try:
        project_dir = os.path.dirname(os.path.dirname(current_dir))
        sys.path.insert(0, project_dir)
        from src.models.steering_vector import obtain_vsv  # noqa: F401
        from src.layers.llm_layer import add_vsv_layers, remove_vsv_layers, get_layers  # noqa: F401
        VSV_AVAILABLE = True
    except Exception as e2:
        print(f"Warning: Vector steering modules not available: {e2}")
        print("Vector steering will be disabled.")
        VSV_AVAILABLE = False

# ============================================================
#                 Correlation Template Utils
# ============================================================

def _safe_corrcoef(X):
    """rowvar=False correlation with robust NaN handling."""
    if X.shape[0] < 2:
        K = X.shape[1]
        return np.zeros((K, K), dtype=np.float64)
    R = np.corrcoef(X, rowvar=False)
    R = np.where(np.isfinite(R), R, 0.0)
    np.fill_diagonal(R, 1.0)
    return R

def _fisher_z(R):
    Z = np.arctanh(np.clip(R, -0.999999, 0.999999))
    Z[np.isnan(Z)] = 0.0
    return Z

def _band_means(M):
    """Mean over symmetric diagonal bands |i-j|=d (skip d=0)."""
    K = M.shape[0]
    out = []
    for d in range(1, K):
        vals = []
        up = np.diag(M, k=d)
        lo = np.diag(M, k=-d)
        if up.size: vals.append(up)
        if lo.size: vals.append(lo)
        if vals:
            vals = np.concatenate(vals)
            out.append(float(np.mean(vals)))
        else:
            out.append(0.0)
    return out  # length K-1

def fit_correlation_template(bos_csv_path, out_path, last_k=None):
    """
    Build an inter-layer correlation template from existing BOS CSV.
    Saves JSON with: layer ids, mu/sigma, R_correct/R_incorrect, D=Z1-Z0,
    band_weights (alpha), and leading eigenvector of D.
    """
    print(f"🔧 Fitting correlation template from: {bos_csv_path}")
    with open(bos_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cos_cols = [c for c in reader.fieldnames if c.startswith("cos_L")]
    if not cos_cols:
        raise RuntimeError("No cos_L* columns found in BOS CSV.")
    cos_cols.sort(key=lambda s: int(s.split("L")[1]))
    if last_k is not None:
        cos_cols = cos_cols[-last_k:]

    C, y = [], []
    for r in rows:
        try:
            vec = [float(r[c]) for c in cos_cols]
            if any([not np.isfinite(v) for v in vec]): continue
            C.append(vec); y.append(int(r["correct"]))
        except Exception:
            continue
    C = np.asarray(C, dtype=np.float64); y = np.asarray(y, dtype=int)
    if C.shape[0] < 4:
        raise RuntimeError("Not enough clean rows to fit a template.")

    mu = C.mean(axis=0)
    sigma = C.std(axis=0); sigma = np.where(sigma < 1e-8, 1.0, sigma)
    C1 = C[y == 1]; C0 = C[y == 0]
    R1 = _safe_corrcoef(C1); R0 = _safe_corrcoef(C0)
    Z1 = _fisher_z(R1); Z0 = _fisher_z(R0)
    D = Z1 - Z0
    alpha = _band_means(D)

    w, V = np.linalg.eig((D + D.T) / 2.0)
    if w.ndim != 1: w = np.diag(w)
    idx = int(np.argmax(np.abs(w)))
    u = np.real(V[:, idx]); u = u / (np.linalg.norm(u) + 1e-12)

    payload = {
        "layers": [int(c.split("L")[1]) for c in cos_cols],
        "K": int(len(cos_cols)),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "R_correct": R1.tolist(),
        "R_incorrect": R0.tolist(),
        "D": D.tolist(),
        "band_weights": alpha,
        "eigvec": u.tolist(),
        "eigval": float(w[idx]),
        "notes": "Use with BOSLDAPredictor --corr_template to augment score."
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✅ Correlation template saved to: {out_path}")

def _outer_no_diag(z):
    M = np.outer(z, z)
    np.fill_diagonal(M, 0.0)
    return M

def _band_means_of_matrix(M):
    return _band_means(M)

# ============================================================
#             Physics Template (Resonant & Manifold)
# ============================================================

def fit_resonant_template(bos_csv_path, out_path, last_k=None, eps=1e-6):
    """
    Build a physics template from BOS CSV:
      - z-score stats (mu/sigma) over last-K layer cosines
      - resonant direction t = normalize(mean_wrong_z - mean_correct_z)
      - "correct-manifold" params: mu_corr (raw), Sigma_corr_inv (raw)
      - meta: layers used, K, mid/late split index
    """
    print(f"🔧 Fitting resonant template from: {bos_csv_path}")
    with open(bos_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cos_cols = [c for c in reader.fieldnames if c.startswith("cos_L")]
    if not cos_cols: raise RuntimeError("No cos_L* columns found.")
    cos_cols.sort(key=lambda s: int(s.split("L")[1]))
    if last_k is not None:
        cos_cols = cos_cols[-last_k:]

    C, y = [], []
    for r in rows:
        try:
            vec = [float(r[c]) for c in cos_cols]
            if any([not np.isfinite(v) for v in vec]): continue
            C.append(vec); y.append(int(r["correct"]))
        except Exception:
            continue
    C = np.asarray(C, dtype=np.float64); y = np.asarray(y, dtype=int)
    if C.shape[0] < 4:
        raise RuntimeError("Not enough rows to fit resonant template.")

    # z-score stats for matched filter
    mu = C.mean(axis=0)
    sigma = C.std(axis=0); sigma = np.where(sigma < 1e-8, 1.0, sigma)
    Z = (C - mu) / sigma

    Z1 = Z[y == 1]; Z0 = Z[y == 0]
    if Z0.shape[0] == 0 or Z1.shape[0] == 0:
        raise RuntimeError("Need both correct and incorrect samples to fit template.")

    m1 = Z1.mean(axis=0)  # correct
    m0 = Z0.mean(axis=0)  # incorrect
    t = m0 - m1
    t /= (np.linalg.norm(t) + 1e-12)

    # "Correct manifold" params in RAW space
    C1 = C[y == 1]
    mu_corr = C1.mean(axis=0)
    S_corr = np.cov(C1, rowvar=False, bias=False) if C1.shape[0] > 1 else np.eye(C.shape[1]) * eps
    S_corr += eps * np.eye(S_corr.shape[0])
    S_corr_inv = np.linalg.inv(S_corr)

    payload = {
        "layers": [int(c.split("L")[1]) for c in cos_cols],
        "K": int(len(cos_cols)),
        "z_mu": mu.tolist(),
        "z_sigma": sigma.tolist(),
        "resonant_template": t.tolist(),
        "mu_corr": mu_corr.tolist(),
        "Sigma_corr_inv": S_corr_inv.tolist(),
        "mid_late_split": int(len(cos_cols) // 2),
        "notes": "Use with BOSLDAPredictor --resonant_template for physics augmentation."
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✅ Resonant template saved to: {out_path}")

# ============================================================
#                    Per-sample feature utils
# ============================================================

def compute_correlation_features(cos_per_layer, layer_indices):
    """
    Lightweight per-sample correlation/morphology features from the last-K cosines.
    Independent of the class template; requires no retraining.
    """
    cos_values = [cos_per_layer[i] for i in layer_indices]
    n_layers = len(cos_values)

    features, names = [], []

    # 1) Consecutive deltas + trend
    if n_layers >= 2:
        diffs = np.diff(cos_values)
        features += [float(np.mean(diffs)), float(np.std(diffs))]
        names += ["consec_diff_mean", "consec_diff_std"]
        x = np.arange(n_layers, dtype=np.float64)
        slope = float(np.polyfit(x, np.asarray(cos_values, dtype=np.float64), 1)[0]) if n_layers > 1 else 0.0
        features.append(slope)
        names.append("cosine_trend_slope")

    # 2) Block coherence & cross-block corr (3 blocks if possible)
    if n_layers >= 6:
        k = n_layers // 3
        early, mid, late = cos_values[:k], cos_values[k:2*k], cos_values[2*k:]

        def cv(block):
            arr = np.asarray(block, dtype=np.float64)
            if arr.size <= 1: return 0.0
            return float(np.std(arr) / (np.mean(np.abs(arr)) + 1e-8))

        features += [cv(early), cv(mid), cv(late)]
        names += ["early_block_coherence", "middle_block_coherence", "late_block_coherence"]

        def s_corr(a, b):
            a = np.asarray(a); b = np.asarray(b)
            if a.size <= 1 or b.size <= 1: return 0.0
            c = np.corrcoef(a[:min(a.size, b.size)], b[:min(a.size, b.size)])[0, 1]
            return 0.0 if not np.isfinite(c) else float(c)

        features += [s_corr(early, mid), s_corr(mid, late), s_corr(early, late)]
        names += ["early_mid_correlation", "mid_late_correlation", "early_late_correlation"]

    # 3) Distance-diagonal correlations
    if n_layers >= 4:
        for d in (1, 2, 3):
            if d < n_layers:
                a = cos_values[:-d]; b = cos_values[d:]
                if len(a) > 1 and len(b) > 1:
                    c = np.corrcoef(a, b)[0, 1]; c = 0.0 if not np.isfinite(c) else float(c)
                else:
                    c = 0.0
                features.append(c); names.append(f"distance_{d}_correlation")

    # 4) Autocorr
    if n_layers >= 4:
        for lag in (1, 2):
            a = cos_values[:-lag]; b = cos_values[lag:]
            if len(a) > 1 and len(b) > 1:
                c = np.corrcoef(a, b)[0, 1]; c = 0.0 if not np.isfinite(c) else float(c)
            else:
                c = 0.0
            features.append(c); names.append(f"autocorr_lag_{lag}")

    # 5) Spectral roughness
    if n_layers >= 4:
        arr = np.asarray(cos_values, dtype=np.float64)
        fft_vals = np.abs(np.fft.fft(arr - arr.mean()))
        dom = float(np.max(fft_vals[1:]) / (np.sum(fft_vals[1:]) + 1e-8)) if fft_vals.size > 1 else 0.0
        mid = fft_vals.size // 2
        ratio = float(np.sum(fft_vals[1:mid]) / (np.sum(fft_vals[mid:]) + 1e-8)) if mid > 1 else 0.0
        features += [dom, ratio]; names += ["dominant_freq_ratio", "low_high_freq_ratio"]

    # 6) Stats & shape
    arr = np.asarray(cos_values, dtype=np.float64)
    features += [float(np.var(arr)), float(np.max(arr) - np.min(arr)),
                 float(np.abs(arr[-1] - arr[0])) if arr.size >= 2 else 0.0]
    names += ["cosine_variance", "cosine_range", "first_last_diff"]

    if n_layers >= 3:
        iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        peaks = sum(1 for i in range(1, arr.size - 1) if arr[i] > max(arr[i-1], arr[i+1]))
        valleys = sum(1 for i in range(1, arr.size - 1) if arr[i] < min(arr[i-1], arr[i+1]))
        features += [float(np.median(arr)), iqr, float(peaks), float(valleys)]
        names += ["cosine_median", "cosine_iqr", "num_peaks", "num_valleys"]

        grad = np.gradient(arr)
        features += [float(np.mean(np.abs(grad))), float(np.var(grad))]
        names += ["avg_gradient_mag", "gradient_variance"]

        if n_layers >= 4:
            grad2 = np.gradient(grad)
            features += [float(np.mean(np.abs(grad2)))]
            names += ["avg_curvature"]

    return {"values": features, "names": names}

def ringdown_params(c):
    """
    Quick ringdown parameters from a 1D sequence c (last-K cosines).
    Returns alpha (damping), omega (modal freq), Q, R2.
    """
    c = np.asarray(c, dtype=np.float64)
    c0 = c - c.mean()
    # frequency from FFT peak (exclude DC)
    fft = np.abs(np.fft.rfft(c0))
    if len(fft) > 2:
        k = int(np.argmax(fft[1:]) + 1)
        omega = 2*np.pi * k / max(len(c)-1, 1)
    else:
        omega = 0.0
    # envelope slope for alpha
    env = np.maximum(np.abs(c0), 1e-6)
    x = np.arange(len(env))
    try:
        slope, _ = np.polyfit(x, np.log(env), 1)
    except Exception:
        slope = 0.0
    alpha = max(-float(slope), 0.0)
    Q = (omega / (2*alpha)) if (alpha > 0 and omega > 0) else 0.0
    # crude fit R^2 via linear projection
    A = np.vstack([np.exp(-alpha*x)*np.cos(omega*x), np.exp(-alpha*x)*np.sin(omega*x)]).T
    try:
        beta, *_ = np.linalg.lstsq(A, c0, rcond=None)
        pred = A @ beta
        R2 = 1.0 - (np.var(c0 - pred) / (np.var(c0) + 1e-12))
    except Exception:
        R2 = 0.0
    return float(alpha), float(omega), float(Q), float(R2)

def matched_filter_score(c, tmpl):
    """
    Matched filter: dot of z-scored c with resonant direction t.
    tmpl contains z_mu, z_sigma, resonant_template, layers.
    """
    z_mu = np.asarray(tmpl["z_mu"], dtype=np.float64)
    z_sigma = np.asarray(tmpl["z_sigma"], dtype=np.float64)
    t = np.asarray(tmpl["resonant_template"], dtype=np.float64)
    z = (np.asarray(c, dtype=np.float64) - z_mu) / (z_sigma + 1e-8)
    return float(np.dot(z, t))

def mahalanobis_to_correct(c, mu_corr, Sigma_corr_inv):
    d = np.asarray(c, dtype=np.float64) - np.asarray(mu_corr, dtype=np.float64)
    S_inv = np.asarray(Sigma_corr_inv, dtype=np.float64)
    return float(d @ (S_inv @ d))

# ============================================================
#                    BOS LDA Predictor
# ============================================================

class BOSLDAPredictor:
    """
    Compute P(correct | BOS features) using LDA parameters
    saved in ..._BOS_stats.json (mu1, mu0, pooled Sigma, priors, feature_names).

    Optionally AUGMENT the linear score with:
      (a) inter-layer correlation template scores (frob/band/eig),
      (b) physics template scores: matched-filter, ringdown (Q, alpha),
          amplification ratio, late energy, Mahalanobis-to-correct.

    score = (w^T x + b)
            + beta1*templ_frob + beta2*templ_band + beta3*templ_eig
            + gamma_mf*mf + gamma_Q*Q + gamma_alpha*(-alpha) + gamma_amp*amp_ratio
            + gamma_energy*late_energy + gamma_maha*maha
    """
    def __init__(self, stats_json_path: str,
                 corr_template_path: str | None = None,
                 betas=(1.0, 1.0, 1.0),
                 resonant_template_path: str | None = None,
                 gammas=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # mf, Q, alpha, amp, energy, maha
                 eps: float = 1e-6):
        with open(stats_json_path, "r") as f:
            stats = json.load(f)

        all_feature_names = stats["feature_names"]
        cs = stats["class_stats"]
        mu1_all = np.asarray(cs["mu1"], dtype=np.float64)
        mu0_all = np.asarray(cs["mu0"], dtype=np.float64)
        Sigma_all = np.asarray(cs["Sigma"], dtype=np.float64)

        valid_mask = ~(np.isnan(mu1_all) | np.isnan(mu0_all))
        self.feature_names = [name for i, name in enumerate(all_feature_names) if valid_mask[i]]
        self.mu1 = mu1_all[valid_mask]
        self.mu0 = mu0_all[valid_mask]
        Sigma = Sigma_all[np.ix_(valid_mask, valid_mask)]

        self.pi1 = float(cs["priors"]["pi1"])
        self.pi0 = float(cs["priors"]["pi0"])

        d = Sigma.shape[0]
        self.Sigma = Sigma + eps * np.eye(d)
        self.Sigma_inv = np.linalg.inv(self.Sigma)

        delta = self.mu1 - self.mu0
        self.w = self.Sigma_inv @ delta
        quad1 = float(self.mu1 @ (self.Sigma_inv @ self.mu1))
        quad0 = float(self.mu0 @ (self.Sigma_inv @ self.mu0))
        self.b = -0.5 * (quad1 - quad0) + np.log(max(self.pi1, 1e-12) / max(self.pi0, 1e-12))

        # Correlation template (optional)
        self.template = None
        self.betas = tuple(float(x) for x in betas)
        if corr_template_path:
            with open(corr_template_path, "r") as f:
                T = json.load(f)
            self.template = {
                "layers": T["layers"],
                "mu": np.asarray(T["mu"], dtype=np.float64),
                "sigma": np.asarray(T["sigma"], dtype=np.float64),
                "D": np.asarray(T["D"], dtype=np.float64),
                "band_weights": np.asarray(T["band_weights"], dtype=np.float64),
                "eigvec": np.asarray(T["eigvec"], dtype=np.float64),
            }
            self.template["sigma"] = np.where(self.template["sigma"] < 1e-8, 1.0, self.template["sigma"])
            print(f"📐 Loaded correlation template with K={len(self.template['layers'])}")

        # Resonant/physics template (optional)
        self.resonant = None
        self.gammas = tuple(float(x) for x in gammas)  # mf, Q, alpha, amp, energy, maha
        if resonant_template_path:
            with open(resonant_template_path, "r") as f:
                R = json.load(f)
            self.resonant = {
                "layers": R["layers"],
                "z_mu": np.asarray(R["z_mu"], dtype=np.float64),
                "z_sigma": np.asarray(R["z_sigma"], dtype=np.float64),
                "resonant_template": np.asarray(R["resonant_template"], dtype=np.float64),
                "mu_corr": np.asarray(R["mu_corr"], dtype=np.float64),
                "Sigma_corr_inv": np.asarray(R["Sigma_corr_inv"], dtype=np.float64),
                "split": int(R.get("mid_late_split", max(1, R["K"]//2))),
            }
            self.resonant["z_sigma"] = np.where(self.resonant["z_sigma"] < 1e-8, 1.0, self.resonant["z_sigma"])
            print(f"🧪 Loaded resonant template with K={len(self.resonant['layers'])}")

    # ---- helpers to build x in the correct order ----
    def vector_from_csv_row(self, row: dict) -> np.ndarray:
        """
        Build feature vector x in the same order as self.feature_names.
        Correlation morphology features are computed on the fly from cos_L* in the row.
        """
        scalars = {
            "bos_mean_late": row.get("bos_mean_late", None),
            "bos_std_late": row.get("bos_std_late", None),
            "entropy_bos": row.get("entropy_bos", None),
            "margin_bos": row.get("margin_bos", None),
            "gap_EL": row.get("gap_EL", None),
        }

        cos_layer_vals = {}
        layers_in_row = []
        for k, v in row.items():
            if k.startswith("cos_L"):
                try:
                    lid = int(k.split("L")[1])
                    cos_layer_vals[lid] = float(v)
                    layers_in_row.append(lid)
                except Exception:
                    continue
        layers_in_row = sorted(set(layers_in_row))

        if len(layers_in_row) >= 2:
            K = min(8, len(layers_in_row))
            last_ids = layers_in_row[-K:]
            cos_seq = {lid: cos_layer_vals[lid] for lid in last_ids}
            corr_feats = compute_correlation_features(cos_seq, last_ids)
        else:
            corr_feats = {"values": [], "names": []}

        all_vals = {}
        for n, v in scalars.items():
            if v is not None: all_vals[n] = float(v)
        for lid, val in cos_layer_vals.items():
            all_vals[f"cos_L{lid}"] = float(val)
        for n, v in zip(corr_feats["names"], corr_feats["values"]):
            all_vals[n] = float(v)

        x = []
        for name in self.feature_names:
            if name not in all_vals:
                raise KeyError(f"Missing feature '{name}' in row-derived features.")
            x.append(all_vals[name])
        return np.asarray(x, dtype=np.float64)

    # ---- correlation template scoring from a CSV row ----
    def template_scores_from_row(self, row: dict):
        if self.template is None:
            return 0.0, 0.0, 0.0
        LIDS = self.template["layers"]
        c = []
        for lid in LIDS:
            key = f"cos_L{lid}"
            if key not in row: return 0.0, 0.0, 0.0
            val = float(row[key])
            if not np.isfinite(val): return 0.0, 0.0, 0.0
            c.append(val)
        c = np.asarray(c, dtype=np.float64)
        mu = self.template["mu"]; sigma = self.template["sigma"]
        z = (c - mu) / sigma
        M = _outer_no_diag(z)
        D = self.template["D"]
        frob = float(np.sum(M * D) / (M.shape[0] * M.shape[1]))
        bands = _band_means_of_matrix(M)
        alpha = self.template["band_weights"]
        band_score = float(np.dot(alpha, bands))
        eig = float(np.dot(self.template["eigvec"], z))
        return frob, band_score, eig

    # ---- physics template scoring from a CSV row ----
    def physics_scores_from_row(self, row: dict):
        if self.resonant is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}

        LIDS = self.resonant["layers"]
        c = []
        for lid in LIDS:
            key = f"cos_L{lid}"
            if key not in row: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}
            val = float(row[key])
            if not np.isfinite(val): return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}
            c.append(val)
        c = np.asarray(c, dtype=np.float64)

        # Matched filter
        mf = matched_filter_score(c, self.resonant)

        # Ringdown params
        alpha, omega, Q, R2 = ringdown_params(c)

        # Amplification & energy (mid vs late according to split)
        s = self.resonant["split"]
        mid = c[:max(1, s)]
        late = c[s:] if len(c) > s else c[-max(1, len(c)//2):]
        amp = float(np.mean(np.abs(late)) / (np.mean(np.abs(mid)) + 1e-8))
        energy = float(np.mean(late**2))  # normalized mean-square

        # Mahalanobis to correct manifold (raw)
        maha = mahalanobis_to_correct(c, self.resonant["mu_corr"], self.resonant["Sigma_corr_inv"])

        extras = {"alpha": alpha, "omega": omega, "Q": Q, "R2": R2, "amp_ratio": amp, "late_energy": energy}
        return mf, Q, alpha, amp, energy, maha, extras

    # ---- probability API ----
    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            ez = np.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = np.exp(z)
            return ez / (1.0 + ez)

    def base_score(self, x: np.ndarray) -> float:
        return float(self.w @ x + self.b)

    def augmented_score_from_row(self, row: dict, x: np.ndarray) -> tuple[float, dict]:
        s0 = self.base_score(x)
        frob, band, eig = self.template_scores_from_row(row)
        mf, Q, alpha, amp, energy, maha, phys_extras = self.physics_scores_from_row(row)
        s_aug = (s0
                 + self.betas[0]*frob + self.betas[1]*band + self.betas[2]*eig
                 + self.gammas[0]*mf + self.gammas[1]*Q + self.gammas[2]*(-alpha)
                 + self.gammas[3]*amp + self.gammas[4]*energy + self.gammas[5]*maha)
        extras = {
            "s_base": s0,
            "templ_frob": frob, "templ_band": band, "templ_eig": eig,
            "mf": mf, "Q": Q, "alpha": alpha, "amp_ratio": amp, "late_energy": energy, "maha": maha,
            "beta1": self.betas[0], "beta2": self.betas[1], "beta3": self.betas[2],
            "gamma_mf": self.gammas[0], "gamma_Q": self.gammas[1], "gamma_alpha": self.gammas[2],
            "gamma_amp": self.gammas[3], "gamma_energy": self.gammas[4], "gamma_maha": self.gammas[5],
            **phys_extras
        }
        return s_aug, extras

    def p_correct_from_row(self, row: dict, use_augmented: bool = True) -> tuple[float, dict]:
        x = self.vector_from_csv_row(row)
        if use_augmented:
            s, extras = self.augmented_score_from_row(row, x)
            return self._sigmoid(s), extras
        else:
            s = self.base_score(x)
            return self._sigmoid(s), {"s_base": s}

# ============================================================
#                 Evaluation / CLI plumbing
# ============================================================

def evaluate_with_lda_correction(bos_csv_path: str,
                                 lda_stats_path: str,
                                 corr_template_path: str | None = None,
                                 betas=(1.0, 1.0, 1.0),
                                 resonant_template_path: str | None = None,
                                 gammas=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                 flip_threshold: float = 0.5,
                                 use_augmented: bool = True,
                                 output_dir: str = "./"):
    """
    1) Loads BOS CSV
    2) Computes P(correct) per row from LDA (optionally augmented by correlation & physics templates)
    3) Flips if P(incorrect) > threshold
    4) Saves JSON/CSV with details
    """
    print(f"📊 Loading BOS features from: {bos_csv_path}")
    print(f"📊 Loading LDA stats from: {lda_stats_path}")
    if corr_template_path:
        print(f"📐 Using correlation template: {corr_template_path} (betas={betas})")
    if resonant_template_path:
        print(f"🧪 Using resonant template: {resonant_template_path} (gammas={gammas})")

    predictor = BOSLDAPredictor(lda_stats_path, corr_template_path, betas,
                                resonant_template_path, gammas)

    with open(bos_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"📈 Loaded {len(rows)} samples")

    orig_correct = 0
    aug_correct = 0
    preds = []

    for row in tqdm(rows, desc="Scoring samples"):
        oc = bool(int(row["correct"]))
        if oc: orig_correct += 1

        true_label = row["label"]
        model_resp = row["response"]

        p_c, extras = predictor.p_correct_from_row(row, use_augmented=use_augmented)
        p_i = 1.0 - p_c

        if p_i > flip_threshold:
            corrected = "No" if model_resp == "Yes" else "Yes"
            flipped = True
        else:
            corrected = model_resp
            flipped = False

        cc = (corrected == true_label)
        if cc: aug_correct += 1

        preds.append({
            "entry_id": row["entry_id"],
            "true_label": true_label,
            "original_response": model_resp,
            "original_correct": oc,
            "corrected_response": corrected,
            "corrected_correct": cc,
            "p_correct": p_c,
            "p_incorrect": p_i,
            "s_base": extras.get("s_base"),
            "templ_frob": extras.get("templ_frob", 0.0),
            "templ_band": extras.get("templ_band", 0.0),
            "templ_eig": extras.get("templ_eig", 0.0),
            "mf": extras.get("mf", 0.0),
            "Q": extras.get("Q", 0.0),
            "alpha": extras.get("alpha", 0.0),
            "amp_ratio": extras.get("amp_ratio", 0.0),
            "late_energy": extras.get("late_energy", 0.0),
            "maha": extras.get("maha", 0.0),
            "beta1": extras.get("beta1", 0.0),
            "beta2": extras.get("beta2", 0.0),
            "beta3": extras.get("beta3", 0.0),
            "gamma_mf": extras.get("gamma_mf", 0.0),
            "gamma_Q": extras.get("gamma_Q", 0.0),
            "gamma_alpha": extras.get("gamma_alpha", 0.0),
            "gamma_amp": extras.get("gamma_amp", 0.0),
            "gamma_energy": extras.get("gamma_energy", 0.0),
            "gamma_maha": extras.get("gamma_maha", 0.0),
            "flipped": flipped
        })

    N = len(rows)
    orig_acc = 100.0 * orig_correct / max(N, 1)
    aug_acc = 100.0 * aug_correct / max(N, 1)

    print("\n🏁 Evaluation Results")
    print(f"  Total samples: {N}")
    print(f"  🎯 Original model accuracy:  {orig_acc:.2f}% ({orig_correct}/{N})")
    print(f"  🎯 Augmented accuracy:       {aug_acc:.2f}% ({aug_correct}/{N})"
          f"{'  (corr+phys)' if (corr_template_path or resonant_template_path) and use_augmented else ''}")
    print(f"  📈 Improvement: {aug_acc - orig_acc:.2f} pts")
    flipped_count = sum(int(p["flipped"]) for p in preds)
    print(f"  🔄 Predictions flipped: {flipped_count}/{N} ({(100.0*flipped_count/max(N,1)):.1f}%)")

    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "lda_correction_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "original_accuracy": orig_acc,
            "augmented_accuracy": aug_acc,
            "improvement": aug_acc - orig_acc,
            "total_samples": N,
            "flipped_count": flipped_count,
            "use_augmented": use_augmented,
            "flip_threshold": flip_threshold,
            "betas": list(betas),
            "gammas": list(gammas),
            "corr_template": corr_template_path,
            "resonant_template": resonant_template_path
        }, f, indent=2)
    print(f"💾 Summary saved to: {out_json}")

    out_csv = os.path.join(output_dir, "lda_predictions.csv")
    with open(out_csv, "w", newline="") as f:
        fieldnames = ["entry_id", "true_label", "original_response", "original_correct",
                      "corrected_response", "corrected_correct", "p_correct", "p_incorrect",
                      "s_base", "templ_frob", "templ_band", "templ_eig",
                      "mf", "Q", "alpha", "amp_ratio", "late_energy", "maha",
                      "beta1", "beta2", "beta3",
                      "gamma_mf", "gamma_Q", "gamma_alpha", "gamma_amp", "gamma_energy", "gamma_maha",
                      "flipped"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for p in preds:
            w.writerow(p)
    print(f"💾 Predictions saved to: {out_csv}")

    return {
        "original_accuracy": orig_acc,
        "augmented_accuracy": aug_acc,
        "improvement": aug_acc - orig_acc,
        "total_samples": N,
        "flipped_count": flipped_count,
        "predictions": preds
    }

# ============================================================
#                            CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LDA-based correction of audio hallucination predictions "
                    "(+ optional correlation & physics template augmentation)"
    )

    # Normal evaluation args
    parser.add_argument("--bos_csv", type=str, help="Path to *_BOS_features.csv")
    parser.add_argument("--lda_stats", type=str, help="Path to *_BOS_stats.json produced earlier")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save results")

    # Flip threshold
    parser.add_argument("--flip_threshold", type=float, default=0.5, help="Flip if P(incorrect) > threshold")

    # Augmentation with correlation template
    parser.add_argument("--corr_template", type=str, default=None, help="Path to correlation template JSON")
    parser.add_argument("--use_augmented", action="store_true",
                        help="Use augmented score if template(s) provided")

    parser.add_argument("--beta1", type=float, default=1.0, help="Weight for templ_frob")
    parser.add_argument("--beta2", type=float, default=1.0, help="Weight for templ_band")
    parser.add_argument("--beta3", type=float, default=1.0, help="Weight for templ_eig")

    # Physics (resonant) template
    parser.add_argument("--resonant_template", type=str, default=None,
                        help="Path to resonant/physics template JSON")
    parser.add_argument("--gamma_mf", type=float, default=0.0, help="Weight for matched-filter score")
    parser.add_argument("--gamma_Q", type=float, default=0.0, help="Weight for Q (quality factor)")
    parser.add_argument("--gamma_alpha", type=float, default=0.0,
                        help="Weight for (-alpha) so positive weight rewards lower damping")
    parser.add_argument("--gamma_amp", type=float, default=0.0, help="Weight for amplification ratio (late/mid)")
    parser.add_argument("--gamma_energy", type=float, default=0.0, help="Weight for late-layer energy")
    parser.add_argument("--gamma_maha", type=float, default=0.0, help="Weight for Mahalanobis-to-correct")

    # Fitting templates (one-off)
    parser.add_argument("--fit_corr_template_from", type=str, default=None,
                        help="Fit correlation template from this BOS CSV and exit")
    parser.add_argument("--corr_template_out", type=str, default=None,
                        help="Where to save fitted correlation template JSON")
    parser.add_argument("--template_last_k", type=int, default=None,
                        help="Use only last-K cos_L* columns when fitting corr template")

    parser.add_argument("--fit_resonant_from", type=str, default=None,
                        help="Fit resonant (physics) template from this BOS CSV and exit")
    parser.add_argument("--resonant_out", type=str, default=None,
                        help="Where to save fitted resonant template JSON")
    parser.add_argument("--resonant_last_k", type=int, default=None,
                        help="Use only last-K cos_L* when fitting resonant template")

    args = parser.parse_args()

    # If asked to fit a template, do that and exit
    if args.fit_corr_template_from:
        if not args.corr_template_out:
            raise SystemExit("--corr_template_out is required when using --fit_corr_template_from")
        fit_correlation_template(args.fit_corr_template_from, args.corr_template_out,
                                 last_k=args.template_last_k)
        return

    if args.fit_resonant_from:
        if not args.resonant_out:
            raise SystemExit("--resonant_out is required when using --fit_resonant_from")
        fit_resonant_template(args.fit_resonant_from, args.resonant_out,
                              last_k=args.resonant_last_k)
        return

    # Otherwise, run evaluation
    if not args.bos_csv or not args.lda_stats:
        raise SystemExit("--bos_csv and --lda_stats are required for evaluation")

    betas = (args.beta1, args.beta2, args.beta3)
    gammas = (args.gamma_mf, args.gamma_Q, args.gamma_alpha,
              args.gamma_amp, args.gamma_energy, args.gamma_maha)

    evaluate_with_lda_correction(
        bos_csv_path=args.bos_csv,
        lda_stats_path=args.lda_stats,
        corr_template_path=args.corr_template,
        betas=betas,
        resonant_template_path=args.resonant_template,
        gammas=gammas,
        flip_threshold=args.flip_threshold,
        use_augmented=bool((args.corr_template or args.resonant_template) and args.use_augmented),
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
