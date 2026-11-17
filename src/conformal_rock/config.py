"""Settings of the released study (profile "paper").

These values are copied from ``results/logs/config.json`` of the v2.0.0 release. Changing them changes the
experiment; the paper's numbers correspond exactly to the defaults below.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# (name, column in DeePore's Y array, transform, axis label)
PROPS = [
    ("log_k", 0, "log10", "log10 permeability [px^2]"),
    ("log_F", 1, "log10", "log10 formation factor"),
    ("tau", 4, "id", "tortuosity"),
    ("log_ssa", 10, "log10", "log10 specific surface [px^-1]"),
    ("log_rp", 7, "log10", "log10 mean pore radius [px]"),
    ("E_rel", 14, "id", "relative Young's modulus"),
]
PROP_NAMES = [p[0] for p in PROPS]
J_K = PROP_NAMES.index("log_k")

# DeePore's own outlier rules (columns of the first 15 single-value properties)
EXCLUSION_RULES = [
    ("formation factor > 120", 1, ">", 120.0),
    ("tortuosity > 1.8", 4, ">", 1.8),
    ("permeability < 1e-4 px^2", 0, "<", 1e-4),
    ("cementation factor < 1e-5", 2, "<", 1e-5),
    ("relative Young's modulus > 0.7", 14, ">", 0.7),
]

ZENODO_URL = "https://zenodo.org/records/4297035/files/DeePore_Dataset.h5?download=1"
ZENODO_MD5 = "d56fbd79f451353af131402c42b179d7"


@dataclass
class Config:
    profile: str = "paper"
    n_max: int = 0                      # 0 = use every valid volume
    work_res: int = 64
    hi_res: int = 128
    poremap_cap: int = 24
    res_subset_n: int = 2000
    conn_subset_n: int = 300
    data_seed: int = 12345
    frac_train: float = 0.55
    frac_val: float = 0.05
    frac_cal: float = 0.20
    frac_test: float = 0.20
    base: int = 16
    p_drop: float = 0.1
    epochs: int = 24
    batch: int = 48
    lr: float = 2e-3
    wd: float = 1e-4
    w_nll: float = 0.5
    w_q: float = 1.0
    n_seeds: int = 10
    n_abl: int = 3
    ens_M: int = 5
    mc_T: int = 30
    mc_T_grid: list = field(default_factory=lambda: [5, 10, 20, 30, 50, 100])
    alpha: float = 0.1
    q_lo: list = field(default_factory=lambda: [0.01, 0.025, 0.05, 0.10, 0.15])
    sweep_targets: list = field(default_factory=lambda: [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98])
    eps_rel: float = 1e-3
    n_groups: int = 5
    ood_top_frac: float = 0.25
    n_families: int = 10
    n_family_folds: int = 3
    n_repart: int = 300
    n_tilt_rep: int = 20
    tol_factors: list = field(default_factory=lambda: [1.25, 1.5, 2.0, 3.0])
    risk_levels: list = field(default_factory=lambda: [0.01, 0.05, 0.10])
    ltt_delta: float = 0.1
    guard_delta: float = 0.05

    @property
    def q_levels(self) -> list:
        """All quantile levels of the network's quantile head: q_lo followed by their complements."""
        return list(self.q_lo) + [1 - t for t in self.q_lo]

    def to_dict(self) -> dict:
        return asdict(self)


CFG = Config()
