# Convention record (consistent with config.py / gram_scatter.py):
#   DTD: cosine d_i·d_j / (‖d_i‖‖d_j‖)   ZTZ: raw Z.T @ Z

import json
import subprocess
from pathlib import Path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent,
        ).decode().strip()
    except Exception:
        return "unknown"


def save_membership(
    path: Path | str,
    *,
    arch: str,
    layer: int,
    concept_id: str,
    mode: str,           # "atom" | "probe"
    atom_k: int,
    s_main: list[int],
    s_abs: list[int],
    support_counts: dict[int, int],
    theta_fire: float,
    tau: float,
    share_threshold: float,
    min_support: int,
    n_pos_tokens: int,
    dtd_convention: str,
    ztz_convention: str,
    probe_cos: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "arch": arch,
        "layer": layer,
        "concept_id": concept_id,
        "mode": mode,
        "atom_k": atom_k,
        "S_main": s_main,
        "S_abs": s_abs,
        "support_counts": {str(k): v for k, v in support_counts.items()},
        "thresholds": {
            "theta_fire": theta_fire,
            "tau": tau,
            "share_threshold": share_threshold,
            "min_support": min_support,
        },
        "probe_cos": probe_cos,
        "n_pos_tokens": n_pos_tokens,
        "dtd_convention": dtd_convention,
        "ztz_convention": ztz_convention,
        "git_commit": _git_commit(),
    }
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[io] wrote {path}  |S_main|={len(s_main)} |S_abs|={len(s_abs)}")


def load_membership(path: Path | str) -> dict:
    with open(path) as f:
        return json.load(f)


def membership_path(results_dir: Path | str, arch: str, layer: int, concept_id: str, mode: str) -> Path:
    return Path(results_dir) / f"{arch}_L{layer}_{concept_id}_{mode}.json"
