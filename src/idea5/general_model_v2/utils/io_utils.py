from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


THIS_FILE = Path(__file__).resolve()
GENERAL_MODEL_V2_ROOT = THIS_FILE.parents[1]
DEFAULT_RESULT_ROOT = GENERAL_MODEL_V2_ROOT / "results"
DEFAULT_LOG_ROOT = GENERAL_MODEL_V2_ROOT / "logs"


def build_run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def format_number_for_name(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def format_imitation_reg_for_name(value: float) -> str:
    scaled = float(value) * 100.0
    rounded = round(scaled)
    if abs(scaled - rounded) < 1e-9 and 0 <= rounded <= 999:
        return f"{int(rounded):03d}"
    return format_number_for_name(value)


def build_scenario_name(config: Dict[str, Any]) -> str:
    env = config["env"]
    c_value = int(env["C"]) if float(env["C"]).is_integer() else env["C"]
    parts = [
        f"two_pool_{config['model_name']}",
        f"train{config['data']['train_regime']}",
        f"C{c_value}",
        f"F{env['F']}",
        f"T{env['T']}",
        f"flush{int(env.get('flush_levels', 17))}",
        f"tau{format_number_for_name(config['reward']['money_tau'])}",
        f"seed{config['seed']}",
    ]
    reward_scale = float(config.get("train", {}).get("reward_scale", 1.0))
    if reward_scale != 1.0:
        parts.append(f"rscale{format_number_for_name(reward_scale)}")
    if str(env.get("state_feature_mode", "base")) != "base":
        parts.append(str(env.get("state_feature_mode")))
    if str(env.get("mask_mode", "none")) != "none":
        parts.append(f"mask{str(env.get('mask_mode')).capitalize()}")
    if "eta" in config:
        parts.append(f"eta{format_number_for_name(config['eta'])}")
    imitation = config.get("imitation", {})
    if str(imitation.get("imitation_mode", "none")) == "threshold_pretrain":
        parts.append(
            "imitThreshold"
            f"{int(imitation.get('imitation_episodes', 0))}ep"
            f"{int(imitation.get('imitation_epochs', 1))}"
        )
    imitation_reg_coef = float(imitation.get("imitation_reg_coef", 0.0) or 0.0)
    if imitation_reg_coef > 0.0:
        parts.append(f"imitReg{format_imitation_reg_for_name(imitation_reg_coef)}")
    return "_".join(parts)


def build_run_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    result_root = Path(config["output"]["result_root"]).expanduser().resolve()
    run_stamp = build_run_stamp()
    scenario = build_scenario_name(config)
    run_dir = result_root / config["model_name"] / "runs" / scenario / run_stamp
    checkpoint_dir = run_dir / "checkpoints"
    aggregate_dir = result_root / "aggregates"
    log_dir = Path(config["output"]["log_root"]).expanduser().resolve() / config["model_name"]
    return {
        "result_root": str(result_root),
        "run_stamp": run_stamp,
        "scenario": scenario,
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "aggregate_dir": str(aggregate_dir),
        "log_dir": str(log_dir),
        "config_path": str(run_dir / "config.json"),
        "training_log_path": str(run_dir / "training_log.csv"),
        "validation_log_path": str(run_dir / "validation_log.csv"),
        "results_json_path": str(run_dir / "cross_regime_results.json"),
        "eta_grid_log_path": str(run_dir / "threshold_eta_grid_results.csv"),
        "best_model_path": str(checkpoint_dir / "best_model.pt"),
        "last_model_path": str(checkpoint_dir / "last_model.pt"),
    }


def ensure_run_dirs(paths: Dict[str, Any], save_mode: str) -> None:
    Path(paths["log_dir"]).mkdir(parents=True, exist_ok=True)
    if save_mode == "none":
        return
    Path(paths["run_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["aggregate_dir"]).mkdir(parents=True, exist_ok=True)


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows: List[Dict[str, Any]], path: str | Path, fieldnames: List[str] | None = None) -> None:
    path = Path(path)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_required_run_outputs(
    config: Dict[str, Any],
    paths: Dict[str, Any],
    training_log: List[Dict[str, Any]],
    validation_log: List[Dict[str, Any]],
    cross_regime_results: Dict[str, Any],
) -> None:
    if config["save_mode"] == "none":
        print("Save skipped because save_mode=none.")
        return
    save_json(config, paths["config_path"])
    write_csv(training_log, paths["training_log_path"])
    write_csv(validation_log, paths["validation_log_path"])
    save_json(cross_regime_results, paths["results_json_path"])
    print(f"Saved results to: {paths['run_dir']}")


def list_cross_regime_files(result_root: str | Path) -> List[Path]:
    return sorted(Path(result_root).expanduser().resolve().glob("*/runs/**/cross_regime_results.json"))
