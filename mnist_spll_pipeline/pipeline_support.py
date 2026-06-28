from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import yaml


StageFn = Callable[[Dict[str, Any]], None]
StageSpec = Tuple[str, str]


class TerminalProgressBar:
    """Small dependency-free terminal progress indicator used by both pipelines."""

    def __init__(
        self,
        total: int,
        *,
        desc: str = "Progress",
        unit: str = "items",
        enabled: bool = True,
        width: int = 28,
    ) -> None:
        self.total = max(int(total), 0)
        self.desc = desc
        self.unit = unit
        self.enabled = enabled
        self.width = max(int(width), 10)
        self.current = 0
        self.started_at = time.perf_counter()
        self._last_line_len = 0
        if self.enabled:
            self._render()

    def update(self, step: int = 1, *, postfix: str = "") -> None:
        self.current = min(self.total, self.current + int(step))
        if self.enabled:
            self._render(postfix=postfix)

    def finish(self, *, postfix: str = "done") -> None:
        self.current = self.total
        if self.enabled:
            self._render(postfix=postfix)
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _render(self, *, postfix: str = "") -> None:
        columns = shutil.get_terminal_size((100, 20)).columns
        usable_width = min(self.width, max(10, columns // 4))
        fraction = min(max(self.current / self.total, 0.0), 1.0) if self.total else 1.0
        filled = int(round(usable_width * fraction))
        bar = "#" * filled + "-" * (usable_width - filled)

        elapsed = time.perf_counter() - self.started_at
        rate = self.current / elapsed if elapsed > 0 and self.current > 0 else 0.0
        eta_text = ""
        if rate > 0 and self.current < self.total:
            eta_text = f" ETA {(self.total - self.current) / rate:5.1f}s"

        line = (
            f"\r{self.desc}: [{bar}] {self.current}/{self.total} "
            f"({fraction * 100:5.1f}%) {self.unit}{eta_text}"
        )
        if postfix:
            line += f" | {postfix}"
        line += " " * max(0, self._last_line_len - len(line))
        sys.stdout.write(line)
        sys.stdout.flush()
        self._last_line_len = len(line)


def stage_message(current: int, total: int, message: str) -> None:
    print(f"\n[{current}/{total}] {message}", flush=True)


def _deep_merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge mappings while replacing non-mapping values and lists."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Top-level YAML config must be a mapping.")

    extends_value = config.get("extends")
    if extends_value:
        base_path = Path(str(extends_value))
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        base_path = base_path.expanduser().resolve()
        with base_path.open("r", encoding="utf-8") as handle:
            base_config = yaml.safe_load(handle)
        if not isinstance(base_config, dict):
            raise ValueError(f"Base YAML config must be a mapping: {base_path}")
        if base_config.get("extends"):
            raise ValueError(
                "Only one config inheritance level is supported. "
                f"{config_path} extends {base_path}, but that base config also defines extends."
            )
        config = _deep_merge_config(base_config, config)
        config["_base_config_path"] = str(base_path)

    config.pop("extends", None)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def _config_payload(config: Mapping[str, Any]) -> Dict[str, Any]:
    payload = copy.deepcopy(dict(config))
    payload.pop("_config_path", None)
    payload.pop("_config_dir", None)
    payload.pop("_base_config_path", None)
    return payload


def save_config(config: Mapping[str, Any], destination: str | Path) -> None:
    destination = Path(destination)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_config_payload(config), handle, sort_keys=False)


def resolve_path(config: Mapping[str, Any], raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(str(config["_config_dir"])).joinpath(path).resolve()


def ensure_dir(path: str | Path) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_stage_metadata(
    config: Mapping[str, Any],
    stage_name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build reproducibility metadata without importing PyTorch at module load time."""

    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        torch_version = "unavailable"

    payload: Dict[str, Any] = {
        "stage": stage_name,
        "created_at_utc": utc_now_iso(),
        "seed": int(config.get("seed", 42)),
        "config_path": str(config.get("_config_path", "")),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch_version,
    }
    if extra:
        payload.update(extra)
    return payload


def load_stage_fn(stages: Mapping[str, StageSpec], stage_name: str) -> StageFn:
    module_name, function_name = stages[stage_name]
    return getattr(import_module(module_name), function_name)


def run_stage_sequence(
    config: Dict[str, Any],
    *,
    stages: Mapping[str, StageSpec],
    order: Sequence[str],
    heading: str,
) -> None:
    for stage_name in order:
        print(f"\n=== {heading}: {stage_name} ===", flush=True)
        load_stage_fn(stages, stage_name)(config)


def run_pipeline_cli(
    *,
    stages: Mapping[str, StageSpec],
    order: Sequence[str],
    description: str,
    config_help: str,
    heading: str,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help=config_help)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["all", *stages.keys()],
        help="Which pipeline stage to run. Default: all",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.stage == "all":
        run_stage_sequence(config, stages=stages, order=order, heading=heading)
    else:
        load_stage_fn(stages, args.stage)(config)


def run_configured_stage_cli(stage_fn: StageFn, *, description: str, config_help: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help=config_help)
    args = parser.parse_args()
    stage_fn(load_config(args.config))
