#!/usr/bin/env python3
"""Run the official nine-task RM-Bench zero-shot suite across one or more GPUs."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "rmbench" / "eval_rmbench_single.py"
POLL_INTERVAL_SEC = 2
TERMINATE_TIMEOUT_SEC = 10


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.output_dir",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [
        override
        for override in HydraConfig.get().overrides.task
        if not _is_blocked_override(override)
    ]


def _parse_result(result_file: Path) -> tuple[float, float | None]:
    if not result_file.is_file():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    success_matches = re.findall(r"^Success Rate:\s*([-+0-9.eE]+)\s*$", text, re.MULTILINE)
    reward_matches = re.findall(r"^Reward:\s*([-+0-9.eE]+)\s*$", text, re.MULTILINE)
    if not success_matches:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    success_rate = float(success_matches[-1])
    reward = float(reward_matches[-1]) if reward_matches else None
    return success_rate, reward


@dataclass
class RunningTask:
    task_name: str
    gpu_id: int
    process: subprocess.Popen[str]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_rmbench.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None")
    if not SINGLE_ENTRY.is_file():
        raise FileNotFoundError(f"Single-task entrypoint not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    configured_task = cfg.EVALUATION.task_name
    if configured_task is None or str(configured_task).strip() == "":
        tasks = [str(task) for task in cfg.MULTIRUN.tasks]
    else:
        tasks = [str(configured_task)]
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("`MULTIRUN.tasks` must contain unique task names")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if num_gpus <= 0 or max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.num_gpus` and `max_tasks_per_gpu` must be positive")
    gpu_ids = list(range(num_gpus))

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "manager_config.yaml")
    manager_log = output_dir / "manager.log"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    failed_tasks_file = output_dir / "failed_tasks.txt"
    extra_overrides = _collect_worker_overrides()

    pending = deque(tasks)
    running: list[RunningTask] = []
    results: dict[str, dict[str, float | None]] = {}
    failures: list[dict[str, Any]] = []

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def gpu_running_count(gpu_id: int) -> int:
        return sum(
            state.gpu_id == gpu_id and state.process.poll() is None for state in running
        )

    def launch(task_name: str, gpu_id: int) -> RunningTask:
        task_output = output_dir / task_name
        command = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={ckpt_path}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.output_dir={task_output}",
            *extra_overrides,
        ]
        log(f"launch task={task_name} gpu={gpu_id} command={' '.join(command)}")
        process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), text=True)
        return RunningTask(task_name=task_name, gpu_id=gpu_id, process=process)

    def fill_gpu(gpu_id: int) -> None:
        while pending and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            running.append(launch(pending.popleft(), gpu_id))

    def terminate_running() -> None:
        active = [state for state in running if state.process.poll() is None]
        for state in active:
            log(f"terminate task={state.task_name} gpu={state.gpu_id}")
            state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in active:
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"kill task={state.task_name} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()

    def write_outputs() -> None:
        with summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["task_name", "success_rate", "reward"])
            for task_name in tasks:
                result = results.get(task_name, {})
                writer.writerow(
                    [task_name, result.get("success_rate"), result.get("reward")]
                )
            valid_rates = [
                float(result["success_rate"])
                for result in results.values()
                if result.get("success_rate") is not None
            ]
            mean_rate = sum(valid_rates) / len(valid_rates) if valid_rates else None
            writer.writerow(["__overall__", mean_rate, None])

        payload = {
            "tasks": tasks,
            "per_task": results,
            "mean_success_rate": mean_rate,
            "failures": failures,
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with failed_tasks_file.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")

    log(
        f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} output={output_dir}"
    )
    for gpu_id in gpu_ids:
        fill_gpu(gpu_id)

    try:
        while running:
            progressed = False
            for state in list(running):
                return_code = state.process.poll()
                if return_code is None:
                    continue
                progressed = True
                running.remove(state)
                if return_code != 0:
                    failure = {
                        "task_name": state.task_name,
                        "gpu_id": state.gpu_id,
                        "return_code": return_code,
                        "reason": "worker_failed",
                    }
                    failures.append(failure)
                    log(f"failed task={state.task_name} gpu={state.gpu_id} rc={return_code}")
                else:
                    result_file = output_dir / state.task_name / "official_result" / "_result.txt"
                    try:
                        success_rate, reward = _parse_result(result_file)
                    except Exception as exc:
                        failures.append(
                            {
                                "task_name": state.task_name,
                                "gpu_id": state.gpu_id,
                                "return_code": return_code,
                                "reason": f"result_parse_failed: {exc!r}",
                            }
                        )
                        log(f"result parse failed task={state.task_name}: {exc!r}")
                    else:
                        results[state.task_name] = {
                            "success_rate": success_rate,
                            "reward": reward,
                        }
                        log(
                            f"complete task={state.task_name} gpu={state.gpu_id} "
                            f"success_rate={success_rate}"
                        )
                fill_gpu(state.gpu_id)
            if running and not progressed:
                time.sleep(POLL_INTERVAL_SEC)
    except KeyboardInterrupt:
        log("manager interrupted")
        terminate_running()
        write_outputs()
        raise

    write_outputs()
    if failures:
        raise RuntimeError(
            f"RM-Bench manager completed with {len(failures)} failure(s). "
            f"See {failed_tasks_file}"
        )
    log(f"manager complete mean_success_rate={payload_mean(results)}")


def payload_mean(results: dict[str, dict[str, float | None]]) -> float | None:
    """Return the mean of available task success rates for the final log line."""

    rates = [
        float(result["success_rate"])
        for result in results.values()
        if result.get("success_rate") is not None
    ]
    return sum(rates) / len(rates) if rates else None


if __name__ == "__main__":
    main()
