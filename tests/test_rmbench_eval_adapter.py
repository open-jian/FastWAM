from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.rmbench.eval_rmbench_single import _resolve_official_result_tag
from experiments.rmbench.fastwam_policy import deploy_policy as rmbench_policy
from experiments.rmbench.run_official_eval import _normalize_official_metrics
from experiments.rmbench.run_rmbench_manager import _parse_result, payload_mean
from experiments.robotwin.fastwam_policy import deploy_policy as robotwin_policy


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TASKS = [
    "observe_and_pickup",
    "rearrange_blocks",
    "put_back_block",
    "swap_blocks",
    "swap_T",
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "press_button",
]


def test_rmbench_policy_reuses_robotwin_implementation() -> None:
    assert rmbench_policy.WorldActionRobotWinPolicy is robotwin_policy.WorldActionRobotWinPolicy
    assert rmbench_policy.encode_obs is robotwin_policy.encode_obs
    assert rmbench_policy.eval is robotwin_policy.eval
    assert rmbench_policy.reset_model is robotwin_policy.reset_model


def test_rmbench_get_model_forwards_real_checkpoint(monkeypatch) -> None:
    captured = {}

    def fake_get_model(args):
        captured.update(args)
        return object()

    monkeypatch.setattr(robotwin_policy, "get_model", fake_get_model)
    rmbench_policy.get_model(
        {
            "ckpt_setting": "short-result-tag",
            "checkpoint_path": "/tmp/real-checkpoint.pt",
        }
    )
    assert captured["ckpt_setting"] == "/tmp/real-checkpoint.pt"


def test_rmbench_config_preserves_zero_shot_robotwin_recipe() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="sim_rmbench")

    assert cfg.EVALUATION.task_config == "demo_clean"
    assert cfg.EVALUATION.instruction_type == "unseen"
    assert cfg.seed == 0
    assert cfg.EVALUATION.action_horizon == 32
    assert cfg.EVALUATION.replan_steps == 24
    assert cfg.EVALUATION.num_inference_steps == 10
    assert cfg.EVALUATION.dataset_stats_path.endswith(
        "robotwin_uncond_3cam_384_dataset_stats.json"
    )
    assert cfg.data.train.num_frames == 33
    assert cfg.data.train.concat_multi_camera == "robotwin"
    assert list(cfg.MULTIRUN.tasks) == OFFICIAL_TASKS


def test_rmbench_official_result_tags_are_isolated_and_safe() -> None:
    cfg = OmegaConf.create({"seed": 7, "EVALUATION": {"official_result_tag": None}})
    assert _resolve_official_result_tag(cfg, "checkpoint") == "checkpoint_seed7"

    cfg.EVALUATION.official_result_tag = "checkpoint_seed7_run2"
    assert _resolve_official_result_tag(cfg, "checkpoint") == "checkpoint_seed7_run2"

    cfg.EVALUATION.official_result_tag = "../escape"
    with pytest.raises(ValueError, match="single directory"):
        _resolve_official_result_tag(cfg, "checkpoint")


def test_short_eval_metrics_survive_official_hardcoded_denominator() -> None:
    successes, reward_sum = _normalize_official_metrics(2, 7.5, episode_limit=5)
    assert successes / 100 == 2 / 5
    assert reward_sum / 100 == 7.5 / 5
    with pytest.raises(ValueError, match="positive"):
        _normalize_official_metrics(0, 0, episode_limit=0)


def test_manager_parses_official_result(tmp_path: Path) -> None:
    result_file = tmp_path / "_result.txt"
    result_file.write_text(
        "Instruction Type: unseen\n\nSuccess Rate: 0.3\n\nReward: 1.25\n",
        encoding="utf-8",
    )
    assert _parse_result(result_file) == (0.3, 1.25)
    assert payload_mean(
        {
            "task_a": {"success_rate": 0.3, "reward": 1.0},
            "task_b": {"success_rate": 0.5, "reward": None},
        }
    ) == pytest.approx(0.4)
