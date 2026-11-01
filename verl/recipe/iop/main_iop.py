# Copyright 2026 Individual Contributor: d fei
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Launch independent IOP training with the standard verl TaskRunner."""

import os
from pathlib import Path

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import hydra
import ray
from omegaconf import open_dict

from recipe.iop.iop_ray_trainer import IOPRayPPOTrainer
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device


class IOPTaskRunner(TaskRunner):
    trainer_class = IOPRayPPOTrainer


def prepare_runtime(config):
    """Configure the same driver/worker environment for training and GPU acceptance."""
    os.environ.update(KLSTATS_ENABLE="0", RAY_ENABLE_UV_RUN_RUNTIME_ENV="0")
    os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    root = Path(__file__).resolve().parents[2]
    with open_dict(config):
        runtime = config.ray_kwargs.ray_init.get("runtime_env") or {}
        env = dict(runtime.get("env_vars") or {})
        env.update(KLSTATS_ENABLE="0", KLSTATS_CODE_DIR=str(root / "paper"), RAY_ENABLE_UV_RUN_RUNTIME_ENV="0")
        env.pop("RAY_RUNTIME_ENV_HOOK", None)
        for name in (
            "VLLM_USE_FLASHINFER_SAMPLER", "IOP_AUDIT_BASE_URL", "IOP_AUDIT_MODEL",
            "IOP_AUDIT_MAX_TOKENS", "IOP_AUDIT_TIMEOUT", "IOP_AUDIT_API_KEY_FILE",
        ):
            if name in os.environ:
                env[name] = os.environ[name]
        env["PYTHONPATH"] = os.pathsep.join(
            (str(root), str(root / "paper"), env.get("PYTHONPATH", os.getenv("PYTHONPATH", "")))
        )
        if "LD_LIBRARY_PATH" in os.environ:
            env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
        runtime["env_vars"] = env
        config.ray_kwargs.ray_init.runtime_env = runtime
    return config


@hydra.main(config_path="config", config_name="iop_trainer", version_base=None)
def main(config):
    config = prepare_runtime(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(IOPTaskRunner))


if __name__ == "__main__":
    main()
