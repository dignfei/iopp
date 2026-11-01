# IOP Installation and Reproduction Guide

The commands below assume the current directory as the initial working directory. All paths are relative to it or to the `verl` directory entered later. Linux, an NVIDIA GPU, compatible drivers and CUDA Toolkit, a compiler, and Ninja are required.

## 1. Environment Installation

The following pins the Python/Torch/CUDA combination used by this package. verl and vLLM must still point to the source code included in this package.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.13.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install -r vllm/requirements/build/cuda.txt
VLLM_VERSION_OVERRIDE=0.31.1rc1.dev147+g7914289b7 uv pip install --no-build-isolation -e ./vllm
uv pip install -e ./verl
uv pip install -r verl/recipe/iop/requirements.txt
uv pip install transformers==5.19.0 ray==2.59.0 'tensordict==0.10.0' 'numpy<2' pyarrow==19.0.1 datasets==3.6.0
```

`VLLM_VERSION_OVERRIDE` identifies the version of a source snapshot without `.git`. Installing vLLM builds its extensions; local validation can reuse installed extensions that match this package's native build sources. Training uses SDPA by default; FlashAttention2 can also be used.

Enter the training working directory; run all subsequent commands here:

```bash
cd verl
export PYTHONPATH="$PWD:$PWD/paper:$PWD/../vllm${PYTHONPATH:+:$PYTHONPATH}"
export KLSTATS_CODE_DIR="$PWD/paper"
export KLSTATS_ENABLE=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
if [ -d /usr/lib/wsl/lib ]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi
python -c 'import torch,verl,vllm; print(torch.__version__,torch.cuda.is_available()); print(verl.__file__); print(vllm.__file__)'
```

`VLLM_USE_FLASHINFER_SAMPLER=0` uses the existing sampling implementation, preserving temperature, TopP, and TopK while avoiding a dependency on an additional FlashInfer JIT compilation environment.

## 2. Data

The included `../data/deepmath_103k_train.parquet` contains 103,020 DeepMath question-answer training records, approximately 8.87 MB. Only two records with empty answers were removed from the original data. It contains neither full-length responses nor the OpenCodeReasoning code data used in the paper; the full data are too large. Download the data, code, corresponding execution sandbox, and independent evaluation sets separately.

RL Parquet files require at least the following fields:

- `prompt`: `[{"role":"user","content":"Question and required answer format"}]`;
- `data_source`: a data source supported by the reward verifier; the included math file uses `math`;
- `reward_model`: `{"style":"rule","ground_truth":"Ground-truth answer"}`.

The math prompts in this package require the final answer to start with `Answer:`, matching the built-in math verifier. The IOP reward adapter maps outcomes to 1 for correct and 0 for incorrect. For other tasks, specify a custom binary verifier through `reward.custom_reward_function.path` and `name`; formatting scores or partial credit must not be treated as correctness.

## 3. Optional Stage 1: Repair SFT

Save the prepared data to `../data/repairs.jsonl`. Each line has the following format:

```json
{"question":"Question","failed":"Original failed reasoning","reference":"Correct reference reasoning for the same question","repaired":"Complete, verified repaired reasoning","verified":true}
```

`question` can also be a list of chat messages. The input contains the question, failed trajectory, and correct reference; only the assistant response corresponding to `repaired` receives SFT supervision. The data must already have been verified; the conversion program does not label failed reasoning as a supervision target.

```bash
python -m recipe.iop.prepare_sft \
  --input ../data/repairs.jsonl \
  --output ../data/repairs.parquet
```

Specify the model and the actual number of GPUs. The example below uses Qwen3-32B on a single node with 8 GPUs; `MODEL_PATH` can be changed to a downloaded local model directory.

```bash
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-32B}"
export N_GPUS="${N_GPUS:-8}"
export SFT_DIR="${SFT_DIR:-../runs/iop-sft}"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$N_GPUS" \
  -m verl.trainer.sft_trainer \
  model.path="$MODEL_PATH" \
  model.use_remove_padding=false \
  +model.override_config.attn_implementation=sdpa \
  model.enable_gradient_checkpointing=true \
  engine.strategy=fsdp2 \
  +engine.use_remove_padding=false \
  engine.use_torch_compile=false \
  engine.param_offload=true engine.optimizer_offload=true \
  data.train_files=../data/repairs.parquet data.val_files=null \
  data.train_batch_size=8 data.micro_batch_size_per_gpu=1 \
  data.use_dynamic_bsz=true data.max_token_len_per_gpu=32768 \
  data.max_length=32768 data.truncation=error data.num_workers=0 \
  optim.lr=1e-5 optim.lr_warmup_steps=6 \
  checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
  trainer.nnodes=1 trainer.n_gpus_per_node="$N_GPUS" \
  trainer.logger='[console]' trainer.total_epochs=1 \
  trainer.resume_mode=disable trainer.default_local_dir="$SFT_DIR"
```

After training, use the weights actually exported at the final step to continue with RL:

```bash
export MODEL_PATH="$(find "$SFT_DIR" -type d -path '*/global_step_*/huggingface' | sort -V | tail -n 1)"
test -n "$MODEL_PATH"
```

If SFT is skipped, set `MODEL_PATH` directly to the base model; this corresponds to the paper's no-cold-start variant.

## 4. Stage 2: IOP-core Reinforcement Learning

The following command uses the included math data and does not require an evaluation file. Auditing is disabled by default, corresponding to the paper's no-audit variant. See the next section for enabling auditing in the full configuration.

```bash
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-32B}"
export N_GPUS="${N_GPUS:-8}"
export NNODES="${NNODES:-1}"
export TP="${TP:-2}"
export IOP_LR="${IOP_LR:-1e-6}"
export TRAIN_FILE="${TRAIN_FILE:-../data/deepmath_103k_train.parquet}"
export RUN_DIR="${RUN_DIR:-../runs/iop-core}"
python -m recipe.iop.main_iop \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  data.train_files="$TRAIN_FILE" data.val_files=null \
  data.filter_overlong_prompts=true data.filter_overlong_prompts_workers=4 \
  data.train_batch_size=64 data.max_prompt_length=16384 data.max_response_length=4096 \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr="$IOP_LR" \
  actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
  actor_rollout_ref.rollout.max_model_len=32768 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.generation_config=vllm \
  actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=20 \
  actor_rollout_ref.rollout.val_kwargs.n=32 \
  actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
  trainer.nnodes="$NNODES" trainer.n_gpus_per_node="$N_GPUS" \
  trainer.logger='[console]' trainer.resume_mode=disable \
  trainer.val_before_train=false trainer.test_freq=-1 \
  trainer.total_epochs=1 trainer.total_training_steps=800 trainer.save_freq=25 \
  trainer.default_local_dir="$RUN_DIR"
```

The paper's main comparison matches end-to-end GPU-hour budgets; a fixed 800 steps is not equivalent to reproducing that budget.

First run `ray start --head --port=6379` on the head node, then run `ray start --address=<HEAD_NODE_IP>:6379` on the other nodes. On the head node, set `RAY_ADDRESS=auto`, `NNODES=2`, and `N_GPUS=8` per node, and set TP according to available GPU memory. Before starting Ray, enter the corresponding `verl` working directory on every node. All nodes must use the same source code and environment, with model, data, and output paths that can be shared across nodes.

## 5. Enabling Independent Auditing

`recipe/iop/auditor.py` is provided to call a separately deployed audit model through an HTTP service compatible with `/v1/chat/completions`. It receives the question, failed trajectory, candidate repair, and correct reference, and accepts only the binary decisions `{"approved":0}` or `{"approved":1}`. HTTP errors, empty outputs, and invalid decisions stop the run; they do not automatically approve a candidate.

First deploy gpt-oss-120b, the model used in the paper, on separate auditing resources. For example:

```bash
vllm serve /models/gpt-oss-120b \
  --served-model-name gpt-oss-120b --tensor-parallel-size 8 \
  --max-model-len 65536 --port 8000
```

Set the following in the training terminal:

```bash
export IOP_AUDIT_BASE_URL="http://AUDIT_NODE_IP:8000/v1"
export IOP_AUDIT_MODEL="gpt-oss-120b"
```

Then append the following to the RL command in the previous section:

```text
iop.audit_path=pkg://recipe.iop.auditor iop.audit_name=compute_audit
```
