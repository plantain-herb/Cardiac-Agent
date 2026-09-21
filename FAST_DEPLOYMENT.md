# Fast runtime profile

The `fast` branch preserves the time-horizon portal contract from `main` while
isolating two previously validated acceleration paths:

- Agent: split LLaVA checkpoint served as a Mistral decoder in vLLM plus the
  frozen CLIP/projector/token-embedding `VisionBridge`.
- Report generation: the compact TensorRT MRG challenger, exposed through the
  legacy-compatible `/worker_generate` API on port 21032.

There is deliberately no silent fallback.  If either fast runtime is missing,
startup fails rather than producing an apparently healthy slow deployment.
`main` remains the reference and rollback branch.

## Required runtime inputs

Set these before `./app/start.sh full`:

```bash
export CARDIAC_RUNTIME_PROFILE=fast
export CARDIAC_AGENT_BACKEND=vllm
export CARDIAC_VLLM_PYTHON=/path/to/compatible-vllm-env/bin/python
export CARDIAC_VLLM_MODEL_PATH=/path/to/model_views/mistral_safetensors
export CARDIAC_VLLM_BRIDGE_PATH=/path/to/model_views/vision_bridge
export CARDIAC_MRG_URL=http://127.0.0.1:21032
export CARDIAC_CONDA_PATH=/path/to/anaconda3
export CARDIAC_CONDA_ENV_AGENT=/path/to/service-environment
export CARDIAC_CONDA_ENV_EXPERT=/path/to/service-environment
export CARDIAC_CONDA_ENV_DEMO=/path/to/service-environment
./app/start.sh full
```

The process binds to `127.0.0.1` by default.  Override
`CARDIAC_LISTEN_HOST` only when the network boundary has been reviewed.

## zydb compatibility gate (2026-09-21)

The split checkpoint is already present on zydb at:

```text
/home/qutaiping/nas/dong_explore/cmla_time_horizon_v20260716_r1/models/vllm_LLaVA/model_views/mistral_safetensors
/home/qutaiping/nas/dong_explore/cmla_time_horizon_v20260716_r1/models/vllm_LLaVA/model_views/vision_bridge
```

The accepted reference service currently runs its controller, legacy Agent,
Expert workers, and portal with
`/home/qutaiping/nas/envs/dong_totalseg/bin/python3.11`.  Use that environment
for the three `CARDIAC_CONDA_ENV_*` variables unless a replacement environment
has been independently accepted.  The vLLM Agent remains isolated in
`CARDIAC_VLLM_PYTHON`.

Do **not** point the fast launcher at the existing
`/home/qutaiping/nas/envs/dong_vllm_trt` environment.  It contains vLLM
`0.6.6.post1`, whose `LLM` and input signatures do not support this external
`prompt_embeds` bridge.  The proven bridge needs a compatible vLLM runtime
with `enable_prompt_embeds=True` (the earlier Spark acceptance used vLLM
0.25.0).  Build or migrate that compatible environment first, then rerun the
two-examination acceptance replay.

The compact MRG service is a separate process and must return HTTP 200 from
`${CARDIAC_MRG_URL}/health` before the portal workers start.  Its accepted
launcher source is outside this repository at:

```text
/home/dongzifei/code/cardiac_agent_project/new_model_26_7/scripts/run_challenger_spark.sh
```

It is a speed challenger, not a replacement scientific model: historical
latency improved substantially, but LVEF/LVESV accuracy was mixed.  Keep it in
challenger/shadow status until zydb replay acceptance is recorded.

## Acceptance contract

Fast is deployable only after all of the following pass on the target host:

1. Agent, controller, Expert workers, portal backend, and frontend health.
2. Controller registration for the same model names expected by the portal.
3. Baseline then follow-up upload from the canonical de-identified pair.
4. Browser offer after the second examination for the same patient with a
   different `StudyInstanceUID`.
5. `POST /api/longitudinal/compare` and dashboard rendering.
6. Output parity check against `main`, with latency captured separately.

The current zydb service and canonical pair are recorded in
`../runtime_deployment/ZYDB_TIME_HORIZON_PORTAL.md` in the enclosing project.
