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
export CARDIAC_VLLM_GPU_MEMORY_UTILIZATION=0.72
export CARDIAC_MRG_URL=http://127.0.0.1:21032
export CARDIAC_CONDA_PATH=/path/to/anaconda3
export CARDIAC_CONDA_ENV_AGENT=/path/to/service-environment
export CARDIAC_CONDA_ENV_EXPERT=/path/to/service-environment
export CARDIAC_CONDA_ENV_DEMO=/path/to/service-environment
export CARDIAC_GPU_AGENT=0
export CARDIAC_GPU_SEG_2CH=1 CARDIAC_GPU_SEG_4CH=1
export CARDIAC_GPU_SEG_SA=2 CARDIAC_GPU_SEG_LGE=2
export CARDIAC_GPU_CDS=3 CARDIAC_GPU_NICMS=3
./app/start.sh full
```

The process binds to `127.0.0.1` by default.  Override
`CARDIAC_LISTEN_HOST` only when the network boundary has been reviewed.

## Accepted zydb deployment (2026-09-21)

The split checkpoint is already present on zydb at:

```text
/home/qutaiping/nas/dong_explore/cmla_time_horizon_v20260716_r1/models/vllm_LLaVA/model_views/mistral_safetensors
/home/qutaiping/nas/dong_explore/cmla_time_horizon_v20260716_r1/models/vllm_LLaVA/model_views/vision_bridge
```

The live fast release and its isolated vLLM environment are:

```text
/home/qutaiping/nas/dongzifei_runtime_explore/cardiac_agent_service_1f74107_fast_20260921
/home/qutaiping/nas/envs/dong_cardiac_vllm_fast_0102
```

The accepted versions are vLLM 0.10.2, PyTorch 2.8.0+cu128, Transformers
4.55.4, Tokenizers 0.21.4, and Hugging Face Hub 0.36.2.  The latter three are
intentionally pinned: Transformers 5.17 removed a tokenizer attribute used by
vLLM 0.10.2.  The Agent uses GPU 0 with memory utilization 0.72; 0.55 is too
small for the 13.5-GiB decoder plus bridge and KV cache on a 24-GiB 4090.

The controller, incumbent Expert workers, and portal use the existing venv
`/home/qutaiping/nas/envs/dong_cardiac_portal`.  It resolves to the established
base Python but also contributes portal-only dependencies, so it must be
activated as a venv rather than replaced with the base Conda prefix.

Do **not** point the Agent at the old
`/home/qutaiping/nas/envs/dong_vllm_trt` environment.  Its vLLM 0.6.6.post1
does not accept external `prompt_embeds`.

The compact MRG service is a separate process and must return HTTP 200 from
`${CARDIAC_MRG_URL}/health` before the portal workers start.  The live zydb
release and environment are:

```text
/home/qutaiping/nas/dongzifei_runtime_explore/cardiac_mrg_fast_zydb_20260921
/home/qutaiping/nas/envs/dong_cardiac_mrg_trt_fast
```

Its six TensorRT 10.7 engines were rebuilt on RTX 4090 and all passed the
frozen parity vectors with argmax agreement 1.0.  The portal converts DICOM to
NIfTI only inside the session cache for this fast profile; the challenger then
enforces that allowed-root boundary.  Its scientific status remains
challenger: runtime acceptance does not supersede the frozen accuracy caveats.

The authoritative lifecycle wrapper is copied to zydb at:

```text
/home/qutaiping/nas/dongzifei_runtime_explore/zydb_fast_service_20260921.sh
```

Use `start`, `stop`, `restart`, or `status`.  Its tracked source is
`../runtime_deployment/scripts/zydb_fast_service_20260921.sh` in the enclosing
project.

## Acceptance contract

Fast is deployable only after all of the following pass on the target host:

1. Agent, controller, Expert workers, portal backend, and frontend health.
2. Controller registration for the same model names expected by the portal.
3. Baseline then follow-up upload from the canonical de-identified pair.
4. Browser offer after the second examination for the same patient with a
   different `StudyInstanceUID`.
5. `POST /api/longitudinal/compare` and dashboard rendering.
6. TensorRT frozen-vector parity; scientific comparison against `main` remains
   a separate evaluation question.

The 2026-09-21 zydb acceptance passed all 21 repository tests, registered ten
controller models, completed a real vLLM generation, and replayed the canonical
pair in one session.  Baseline and follow-up each returned 70 metrics; the
second upload returned an enabled longitudinal offer; the comparison endpoint
returned 36 rows and six flagged metrics.  The two upload calls took 40.7 s and
49.2 s respectively in that cold acceptance run.

The current zydb service and canonical pair are recorded in
`../runtime_deployment/ZYDB_TIME_HORIZON_PORTAL.md` in the enclosing project.

## Display correctness follow-up (2026-09-21)

The live-service review found and fixed two independent presentation faults:

- `_get_metric_status` previously returned `normal` for every metric except
  LV/RV EF.  The report now classifies all values that have an explicit UI
  reference interval as `low`, `normal`, or `high`; measurements without a
  defined interval are `unknown` instead of being presented as normal.  The
  browser repeats the interval check so cached report payloads are rendered
  correctly after a refresh.
- The legacy cine-4CH model requires an in-plane Y flip before inference.  The
  worker had dropped that preprocessing step, producing masks over the chest
  wall despite matching NIfTI metadata.  It now applies the model-space flip
  and restores the prediction to the source image grid before saving.  Overlay
  generation also rejects differences in size, spacing, origin, or direction
  instead of silently drawing an invalid mask.

The diagnosis was reproduced on the live 4090 input and verified with a
controlled before/after three-frame overlay.  Keep both the orientation
round-trip tests and the overlay geometry tests in the deployment gate.

For clients whose operating system reserves local ports 8080/8005, the
frontend accepts a validated `apiPort` query parameter.  For example, forward
the UI and API to local ports 18080/18005, then open
`http://127.0.0.1:18080/?apiPort=18005`.  Without this parameter the established
8005 API contract remains unchanged.
