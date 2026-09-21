from app.config import EXPERT_WORKERS, RUNTIME_PROFILE
from serve.vllm_utils import normalize_image_tokens


def test_fast_branch_defaults_to_fast_mrg():
    assert RUNTIME_PROFILE == "fast"
    assert EXPERT_WORKERS["MRGWorker"].endswith(":21032")


def test_vllm_worker_normalizes_image_tokens():
    prompt = "USER: inspect this study\nASSISTANT:"
    normalized = normalize_image_tokens(prompt, 2)
    assert normalized.count("<image>") == 2
    assert normalized.startswith("USER:")
