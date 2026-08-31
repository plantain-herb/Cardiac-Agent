"""Public LLaVA package exports.

Keep model imports lazy so lightweight consumers (for example the web API,
which only needs ``llava.conversation``) do not initialize PyTorch and the
transformer stack during startup.
"""

__all__ = ["LlavaLlamaForCausalLM"]


def __getattr__(name):
    if name == "LlavaLlamaForCausalLM":
        from .model import LlavaLlamaForCausalLM

        return LlavaLlamaForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
