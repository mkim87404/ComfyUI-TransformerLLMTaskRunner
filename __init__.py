from .nodes import TransformerLLMTaskRunner

NODE_CLASS_MAPPINGS = {
    "TransformerLLMTaskRunner": TransformerLLMTaskRunner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TransformerLLMTaskRunner": "Transformer LLM Task Runner",
}

__version__ = "1.0.0"