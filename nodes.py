import os
import gc
import folder_paths
import torch
import comfy.model_management as model_management
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# To accept any type for the optional dynamic prompt formatting inputs (required trick for ComfyUI graph compatibility, may not work with reroutes)
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

anyType = AnyType("*")

# Model root (standard folder used by many ComfyUI LLM nodes)
MODEL_ROOT = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(MODEL_ROOT, exist_ok=True)
folder_paths.add_model_folder_path("LLM", MODEL_ROOT)

# Preset models + grows with automatic discovery of any locally downloaded model folders
MODEL_LIST = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b",
    "google/gemma-2-2b",
    "google/recurrentgemma-9b-it",
    "google/recurrentgemma-2b-it",
    "google/gemma-7b-it",
    "mistralai/Mistral-7B-v0.1",
    "HuggingFaceTB/SmolLM2-135M",
]

# Append any extra model folders the user already downloaded
for folder in [f for f in os.listdir(MODEL_ROOT) if os.path.isdir(os.path.join(MODEL_ROOT, f))]:
    if not any(m.endswith(folder) for m in MODEL_LIST):
        MODEL_LIST.append(folder)

class TransformerLLMTaskRunner:
    """
    A dependency-safe, memory-conscious node for running basic transformer LLMs inside ComfyUI.
    • Uses only torch + transformers (already present in ComfyUI installs), without pinning or downgrading currently installed versions.
    • Robust & device-agnostic VRAM/RAM cleanup after each run with option to keep model loaded.
    • Supports dynamic prompt formatting with up to 6 dynamic inputs of any type, auto converted to string.
    • Automatic model download, custom model id support, auto attention fallbacks, chat template support.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": ("STRING", {
                    "multiline": True,
                    "default": "Write your LLM task here.\nOptionally use {arg0}, {arg1}... placeholders to inject dynamic variables of any type from other nodes.",
                    "tooltip": "Enter full LLM prompt. Inject dynamic variables with {arg0}, {arg1}..."
                }),
                "model": (MODEL_LIST, {
                    "default": MODEL_LIST[0],
                    "tooltip": "Preset or custom model folders found in ComfyUI/models/LLM"
                }),
                "custom_model_hf_id": ("STRING", {
                    "default": "",
                    "tooltip": "Copy & Paste any Hugging Face LLM repo ID here to override the model dropdown selection e.g. Qwen/Qwen2.5-7B-Instruct"
                }),
                "dtype": (["auto", "float32", "float16", "bfloat16", "float64"], {
                    "default": "auto",
                    "tooltip": "Data type for model weights. 'auto' respects model config."
                }),
                "attn_implementation": (["auto", "eager", "sdpa"], {
                    # More advanced attention mechanisms like "flash_attention_2" / Sage Attention / xformers / will require custom modification to this custom node, and likely require specific version lock downs for cuda+torch+transformer stacks that may compromise your ComfyUI environment or other installed custom nodes if not set up correctly.
                    "default": "auto",
                    "tooltip": "Attention backend. 'auto' respects model config, 'eager' is the safest but slowest fallback, 'sdpa' is faster and requires no extra pip install (shipped with PyTorch)."
                }),
                "device_map": (["auto", "sequential", "cpu"], {
                    "default": "auto",
                    "tooltip": "Defines how to load/offload model layers. 'auto' for smart GPU+CPU offload, 'cpu' for low VRAM setup."
                }),
                "max_new_tokens": ("INT", {
                    "default": 1024, "min": 1, "max": 4096,
                    "tooltip": "Maximum tokens the LLM may generate. Higher value means longer output cap and longer maximum wait time for output. Bounded to 1–4096 for now."
                }),
                "trust_remote_code": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only enable for models that explicitly require it in their HF card."
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Set True for faster re-runs (model stays in memory + still clears cache), and False for full model unload + max VRAM/RAM recovery after each node execution."
                }),
            },
            "optional": {
                "arg0": (anyType, {"forceInput": True, "tooltip": "Optional dynamic input to replace {arg0} in the task prompt, auto converted to string."}),
                "arg1": (anyType, {"forceInput": True, "tooltip": "Optional dynamic input to replace {arg1} in the task prompt, auto converted to string."}),
                "arg2": (anyType, {"forceInput": True, "tooltip": "Optional dynamic input to replace {arg2} in the task prompt, auto converted to string."}),
                "arg3": (anyType, {"forceInput": True, "tooltip": "Optional dynamic input to replace {arg3} in the task prompt, auto converted to string."}),
                "arg4": (anyType, {"forceInput": True, "tooltip": "Optional dynamic input to replace {arg4} in the task prompt, auto converted to string."}),
                "arg5": (anyType, {"forceInput": True, "tooltip": "Optional dynamic input to replace {arg5} in the task prompt, auto converted to string."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("The LLM generated text output, with the input prompt removed.",)
    FUNCTION = "run_task"
    CATEGORY = "LLM/Transformer"
    DESCRIPTION = "Offload any task to a local transformer LLM. e.g. prompt enhancement, reasoning, translation, calculation, text summarization, JSON manipulation, etc."

    def load_model(self, model_id: str, dtype: str, trust_remote_code: bool, attn_impl: str, device_map: str):
        if self.model is not None:
            return

        model_name = model_id.rsplit("/", 1)[-1]
        model_path = os.path.join(MODEL_ROOT, model_name)

        if not os.path.exists(model_path):
            print(f"Downloading {model_id} → {model_path}")
            snapshot_download(repo_id=model_id, local_dir=model_path, local_dir_use_symlinks=False)

        torch_dtype = None if dtype == "auto" else getattr(torch, dtype)
        kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
            "trust_remote_code": trust_remote_code,
        }
        if attn_impl != "auto":
            kwargs["attn_implementation"] = attn_impl

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            if device_map == "cpu":
                self.device = "cpu"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(" - OOM: Falling back to CPU + float32")
                kwargs.update({"device_map": "cpu", "torch_dtype": torch.float32})
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
                self.device = "cpu"
            elif attn_impl != "sdpa":
                print(" - Attn fallback → sdpa")
                kwargs["attn_implementation"] = "sdpa"
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            else:
                print(" - Attn fallback → eager")
                kwargs["attn_implementation"] = "eager"
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    def run_task(self, task, model, custom_model_hf_id, dtype, attn_implementation,
                 device_map, max_new_tokens, trust_remote_code, keep_model_loaded,
                 arg0="", arg1="", arg2="", arg3="", arg4="", arg5=""):

        # AnyType → string conversion for the optional dynamic prompt formatting inputs
        arg0 = "" if arg0 is None else str(arg0)
        arg1 = "" if arg1 is None else str(arg1)
        arg2 = "" if arg2 is None else str(arg2)
        arg3 = "" if arg3 is None else str(arg3)
        arg4 = "" if arg4 is None else str(arg4)
        arg5 = "" if arg5 is None else str(arg5)

        print("### TransformerLLMTaskRunner • Loading LLM ###")
        try:
            model_id = custom_model_hf_id.strip() or model
            self.load_model(model_id, dtype, trust_remote_code, attn_implementation, device_map)

            with torch.no_grad():
                full_prompt = task.format(arg0=arg0, arg1=arg1, arg2=arg2, arg3=arg3, arg4=arg4, arg5=arg5).strip()

                if hasattr(self.tokenizer, "apply_chat_template"):
                    messages = [{"role": "user", "content": full_prompt}]
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                else:
                    inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

                input_len = inputs["input_ids"].shape[-1]
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                result = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

            return (result,)
        except Exception as e:
            print(f" - Error during LLM load/run:\n{e}")
            return ("",)
        finally:
            print("### TransformerLLMTaskRunner • Full memory cleanup ###")
            try:
                # Unload the model
                if not keep_model_loaded:
                    # Deliberate design choice:
                    # 1. do not call self.model.to("cpu") to unload the model from gpu because it conflicts with accelerate/device_map="auto" hooks and can leak memory or raise warnings.
                    # Instead, simply deleting the Python reference and invoking py garbage collection is the safest, most reliable method that works for sharded models, pure GPU, pure CPU, or sequential – no exceptions.
                    # 2. do not overuse torch.cuda.synchronize() as it will block execution until all queued CUDA work is finished, adding unnecessary latency and compromises per-request throughput which is not strictly necessary for freeing memory and moving on.
                    if self.model is not None:
                        del self.model
                        self.model = None
                    if self.tokenizer is not None:
                        del self.tokenizer
                        self.tokenizer = None

                # 1st Pass - Clearing VRAM cache
                # - CUDA Devices: torch.cuda.synchronize() + torch.cuda.empty_cache() + torch.cuda.ipc_collect()
                # - Non-CUDA Devices (MPS/XPU/NPU/MLU): empty_cache()
                model_management.soft_empty_cache(True)
                
                # Force Python + torch garbage collection (Clearing CPU RAM)
                gc.collect()

                # 2nd Pass - Double empty_cache() is a well-known best practice pattern to catch any lingering tensors after GC
                if torch.cuda.is_available():
                    # Skipping torch.cuda.synchronize() on the 2nd pass
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.reset_peak_memory_stats()    # Optional but keeps CUDA stats clean
                elif hasattr(torch, 'mps') and torch.mps.is_available():
                    torch.mps.empty_cache()
                elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                    torch.xpu.empty_cache()
                elif hasattr(torch, 'npu') and torch.npu.is_available():
                    torch.npu.empty_cache()
                elif hasattr(torch, 'mlu') and torch.mlu.is_available():
                    torch.mlu.empty_cache()
                
                print(" - VRAM/RAM cleanup complete.")
            except Exception as cleanup_err:
                print(f" - Error during memory cleanup:\n{cleanup_err}")