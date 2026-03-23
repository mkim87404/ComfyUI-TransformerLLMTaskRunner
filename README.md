# ComfyUI-TransformerLLMTaskRunner
---
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Node-blue)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, dependency-safe, and memory-conscious ComfyUI custom node that lets you run basic **transformer LLMs** directly inside ComfyUI workflows.

[![ComfyUI Node Screenshot](https://private-user-images.githubusercontent.com/187950231/567715119-2e5bbbeb-3f11-4dec-a599-0e1518423b0f.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzQyNjk0NDcsIm5iZiI6MTc3NDI2OTE0NywicGF0aCI6Ii8xODc5NTAyMzEvNTY3NzE1MTE5LTJlNWJiYmViLTNmMTEtNGRlYy1hNTk5LTBlMTUxODQyM2IwZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMzIzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDMyM1QxMjMyMjdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kMmRmNzllMDE3MzQ4OTZiNzkyMjI4YTA4NmExZGM5ZjA0ZTE5Njk4ODU3ODY3ODM1YTU4YTQ4MmFlYThhNDdlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.d_WyPG1J-X4uJutf6dEKyBeFmY1-R4uDgpmLD2PFo_8)](https://private-user-images.githubusercontent.com/187950231/567715119-2e5bbbeb-3f11-4dec-a599-0e1518423b0f.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzQyNjk0NDcsIm5iZiI6MTc3NDI2OTE0NywicGF0aCI6Ii8xODc5NTAyMzEvNTY3NzE1MTE5LTJlNWJiYmViLTNmMTEtNGRlYy1hNTk5LTBlMTUxODQyM2IwZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMzIzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDMyM1QxMjMyMjdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kMmRmNzllMDE3MzQ4OTZiNzkyMjI4YTA4NmExZGM5ZjA0ZTE5Njk4ODU3ODY3ODM1YTU4YTQ4MmFlYThhNDdlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.d_WyPG1J-X4uJutf6dEKyBeFmY1-R4uDgpmLD2PFo_8)

**Useful for:** prompt enhancement, Booru image tag pruning, reasoning chains, JSON output, calculations, generic logic/thinking tasks offloading, etc.

## ✨ Features
---
- Uses `transformers` + `torch` versions already installed on your ComfyUI instance (zero dependency conflicts or version pinning / downgrade), for maximum compatibility with evolving ComfyUI environments and other custom nodes.
- Secure & Aggressive VRAM/RAM cleanup after every run (or optional “keep model loaded” mode for faster reruns)
- Dynamic prompt formatting with up to 6 string variable injections + built-in chat-template support for modern instruct models
- Automatic Hugging Face model downloads + Custom model support
- Smart fallbacks (OOM → CPU Fallback, auto attention fallback → sdpa/eager)
- Full control: dtype, device_map, attn_implementation, max tokens, trust_remote_code

## 🛠️ Installation
---
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mkim87404/ComfyUI-TransformerLLMTaskRunner.git
cd ComfyUI-TransformerLLMTaskRunner
pip install -r requirements.txt    # these are all harmless libraries that won't cause any dependency headache with torch/cuda/transformers, etc.
```
Then restart ComfyUI → double click anywhere in the workflow and search for **"Transformer LLM Task Runner"**

## 📥 Model Downloads
---
**Auto download (recommended):** Just select a model from the node's model drop down menu or copy-paste a custom Hugging Face model ID into "custom_model_hf_id" and run. The node will auto-download the model on first run.

**Manual download:**
```bash
# Run this from your ComfyUI Installation Root, and the model will appear on the node's model selection drop down menu on your next ComfyUI launch.
hf download Qwen/Qwen2.5-7B-Instruct --local-dir "ComfyUI/models/LLM/Qwen2.5-7B-Instruct"
```

## 🚀 Usage
---
Add the node into your ComfyUI workflow and describe the task you want the LLM to run in any language supported by the LLM (with optional dynamic string inputs) and run. The LLM's output string can then be used by any downstream node in the workflow (e.g. prompts, string concatenate, boolean switch routing, Save Text etc.)

**Example task (image2image edit prompt refinement):**
```text
From the given description of an image, preserve as much of its content as is, while removing only parts that directly contradict the new instruction.

description:
{arg0}

new instruction:
{arg1}
```

{arg0} - This could be a comma separated Booru/Danbooru-style tags description of an initial image, possibly auto-extracted from Vision Models like Florence2, etc. in an earlier node.

{arg1} - Any natural language description for the image change you are trying to prompt, to be used for removing contradicting tags in {arg0}.

This task will output a pruned description of the initial image containing only non-contradicting elements that you might want to preserve in the new image, and you could concatenate this output with {arg1} (your intended image changes) to form your final new image prompt.

## 📊 Node Parameters Table
---
| Parameter | Type | Default | Description |
|---|---|---|---|
| **task** | String (multiline) | (Template) | Your task prompt with optional {argN} string placeholders |
| **model** | Dropdown | Qwen/Qwen2.5-7B-Instruct | Preset + custom models found in ComfyUI/models/LLM |
| **custom_model_hf_id** | String | (Optional) | Override the model dropdown selection with any HF model id e.g. Qwen/Qwen2.5-7B-Instruct |
| **dtype** | Dropdown | auto | Data type for model weights. 'auto' respects model config. |
| **attn_implementation** | Dropdown | auto | Attention backend. 'auto' respects model config, 'eager' is the safest but slowest fallback, 'sdpa' is faster and requires no extra pip install (shipped with PyTorch). |
| **device_map** | Dropdown | auto | Defines how to load/offload model layers. 'auto' for smart GPU+CPU offload, 'cpu' for low VRAM setup |
| **max_new_tokens** | Int | 1024 | Maximum tokens the LLM may generate. Higher value means longer output cap and longer maximum wait time for output. Bounded to 1–4096 for now. |
| **trust_remote_code** | Boolean | False | Set True only for models that explicitly require this config |
| **keep_model_loaded** | Boolean | False | Set True for faster reruns (keeps model loaded + still clears cache), and False for full model unload + max VRAM recovery after each node execution |
| **argN** | String | (Optional) | Optional dynamic string input to replace {argN} in your task prompt |

## 🤖 Supported Models
---
This node is designed for standard decoder-only transformer architectures. As long as a model meets the following criteria, it should work out-of-the-box:
* **Model Format**: Must be in the Safetensors or PyTorch (.bin) format (GGUF, EXL2, or AWQ specialized formats are not supported by the current version of this node).
* **Folder Structure**: The model folder must contain a standard config.json, tokenizer_config.json, and the weight files.
* **Loader Class & Pipeline**: Any model that supports loading via the Hugging Face `AutoModelForCausalLM` and `AutoTokenizer` classes or the `transformers.pipeline` wrapper for "text-generation" are fully supported by this node.
* **Architecture**: Optimized for popular standard architectures including Qwen (v2/2.5), Llama (v3.1/3.2), Mistral, Gemma (v2), and SmolLM.
* **Tips for finding compatible models**:
When browsing Hugging Face, look for models with the "Transformers" and "Safetensors" tags.
Avoid models labeled strictly as "GGUF" (used for llama.cpp) or "GPTQ" (unless you have the specific quantization libraries installed). Check the config.json file in the model's repository and if the model_type is a recognized transformer (like llama, qwen2, or gemma2), it is highly likely to work.
* **Remote Code**: If a model requires custom logic (like Florence-2 or any newer experimental architectures), ensure to set trust_remote_code to True in the node settings, and follow the exact config instructions as per the hugging face model page.
    * If you find you need to make complex tweaks to this custom node to accommodate more advanced model architectures, I would generally recommend switching to their dedicated ComfyUI Custom Nodes such as [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2), [ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL), etc. but the tradeoff with these advanced nodes is that they might start introducing dependency conflicts to your ComfyUI environment as you install more advanced attention backends and/or model quantization or compression libraries, and some of these custom nodes may not be implementing memory clean up as thoroughly and gracefully as you might expect.

## 🧠 Memory Management Designs
---
* When keep_model_loaded = False, model and tokenizer are explicitly unbound and garbage collected after every execution of the node. On next run, the node will correctly re-load the model if not found.
    * Avoided model.to("cpu") to unload the model from gpu as it conflicts with accelerate hooks used by device_map="auto" and can cause memory leaks / warnings. del model unbinding + py garbage collection is the safest solution for sharded, GPU, CPU, or sequential loads.
    * Avoided torch.cuda.synchronize() as it will block execution until all queued CUDA work is finished, adding unnecessary latency and compromises per-request throughput which is not strictly necessary for freeing memory and moving on.
* Secure & explicit cache + tensor clearing with torch.cuda.empty_cache() + torch.cuda.reset_peak_memory_stats() + gc.collect() after each run regardless of keep_model_loaded config.
* Clean up logic is always executed, surviving OOM, crashes, and partial loads
* Best-effort memory clearance: Despite the above implementations, PyTorch’s caching allocator can still fragment over many runs, but this is generally harmless and fully reversed on ComfyUI Session termination. The only way to reliably reset everything on demand is simply restarting the ComfyUI Python process (one click in Manager or terminal), which is not a limitation of the node – it’s how PyTorch is designed for optimization as [documented](https://docs.pytorch.org/docs/stable/notes/cuda.html).

## 📜 License
---
[**MIT License**](https://github.com/mkim87404/ComfyUI-TransformerLLMTaskRunner/blob/main/LICENSE) – feel free to use in any personal or commercial project, fork, or open issues/PRs – contributions and feedback all welcome!