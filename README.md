# ComfyUI Ollama Prompt Encode

A prompt generator and CLIP encoder using AI provided by [Ollama](https://ollama.com).

![Example Usage](./docs/usage_1.png)

## Prerequisites

Install [Ollama](https://ollama.com) and have the service running.

This node has been tested with ollama version `0.4.6`.

## Installation

Choose one of the following methods to install the node:

### via ComfyUI Manager

If you have the [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed, you can install the node from the `Install Custom Nodes`.

Search for `Ollama Prompt Encode` and click `Install`.

![ComfyUI Manager](./docs/comfyui_manager.png)

### via Comfy CLI

If you have the [Comfy CLI](https://github.com/Comfy-Org/comfy-cli) installed, you can install the node from the command line.

```sh
comfy node registry-install comfyui-ollama-prompt-encode
```

The registry instance can be found on (registry.comfy.org)[https://registry.comfy.org/publishers/michaelstanden/nodes/comfyui-ollama-prompt-encode].

### via Git

Clone this repository into your `<comfyui>/custom_nodes` directory.

```sh
cd <comfyui>/custom_nodes
git clone https://github.com/ScreamingHawk/comfyui-ollama-prompt-encode
```

## Usage

![Example Usage](./docs/usage_1.png)

The `Ollama CLIP Prompt Encode` node is designed to replace the default `CLIP Text Encode (Prompt)` node. It generates a prompt using the Ollama AI model and then encodes the prompt with CLIP.

The node will output the generated prompt as a `string`. This can be viewed with [rgthree's `Display Any` node](https://github.com/rgthree/rgthree-comfy?tab=readme-ov-file#display-any).

An [example workflow](./docs/ollama_basic_workflow.json) is available in the `docs` folder.

### Ollama URL

The URL to the Ollama service. The default is `http://localhost:11434`.

### Ollama Model

This is the model that is used to generate your prompt.

Some models that work well with this prompt generator are:

- `orca-mini`
- `mistral`
- `tinyllama`

The node will automatically download the model if it is not already present on your system.

Smaller models are recommended for faster generation times.

### Seed

The seed that will be used to generate the prompt. This is useful for generating the same prompt multiple times or ensuring a different prompt is generated each time.

### Prepend Tags

A string that will be prepended to the generated prompt.

This is useful for models like `pony` that work best with extra tags like `score_9, score_8_up`.

### Text

The text that will be used by the AI model to generate the prompt.

### Comma Separated Response

If checked, the node will generate a prompt with a high number of tags separated by commas. e.g. `young girl, photorealistic, blue hair`. This is better for models that work better with more tags like `pony`.

If unchecked, the node will generate a prompt with a more descriptive prompt. e.g. `A photorealistic image of a young girl with blue hair`. This is better for models that work better with more descriptive prompts like `Flux`.

## Testing

Run the tests with:

```sh
python -m unittest
```

## Credits

[Michael Standen](https://michael.standen.link)

This software is provided under the [MIT License](https://tldrlegal.com/license/mit-license) so it's free to use so long as you give me credit.

## Fork: LLM providers, system prompt separation and fixes

This fork (`zhaoruizhe81-bot`) extends version 2.2.0 with the following changes (v2.3.0):

### OpenAI-compatible providers

`llm_provider` switches between:

- `ollama` — the native Ollama API (default, `base_url` like `http://localhost:11434`)
- `openai` — any OpenAI-compatible `/v1/chat/completions` endpoint (llama.cpp server, vLLM, LM Studio, one-api/new-api relays, ...). `base_url` accepts `http://host:8000` or `http://host:8000/v1`; `api_key` is sent as a Bearer token (`EMPTY` works for local servers).

Ollama itself exposes an OpenAI-compatible endpoint, so `openai` + `http://localhost:11434/v1` also works.

### System prompt / description split

The old single `text` input is replaced by:

- `system_prompt` — instructions for the LLM. Leave it **empty** to keep the built-in behaviour (comma/descriptive system message + few-shot samples from the bundled CSVs).
- `description` — the scene description to expand/translate.

`ollama_model` was renamed to `model` and a `timeout` input (default 300s) replaces the old hardcoded 60s limit. **This is a breaking change**: re-add the node in existing workflows.

### Fixes

- The CLIP encode path now keeps every extra conditioning key (`attention_mask`, hook keys) returned by `encode_token_weights`. The old code dropped them, which corrupted conditioning on mask-dependent CLIP types (`qwen_image`) and produced pure noise. See [upstream issue #11](https://github.com/ScreamingHawk/comfyui-ollama-prompt-encode/issues/11) and [PR #12](https://github.com/ScreamingHawk/comfyui-ollama-prompt-encode/pull/12).
- `<think>...</think>` blocks from reasoning models (qwen3 etc.) are stripped before CLIP encoding.
- The Ollama model pull is now on-demand (only when the model is missing) and best-effort, instead of a mandatory registry call on every generation.
- `prepend_tags` no longer produces a leading `", "` when the generated prompt or the tags are empty.

### v2.4.x — Ollama robustness options (optional inputs, old workflows keep working)

- `temperature` (v2.3.1) — sampling temperature, default 0.8. `seed != 0` still forces 0 for reproducibility. On `openai`, a temperature-locked reasoning model (e.g. `kimi-k2.7-code-*`, which only allows temperature 1) triggers an automatic retry without the field instead of an HTTP 400; API error bodies are now surfaced in the exception message.
- `num_predict` (default 1024, 0 = unlimited) — hard cap on generated tokens for the **Ollama path only**. Small local models sometimes ignore tag-count instructions or fall into repetition loops (observed: 915 tokens for a "15–40 tags" request); the cap bounds the worst case so a run can never stall until the timeout. Not sent on `openai` because reasoning models have provider-specific `max_tokens` semantics.
- `repeat_penalty` (default 1.1, Ollama path only) — raises above 1.1 if a local model falls into tag-repetition loops.
- `think` (default off, Ollama path only) — allows reasoning chains (qwen3 etc.). Off by default: chain-of-thought is useless for tag generation, slow, and burns through `num_predict` before any content is produced (models that ignore the flag still get inline `<think>` stripped).
- `keep_alive` (default `1m`, Ollama path only) — how long the model stays in VRAM after a run (`1m`, `10m`, `-1` = forever; empty = server default). On an 8GB card, a resident 3–4GB LLM next to ComfyUI's diffusion models causes WDDM paging that slows generation ~10x; unloading after each run frees VRAM for sampling at the cost of a ~10s reload.

### v2.5.0 — Iterate on a known-good prompt (optional inputs)

- `last_prompt` — paste the prompt of a good roll here (copy it from a ShowText node).
- `optimize` — merge mode: the LLM receives an optimizer persona (your `system_prompt` if set, otherwise a built-in one) plus 【原提示词】(`last_prompt`) and 【描述/优化点】(`description`), and returns the prompt with your refinements merged in — a good roll becomes the base for iteration instead of a from-scratch regeneration.
- `lock` — skip the LLM entirely and output `last_prompt` verbatim: free, instant, prompt never drifts while you reroll the KSampler seed. Takes precedence over `optimize` when both are on.
- `prepend_tags` are now merged only when missing from the output, so prompts that already carry the quality tags are not doubled up.
