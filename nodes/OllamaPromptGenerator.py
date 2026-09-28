"""
@author: Michael Standen
@title: Ollama Prompt Encode
@nickname: Ollama Prompt Encode
@description: Use LLMs (Ollama or any OpenAI-compatible API) to generate prompts
"""

import os
import csv
import json
import re
import time
import urllib.error
import urllib.request

from ollama import Client, Options
from .timeout import timeout as with_timeout

# Reasoning models (e.g. qwen3) may inline their chain of thought; it must
# never reach the CLIP encoder.
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

SYSTEM_MESSAGES = {
    "descriptive": "You describe pictures. I will give you a brief description of the picture. You will describe the picture in intricate detail. Describe clothing, pose, expression, setting, lighting, and any other details you can think of. Use long descriptive sentences.",
    "comma": "You describe pictures. I will give you a brief description of the picture. You reply with comma separated keywords that describe the picture. Describe clothing, pose, expression, setting, and any other details you can think of. Use comma separated keywords. Do not use sentences. Use brevity.",
}

class OllamaPromptGenerator:
    # Defaults
    DEFAULT_TIMEOUT = 300
    DEFAULT_URL = "http://localhost:11434"
    DEFAULT_MODEL = "huihui_ai/qwen3-abliterated:4b"
    # Unload the model from VRAM shortly after each call: an 8GB card cannot
    # hold this model alongside ComfyUI's diffusion models, and WDDM paging
    # then slows generation ~10x. Reload costs ~10s per run.
    OLLAMA_KEEP_ALIVE = "1m"
    DEFAULT_NUM_PREDICT = 1024
    # Final prompt produced by each node instance (keyed by node unique_id).
    # Lets lock reuse it without manual copy-paste; lives in RAM, so
    # a ComfyUI restart clears it.
    _last_prompts = {}

    def load_sample_data(self, comma_separated_response: bool = True):
        fname = "sample_data_comma.csv" if comma_separated_response else "sample_data_descriptive.csv"
        fname = os.path.join(os.path.dirname(__file__), fname)
        sample_data = []
        with open(fname, "r") as fin:
            reader = csv.DictReader(fin)
            sample_data = [row for row in reader]
        return sample_data

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_provider": (["ollama", "openai"], {"default": "ollama"}),
                "base_url": ("STRING", {"default": cls.DEFAULT_URL}),
                "api_key": ("STRING", {"default": "EMPTY"}),
                "model": ("STRING", {"default": cls.DEFAULT_MODEL}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prepend_tags": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "system_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "description": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "comma_separated_response": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {"default": cls.DEFAULT_TIMEOUT, "min": 10, "max": 3600}),
            },
            # Optional so workflows saved before v2.3.1 keep loading.
            "optional": {
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                # Ollama path only: cap generated tokens (0 = unlimited) so a
                # looping small model cannot stall a run until the timeout.
                "num_predict": ("INT", {"default": cls.DEFAULT_NUM_PREDICT, "min": 0, "max": 8192,
                                        "tooltip": "Ollama only. Max generated tokens, 0 = unlimited"}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05,
                                             "tooltip": "Ollama only. Raise above 1.1 to break tag-repetition loops"}),
                "think": ("BOOLEAN", {"default": False,
                                      "tooltip": "Ollama only. Allow reasoning chains (qwen3 etc.) - slower, unnecessary for tags"}),
                "keep_alive": ("STRING", {"default": cls.OLLAMA_KEEP_ALIVE,
                                          "tooltip": "Ollama only. How long the model stays in VRAM after a run (1m, 10m, -1). Empty = server default"}),
                # Lock: reuse the node's own previous output verbatim, without
                # any LLM call (per-instance memory, see _last_prompts).
                "lock": ("BOOLEAN", {"default": False,
                                     "tooltip": "锁定模式：跳过 LLM，直接使用本节点上次生成的提示词（ComfyUI 重启后需先跑一次普通生成）"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "prompt",
    )
    FUNCTION = "get_prompt"

    CATEGORY = "Ollama"

    def sanitize_prompt(self, prompt):
        """Sanitize the prompt for use in clip encoding."""
        prompt = THINK_BLOCK_RE.sub("", prompt)
        prompt = prompt.replace(".", ",").replace("\n", ", ")
        prompt = re.sub(r"\s*,\s*,+", ", ", prompt)
        return prompt.strip(" ,\t\n")

    def _build_messages(self, system_prompt, description, comma_separated_response):
        system_prompt = (system_prompt or "").strip()
        if system_prompt:
            # Custom system prompt: the user has full control, no few-shot data.
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": description},
            ]
        # Built-in behaviour: system message plus few-shot sample data.
        system_message = SYSTEM_MESSAGES["comma" if comma_separated_response else "descriptive"]
        messages = [
            {"role": "system", "content": system_message},
        ]
        for row in self.load_sample_data(comma_separated_response):
            messages.append({"role": "user", "content": "Write a prompt for: " + row["text"]})
            messages.append({"role": "assistant", "content": row["prompt"]})
        messages.append({"role": "user", "content": "Write a prompt for: " + description})
        return messages

    def _chat_ollama(self, base_url, model, messages, seed, temperature, num_predict, repeat_penalty, think, keep_alive):
        client = Client(host=base_url)
        # Pull the model only if it is missing, and never let registry hiccups
        # kill the node: if the model is truly absent, the chat call below
        # raises a clear error.
        try:
            listed = client.list()
            names = set()
            for entry in getattr(listed, "models", listed):
                names.add(getattr(entry, "model", None) or getattr(entry, "name", "") or "")
            if model not in names:
                client.pull(model)
        except Exception:
            pass
        opts = Options()
        if seed:
            opts["seed"] = seed
        opts["temperature"] = 0.0 if seed else temperature
        if num_predict and num_predict > 0:
            opts["num_predict"] = int(num_predict)
        if repeat_penalty and repeat_penalty > 1.0:
            opts["repeat_penalty"] = float(repeat_penalty)
        response = client.chat(
            model=model,
            messages=messages,
            think=bool(think),
            options=opts,
            keep_alive=keep_alive or None,
            stream=False,
        )
        return response["message"]["content"]

    def _chat_openai(self, base_url, api_key, model, messages, seed, timeout_seconds, temperature):
        base = base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            endpoint = base
        elif base.endswith("/v1"):
            endpoint = base + "/chat/completions"
        else:
            endpoint = base + "/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": 0.0 if seed else temperature,
        }
        if seed:
            payload["seed"] = int(seed)
        # OpenAI-compat quirk handling: temperature-locked reasoning models
        # (kimi-k2.7-code-*, ...) reject any other value — retry once without
        # the field. Free/shared inference pools return transient 429s — back
        # off and retry those too. Anything else surfaces with its body.
        dropped_temperature = False
        rate_limit_retries = 0
        while True:
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + (api_key or "EMPTY"),
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                if not dropped_temperature and "temperature" in body and "temperature" in payload:
                    del payload["temperature"]
                    dropped_temperature = True
                    continue
                if e.code == 429 and rate_limit_retries < 2:
                    rate_limit_retries += 1
                    try:
                        wait = min(float(e.headers.get("Retry-After", 0)) or 15.0, 30.0)
                    except ValueError:
                        wait = 15.0
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    "OpenAI-compatible API error %s at %s: %s" % (e.code, endpoint, body[:500])
                ) from e

    def get_prompt(self, llm_provider, base_url, api_key, model, seed, prepend_tags, system_prompt, description, comma_separated_response, timeout, temperature=0.8, num_predict=DEFAULT_NUM_PREDICT, repeat_penalty=1.1, think=False, keep_alive=OLLAMA_KEEP_ALIVE, lock=False, unique_id=None):
        """Generates prompt using the configured LLM provider."""
        prepend = (prepend_tags or "").strip(" ,")
        memory_key = str(unique_id or "default")

        def merge_prepend(text):
            # Prepend only when the text does not already carry it, so a
            # prompt captured from this node is not doubled up.
            if not prepend:
                return text
            if prepend.lower() in text.lower():
                return text
            return prepend + ", " + text if text else prepend

        if lock:
            # Lock: reuse this node's previous output verbatim — no LLM call.
            base = self._last_prompts.get(memory_key, "").strip()
            if not base:
                raise RuntimeError("锁定模式需要基底提示词：先以普通模式跑一次生成（ComfyUI 重启会清空记忆）")
            combined = merge_prepend(base)
            self._last_prompts[memory_key] = combined
            return (combined,)

        messages = self._build_messages(system_prompt, description, comma_separated_response)

        use_seed = seed if seed != 0 else None

        @with_timeout(timeout)
        def call_llm():
            if llm_provider == "openai":
                # Cloud models do not loop like small local ones, and reasoning
                # models have provider-specific max_tokens semantics — the
                # Ollama-only knobs below do not apply there.
                return self._chat_openai(base_url, api_key, model, messages, use_seed, timeout, temperature)
            return self._chat_ollama(base_url, model, messages, use_seed, temperature, num_predict, repeat_penalty, think, keep_alive)

        prompt = call_llm()
        generated = self.sanitize_prompt(prompt)
        if prepend and generated:
            combined_prompt = prepend + ", " + generated
        else:
            combined_prompt = prepend or generated
        self._last_prompts[memory_key] = combined_prompt
        return (combined_prompt,)
