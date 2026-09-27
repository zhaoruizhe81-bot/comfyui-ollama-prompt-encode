"""
@author: Michael Standen
@title: Ollama Prompt Encode
@nickname: Ollama Prompt Encode
@description: Use LLMs (Ollama or any OpenAI-compatible API) to generate prompts and perform CLIP text encoding
"""

from .OllamaPromptGenerator import OllamaPromptGenerator

class OllamaCLIPTextEncode(OllamaPromptGenerator):

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()["required"].copy()
        return {
            "required": {
                "clip": ("CLIP",),
                **inputs,
            }
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )
    RETURN_NAMES = (
        "conditioning",
        "prompt",
    )
    FUNCTION = "get_encoded"

    CATEGORY = "Ollama"

    def get_encoded(self, clip, llm_provider, base_url, api_key, model, seed, prepend_tags, system_prompt, description, comma_separated_response, timeout):
        """Gets and encodes the prompt using CLIP."""
        combined_prompt = self.get_prompt(llm_provider, base_url, api_key, model, seed, prepend_tags, system_prompt, description, comma_separated_response, timeout)[0]

        tokens = clip.tokenize(combined_prompt)
        # return_dict=True keeps every extra key from encode_token_weights
        # (attention_mask, hook keys, ...). Dropping them corrupts the
        # conditioning on mask-dependent CLIPs (qwen_image etc.) and produces
        # pure noise. See issue #11.
        cond_dict = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = cond_dict.pop("cond")
        return ([[cond, cond_dict]], combined_prompt)
