"""
@author: Michael Standen
@title: Ollama Prompt Encode
@nickname: Ollama Prompt Encode
@description: Use LLMs (Ollama or any OpenAI-compatible API) to generate prompts and perform CLIP text encoding
"""

from .OllamaPromptGenerator import OllamaPromptGenerator
from .OllamaClipTextEncode import OllamaCLIPTextEncode

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaPromptGenerator": "LLM Prompt Generator (Ollama/OpenAI)",
    "OllamaCLIPTextEncode": "LLM CLIP Prompt Encode (Ollama/OpenAI)",
}

NODE_CLASS_MAPPINGS = {
    "OllamaPromptGenerator": OllamaPromptGenerator,
    "OllamaCLIPTextEncode": OllamaCLIPTextEncode,
}
