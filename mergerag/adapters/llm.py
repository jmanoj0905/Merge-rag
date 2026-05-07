import ollama
from mergerag.core.ports import LLMPort


class OllamaLLM(LLMPort):
    def __init__(self, model: str = "qwen2.5:3b"):
        self._model = model

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        response = ollama.generate(
            model=self._model,
            prompt=prompt,
            options={"num_predict": max_tokens},
        )
        return response["response"]
