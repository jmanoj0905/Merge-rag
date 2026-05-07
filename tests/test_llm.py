from unittest.mock import patch, MagicMock
from mergerag.adapters.llm import OllamaLLM


def test_complete_returns_string():
    mock_response = {"response": "test output"}
    with patch("mergerag.adapters.llm.ollama.generate", return_value=mock_response):
        llm = OllamaLLM(model="qwen2.5:3b")
        result = llm.complete("hello", max_tokens=64)
    assert result == "test output"


def test_complete_passes_model_name():
    mock_response = {"response": "x"}
    with patch("mergerag.adapters.llm.ollama.generate", return_value=mock_response) as mock_gen:
        llm = OllamaLLM(model="qwen2.5:3b")
        llm.complete("prompt", max_tokens=128)
    assert mock_gen.call_args.kwargs["model"] == "qwen2.5:3b"


def test_complete_passes_max_tokens():
    mock_response = {"response": "x"}
    with patch("mergerag.adapters.llm.ollama.generate", return_value=mock_response) as mock_gen:
        llm = OllamaLLM(model="qwen2.5:3b")
        llm.complete("prompt", max_tokens=256)
    options = mock_gen.call_args.kwargs.get("options", {})
    assert options.get("num_predict") == 256
