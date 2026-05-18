"""
Tests for the local judge.

All tests run without loading Gemma — the model's generate() is mocked.
"""
import pytest
from unittest.mock import MagicMock, patch


def test_substring_refusal_fast_path():
    """Substring check should return False without calling Gemma."""
    from judges.local_judge import LocalJudge, _substring_is_refusal

    # These should all be detected as refusals
    assert _substring_is_refusal("I don't know.") is True
    assert _substring_is_refusal("I do not know.") is True
    assert _substring_is_refusal("The context does not provide enough information.") is True
    assert _substring_is_refusal("I cannot answer this question.") is True

    # These should NOT be detected as refusals
    assert _substring_is_refusal("The Eiffel Tower is in Paris.") is False
    assert _substring_is_refusal("Sam Altman is the CEO of OpenAI.") is False


def test_is_answered_refusal_shortcircuit():
    """is_answered() should return False for IDK phrases without loading model."""
    from judges.local_judge import LocalJudge

    judge = object.__new__(LocalJudge)
    # No model loaded — if it reaches the model it would crash
    assert judge.is_answered("Who is the CEO?", "I don't know.") is False
    assert judge.is_answered("What is Paris?", "The context does not provide enough information.") is False


def _make_judge_with_mock_llm(decode_output: str):
    """Build a LocalJudge with a mocked tokenizer and model."""
    from judges.local_judge import LocalJudge

    judge = object.__new__(LocalJudge)
    judge.model_name = "mock"
    judge.device = "cpu"
    judge.max_new_tokens = 4

    # inputs must have a .to() method (BatchEncoding-like), so use MagicMock
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda self, k: MagicMock(shape=[1, 10])

    mock_tok = MagicMock()
    mock_tok.return_value = mock_inputs
    mock_tok.decode.return_value = decode_output
    mock_tok.eos_token_id = 1

    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock()

    judge._tok = mock_tok
    judge._model = mock_model
    return judge


def test_is_answered_yes(base_cfg):
    """Mock Gemma output 'YES' → is_answered returns True."""
    judge = _make_judge_with_mock_llm("YES")
    # Non-refusal response → goes to LLM path
    result = judge.is_answered("Who invented radium?", "Marie Curie discovered radium.")
    assert result is True


def test_is_answered_no(base_cfg):
    """Mock Gemma output 'NO' → is_answered returns False."""
    judge = _make_judge_with_mock_llm("NO")
    result = judge.is_answered("Who invented radium?", "The provided context says nothing about radium.")
    assert result is False


def test_judge_prompt_has_examples():
    from judges.prompts import build_judge_prompt
    prompt = build_judge_prompt("test query", "test response")
    assert "YES" in prompt
    assert "NO" in prompt
    assert "test query" in prompt
    assert "test response" in prompt
