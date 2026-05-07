from __future__ import annotations
import os
import tempfile

from mergerag.core.models import RunScore
from mergerag.adapters.score_store import SQLiteScoreStore


def _make_store() -> SQLiteScoreStore:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return SQLiteScoreStore(db_path=path)


def _make_score(**kwargs) -> RunScore:
    defaults = dict(run_id="run-1", question_id="q-1", gold_answer="Paris", em=1.0, f1=1.0)
    defaults.update(kwargs)
    return RunScore(**defaults)


def test_save_and_get_round_trip():
    store = _make_store()
    store.save(_make_score())
    result = store.get("run-1")
    assert result is not None
    assert result.run_id == "run-1"
    assert result.em == 1.0
    assert result.f1 == 1.0


def test_get_missing_returns_none():
    store = _make_store()
    assert store.get("nonexistent") is None


def test_list_scores_returns_all_saved():
    store = _make_store()
    store.save(_make_score(run_id="r1"))
    store.save(_make_score(run_id="r2"))
    scores = store.list_scores()
    assert len(scores) == 2


def test_overwrite_on_same_run_id():
    store = _make_store()
    store.save(_make_score(run_id="r1", em=0.0, f1=0.5))
    store.save(_make_score(run_id="r1", em=1.0, f1=1.0))
    result = store.get("r1")
    assert result.em == 1.0


def test_gold_answer_preserved():
    store = _make_store()
    store.save(_make_score(gold_answer="Haarlem"))
    result = store.get("run-1")
    assert result.gold_answer == "Haarlem"


def test_question_id_preserved():
    store = _make_store()
    store.save(_make_score(question_id="abc123"))
    result = store.get("run-1")
    assert result.question_id == "abc123"


def test_scored_at_preserved():
    from datetime import datetime
    store = _make_store()
    score = _make_score()
    store.save(score)
    result = store.get("run-1")
    assert isinstance(result.scored_at, datetime)
