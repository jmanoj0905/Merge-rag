from __future__ import annotations
import sqlite3
from datetime import datetime

from mergerag.core.models import RunScore
from mergerag.core.ports import ScoreStorePort

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS scores (
    run_id      TEXT PRIMARY KEY,
    question_id TEXT NOT NULL,
    gold_answer TEXT NOT NULL,
    em          REAL NOT NULL,
    f1          REAL NOT NULL,
    scored_at   TEXT NOT NULL
)
"""


class SQLiteScoreStore(ScoreStorePort):
    def __init__(self, db_path: str):
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, score: RunScore) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO scores "
                "(run_id, question_id, gold_answer, em, f1, scored_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    score.run_id,
                    score.question_id,
                    score.gold_answer,
                    score.em,
                    score.f1,
                    score.scored_at.isoformat(),
                ),
            )

    def get(self, run_id: str) -> RunScore | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scores WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_score(row)

    def list_scores(self, limit: int = 100, offset: int = 0) -> list[RunScore]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scores ORDER BY scored_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_score(r) for r in rows]

    @staticmethod
    def _row_to_score(row: sqlite3.Row) -> RunScore:
        return RunScore(
            run_id=row["run_id"],
            question_id=row["question_id"],
            gold_answer=row["gold_answer"],
            em=row["em"],
            f1=row["f1"],
            scored_at=datetime.fromisoformat(row["scored_at"]),
        )
