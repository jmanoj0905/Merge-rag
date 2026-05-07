from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone

from mergerag.core.models import (
    Chunk, MergedChunk, MergeOp, MergePlan, Citation, RunTrace,
)
from mergerag.core.ports import RunStorePort

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    query           TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    answer          TEXT NOT NULL,
    token_count     INTEGER NOT NULL,
    latency_ms      REAL NOT NULL,
    config          TEXT NOT NULL,
    retrieved_chunks TEXT NOT NULL,
    merged_chunks   TEXT NOT NULL,
    final_context   TEXT NOT NULL,
    merge_plan      TEXT,
    citations       TEXT NOT NULL
)
"""


def _ser_chunk(c: Chunk) -> dict:
    return {"id": c.id, "doc_id": c.doc_id, "text": c.text, "score": c.score, "rank": c.rank}


def _ser_merged(m: MergedChunk) -> dict:
    return {
        "id": m.id, "text": m.text, "score": m.score,
        "source_chunk_ids": m.source_chunk_ids, "merge_type": m.merge_type,
    }


def _ser_context_item(item: Chunk | MergedChunk) -> dict:
    if isinstance(item, MergedChunk):
        return {"type": "merged", **_ser_merged(item)}
    return {"type": "chunk", **_ser_chunk(item)}


def _ser_plan(plan: MergePlan | None) -> str | None:
    if plan is None:
        return None
    return json.dumps({
        "operations": [
            {"type": op.type, "primary_id": op.primary.id, "secondary_id": op.secondary.id}
            for op in plan.operations
        ]
    })


def _deser_chunk(d: dict) -> Chunk:
    return Chunk(id=d["id"], doc_id=d["doc_id"], text=d["text"], score=d["score"], rank=d["rank"])


def _deser_merged(d: dict) -> MergedChunk:
    return MergedChunk(
        id=d["id"], text=d["text"], score=d["score"],
        source_chunk_ids=d["source_chunk_ids"], merge_type=d["merge_type"],
    )


def _deser_context_item(d: dict) -> Chunk | MergedChunk:
    if d["type"] == "merged":
        return _deser_merged(d)
    return _deser_chunk(d)


def _deser_plan(raw: str | None) -> MergePlan | None:
    if raw is None:
        return None
    data = json.loads(raw)
    ops = [
        MergeOp(
            type=op["type"],
            primary=Chunk(id=op["primary_id"], doc_id="", text="", score=0.0, rank=0),
            secondary=Chunk(id=op["secondary_id"], doc_id="", text="", score=0.0, rank=0),
        )
        for op in data["operations"]
    ]
    return MergePlan(operations=ops)


def _row_to_trace(row: sqlite3.Row) -> RunTrace:
    return RunTrace(
        run_id=row["run_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        query=row["query"],
        strategy=row["strategy"],
        collection_name=row["collection_name"],
        answer=row["answer"],
        token_count=row["token_count"],
        latency_ms=row["latency_ms"],
        config=json.loads(row["config"]),
        retrieved_chunks=[_deser_chunk(c) for c in json.loads(row["retrieved_chunks"])],
        merged_chunks=[_deser_merged(m) for m in json.loads(row["merged_chunks"])],
        final_context=[_deser_context_item(x) for x in json.loads(row["final_context"])],
        merge_plan=_deser_plan(row["merge_plan"]),
        citations=[Citation(sentence=c["sentence"], chunk_ids=c["chunk_ids"]) for c in json.loads(row["citations"])],
    )


class SQLiteRunStore(RunStorePort):
    def __init__(self, db_path: str):
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, run: RunTrace) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, created_at, query, strategy, collection_name, answer,
                 token_count, latency_ms, config, retrieved_chunks, merged_chunks,
                 final_context, merge_plan, citations)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run.run_id,
                    run.created_at.isoformat(),
                    run.query,
                    run.strategy,
                    run.collection_name,
                    run.answer,
                    run.token_count,
                    run.latency_ms,
                    json.dumps(run.config),
                    json.dumps([_ser_chunk(c) for c in run.retrieved_chunks]),
                    json.dumps([_ser_merged(m) for m in run.merged_chunks]),
                    json.dumps([_ser_context_item(x) for x in run.final_context]),
                    _ser_plan(run.merge_plan),
                    json.dumps([{"sentence": c.sentence, "chunk_ids": c.chunk_ids} for c in run.citations]),
                ),
            )

    def get(self, run_id: str) -> RunTrace | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return _row_to_trace(row)

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[RunTrace]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_row_to_trace(r) for r in rows]
