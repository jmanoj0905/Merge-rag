from fastapi import APIRouter, HTTPException, Request

from mergerag.api.schemas import RunDetail, RunSummary
from mergerag.api.serializers import run_detail_out

router = APIRouter()


@router.get("/runs", response_model=list[RunSummary])
def list_runs(
    request: Request,
    limit: int = 50,
    offset: int = 0,
) -> list[RunSummary]:
    run_store = request.app.state.run_store
    traces = run_store.list_runs(limit=limit, offset=offset)
    return [
        RunSummary(
            run_id=t.run_id,
            created_at=t.created_at.isoformat(),
            query=t.query,
            strategy=t.strategy,
            collection_name=t.collection_name,
            token_count=t.token_count,
            latency_ms=t.latency_ms,
        )
        for t in traces
    ]


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str, request: Request) -> RunDetail:
    run_store = request.app.state.run_store
    trace = run_store.get(run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run_detail_out(trace)
