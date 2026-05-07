import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile

from mergerag.adapters.retriever import ChromaRetriever
from mergerag.api.config import Settings
from mergerag.api.deps import get_embedder, get_settings_dep
from mergerag.api.schemas import IngestResponse
from mergerag.ingestion import ingest_document

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: Request,
    file: UploadFile,
    collection_name: str = Form(...),
    doc_id: str | None = Form(default=None),
    settings: Settings = Depends(get_settings_dep),
) -> IngestResponse:
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if suffix not in {".txt", ".md"}:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {suffix}. Supported: .txt, .md",
        )

    resolved_doc_id = doc_id if doc_id else Path(file.filename).stem
    if not resolved_doc_id:
        raise HTTPException(
            status_code=422,
            detail="Could not determine doc_id: provide a doc_id or a named file",
        )

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            contents = await file.read()
            tmp.write(contents)

        retriever = ChromaRetriever(
            collection_name=collection_name,
            persist_path=settings.chroma_persist_path,
        )
        embedder = get_embedder(request)

        count = ingest_document(
            path=Path(tmp_path),
            embedder=embedder,
            retriever=retriever,
            doc_id=resolved_doc_id,
        )
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(
        doc_id=resolved_doc_id,
        chunk_count=count,
        collection_name=collection_name,
    )
