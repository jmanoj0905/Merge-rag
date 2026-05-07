import chromadb
from chromadb.errors import NotFoundError as ChromaNotFoundError
from fastapi import APIRouter, Depends, HTTPException

from mergerag.api.config import Settings
from mergerag.api.deps import get_settings_dep
from mergerag.api.schemas import CollectionInfo

router = APIRouter()


def _make_client(settings: Settings) -> chromadb.Client:
    if settings.chroma_persist_path:
        return chromadb.PersistentClient(path=settings.chroma_persist_path)
    return chromadb.EphemeralClient()


@router.get("/collections", response_model=list[CollectionInfo])
def list_collections(
    settings: Settings = Depends(get_settings_dep),
) -> list[CollectionInfo]:
    client = _make_client(settings)
    collections = client.list_collections()
    return [
        CollectionInfo(name=col.name, chunk_count=col.count())
        for col in collections
    ]


@router.delete("/collections/{name}")
def delete_collection(
    name: str,
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    client = _make_client(settings)
    try:
        client.delete_collection(name)
    except (ValueError, ChromaNotFoundError):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{name}' not found",
        )
    return {"deleted": name}
