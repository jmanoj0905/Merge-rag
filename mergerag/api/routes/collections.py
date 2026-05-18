from chromadb.errors import NotFoundError as ChromaNotFoundError
from fastapi import APIRouter, Depends, HTTPException, Request

from mergerag.api.config import Settings
from mergerag.api.deps import clear_hybrid_cache, get_chroma_client, get_settings_dep
from mergerag.api.schemas import CollectionInfo

router = APIRouter()


@router.get("/collections", response_model=list[CollectionInfo])
def list_collections(request: Request) -> list[CollectionInfo]:
    client = get_chroma_client(request)
    collections = client.list_collections()
    return [
        CollectionInfo(name=col.name, chunk_count=col.count())
        for col in collections
    ]


@router.delete("/collections/{name}")
def delete_collection(
    name: str,
    request: Request,
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    client = get_chroma_client(request)
    try:
        client.delete_collection(name)
        clear_hybrid_cache(name, settings.chroma_persist_path)
    except (ValueError, ChromaNotFoundError):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{name}' not found",
        )
    return {"deleted": name}
