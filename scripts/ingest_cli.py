import argparse
import sys
from pathlib import Path

from mergerag.ingestion import ingest_document
from mergerag.adapters.embedder import SentenceTransformerEmbedder
from mergerag.adapters.retriever import ChromaRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a document into ChromaDB with embeddings."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to a single file to ingest",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--persist-path",
        default=None,
        help="Path for ChromaDB persistence; if omitted, uses ephemeral client",
    )

    args = parser.parse_args()

    try:
        embedder = SentenceTransformerEmbedder()
        retriever = ChromaRetriever(
            collection_name=args.collection,
            persist_path=args.persist_path,
        )

        chunk_count = ingest_document(
            path=Path(args.path),
            embedder=embedder,
            retriever=retriever,
        )

        print(
            f"Ingested {chunk_count} chunks from '{args.path}' into collection '{args.collection}'"
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
