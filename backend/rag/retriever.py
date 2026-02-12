from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExampleRetriever:
    def __init__(
        self,
        data_dir: Path | str = "data/jupyter-agent/processed",
        chroma_dir: Path | str = "data/chroma",
        collection_name: str = "jupyter_agent_examples",
        prefer_chroma: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.prefer_chroma = prefer_chroma

        self._chroma_client = None
        self._collection = None
        self._chroma_enabled = False
        if prefer_chroma:
            self._try_init_chroma()

    def _try_init_chroma(self) -> None:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        except Exception:  # noqa: BLE001
            return

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self._chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_fn,
            metadata={"description": "Educational python examples for RAG"},
        )
        self._chroma_enabled = True

    def _iter_jsonl_records(self) -> list[dict[str, Any]]:
        if not self.data_dir.exists():
            return []
        records: list[dict[str, Any]] = []
        for path in sorted(self.data_dir.glob("sample_*.jsonl")):
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not record.get("id"):
                        record["id"] = f"{path.stem}:{line_no}"
                    records.append(record)
        return records

    def _to_document(self, record: dict[str, Any]) -> str:
        rag_text = str(record.get("rag_text", "")).strip()
        if rag_text:
            return rag_text
        return (
            f"question: {record.get('question', '')}\n"
            f"answer: {record.get('answer', '')}\n"
            f"packages: {', '.join(record.get('packages_used', []))}\n"
            f"edu_score: {record.get('edu_score', '')}"
        )

    def index_processed_files(self) -> int:
        if not self._chroma_enabled or not self._collection:
            return 0
        records = self._iter_jsonl_records()
        if not records:
            return 0

        ids = [str(record["id"]) for record in records]
        docs = [self._to_document(record) for record in records]
        metadatas = []
        for record in records:
            metadatas.append(
                {
                    "primary_package": str(record.get("primary_package", "unknown")),
                    "edu_score": int(record.get("edu_score", 0)),
                    "kaggle_dataset_name": str(record.get("kaggle_dataset_name", "")),
                }
            )

        # Upsert keeps index fresh if same ids already exist.
        self._collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
        return len(ids)

    def _retrieve_lexical(self, query: str, limit: int) -> list[dict[str, Any]]:
        query_terms = {t.lower() for t in query.split() if t.strip()}
        if not query_terms:
            return []
        scored: list[tuple[int, dict[str, Any]]] = []
        for record in self._iter_jsonl_records():
            text = f"{record.get('question', '')} {record.get('answer', '')}".lower()
            score = sum(term in text for term in query_terms)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def _retrieve_chroma(self, query: str, limit: int) -> list[dict[str, Any]]:
        if not self._chroma_enabled or not self._collection:
            return []
        result = self._collection.query(query_texts=[query], n_results=limit)
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        out = []
        for idx, doc, meta in zip(ids, documents, metadatas, strict=False):
            out.append(
                {
                    "id": idx,
                    "rag_text": doc,
                    "question": str(doc).split("\n")[0].replace("question: ", "", 1),
                    "answer": "",
                    "metadata": meta or {},
                }
            )
        return out

    def retrieve(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        if self._chroma_enabled:
            chroma_items = self._retrieve_chroma(query=query, limit=limit)
            if chroma_items:
                return chroma_items
        return self._retrieve_lexical(query=query, limit=limit)
