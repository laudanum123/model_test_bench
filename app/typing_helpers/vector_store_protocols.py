# typing_helpers/vector_store_protocols.py
from typing import Protocol, runtime_checkable, List, Any

@runtime_checkable
class SupportsUpdateEmbeddings(Protocol):
    async def update_embeddings(self, embeddings: List[List[float]]) -> None: ...

@runtime_checkable
class DatasetItem(Protocol):
    def keys(self) -> Any: ...
