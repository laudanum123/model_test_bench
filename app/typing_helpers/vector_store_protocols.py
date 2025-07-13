# typing_helpers/vector_store_protocols.py
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SupportsUpdateEmbeddings(Protocol):
    async def update_embeddings(self, embeddings: list[list[float]]) -> None: ...

@runtime_checkable
class DatasetItem(Protocol):
    def keys(self) -> Any: ...
