import os
import tempfile
from contextlib import suppress
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing"""
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    database_url = f"sqlite:///{db_path}"

    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    testingsessionlocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = testingsessionlocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    yield testingsessionlocal()

    # Cleanup
    app.dependency_overrides.clear()
    testingsessionlocal.close_all()
    engine.dispose()
    with suppress(PermissionError):
        os.remove(db_path)
    with suppress(OSError):
        os.rmdir(temp_dir)


@pytest.fixture
def client():
    """Create a test client with database dependency overridden"""
    return TestClient(app)


@pytest.fixture
def mock_datasets():
    """Mock datasets library"""
    with patch("app.api.corpus.load_dataset") as mock_load:
        yield mock_load


@pytest.fixture
def mock_aiofiles():
    """Mock aiofiles for file operations"""
    with patch("app.api.corpus.aiofiles") as mock_aiofiles:
        mock_aiofiles.open.return_value.__aenter__.return_value = AsyncMock()
        yield mock_aiofiles


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing"""
    from datasets import Dataset

    return Dataset.from_dict({
        "text": [
            "This is the first sample text.",
            "This is the second sample text.",
            "This is the third sample text with more content."
        ],
        "id": [1, 2, 3]
    })


@pytest.fixture
def sample_dataset_dict():
    """Sample dataset dict for testing"""
    from datasets import Dataset, DatasetDict

    train_dataset = Dataset.from_dict({
        "text": ["Train text 1", "Train text 2"],
        "id": [1, 2]
    })

    test_dataset = Dataset.from_dict({
        "text": ["Test text 1"],
        "id": [3]
    })

    return DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })


@pytest.fixture
def sample_iterable_dataset():
    """Sample iterable dataset for testing"""
    from datasets import IterableDataset

    def text_generator():
        for i in range(3):
            yield {"text": f"Text {i}", "id": i}

    return IterableDataset.from_generator(text_generator)


@pytest.fixture
def valid_corpus_data():
    """Valid corpus creation data"""
    return {
        "name": "test_corpus",
        "description": "Test corpus for unit testing",
        "source": "huggingface",
        "source_config": {
            "dataset_name": "test-dataset",
            "split": "train",
            "text_column": "text"
        }
    }


@pytest.fixture
def mock_os_makedirs():
    """Mock os.makedirs for directory creation"""
    with patch("app.api.corpus.os.makedirs") as mock_makedirs:
        yield mock_makedirs


@pytest.fixture
def mock_os_path_exists():
    """Mock os.path.exists for file existence checks"""
    with patch("app.api.corpus.os.path.exists") as mock_exists:
        yield mock_exists
