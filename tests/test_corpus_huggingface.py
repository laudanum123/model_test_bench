from unittest.mock import AsyncMock, Mock

import pytest
from datasets import Dataset
from fastapi import HTTPException

from app.api.corpus import create_huggingface_corpus
from app.database import Corpus
from app.models.schemas import HuggingFaceCorpusRequest


def create_hf_request(name="test_corpus", dataset_name="test-dataset", split="train",
                     text_column="text", config_name=None, description=None):
    """Helper function to create HuggingFaceCorpusRequest objects for testing"""
    return HuggingFaceCorpusRequest(
        name=name,
        dataset_name=dataset_name,
        split=split,
        text_column=text_column,
        config_name=config_name,
        description=description
    )


class TestHuggingFaceCorpusCreation:
    """Test cases for HuggingFace corpus creation endpoint"""

    @pytest.mark.asyncio
    async def test_successful_corpus_creation_basic(
        self, temp_db, mock_datasets, mock_aiofiles, sample_dataset
    ):
        """Test successful corpus creation with basic parameters"""
        # Arrange
        mock_datasets.return_value = sample_dataset
        mock_aiofiles.open.return_value.__aenter__.return_value.write = AsyncMock()

        # Act
        request = create_hf_request(description="Test description")
        result = await create_huggingface_corpus(request, db=temp_db)

        # Assert
        assert result["name"] == "test_corpus"
        assert "successfully" in result["message"]
        assert result["id"] is not None

        # Check database record
        corpus = temp_db.query(Corpus).filter(Corpus.id == result["id"]).first()
        assert corpus.name == "test_corpus"
        assert corpus.source == "huggingface"
        assert corpus.source_config["dataset_name"] == "test-dataset"
        assert corpus.source_config["split"] == "train"
        assert corpus.source_config["text_column"] == "text"
        assert corpus.source_config["num_samples"] == 3

    @pytest.mark.asyncio
    async def test_successful_corpus_creation_with_config(
        self, temp_db, mock_datasets, mock_aiofiles, sample_dataset
    ):
        """Test successful corpus creation with config name"""
        # Arrange
        mock_datasets.return_value = sample_dataset
        mock_aiofiles.open.return_value.__aenter__.return_value.write = AsyncMock()

        # Act
        request = create_hf_request(config_name="test-config")
        result = await create_huggingface_corpus(request, db=temp_db)

        # Assert
        assert result["name"] == "test_corpus"
        assert result["id"] is not None

        # Check that load_dataset was called with config
        mock_datasets.assert_called_once_with("test-dataset", "test-config", split="train")

        # Check database record
        corpus = temp_db.query(Corpus).filter(Corpus.id == result["id"]).first()
        assert corpus.source_config["config_name"] == "test-config"

    @pytest.mark.asyncio
    async def test_dataset_dict_with_split_resolution(
        self, temp_db, mock_datasets, mock_aiofiles, sample_dataset_dict
    ):
        """Test handling of DatasetDict with split resolution"""
        # Arrange
        mock_datasets.return_value = sample_dataset_dict
        mock_aiofiles.open.return_value.__aenter__.return_value.write = AsyncMock()

        # Act
        request = create_hf_request()
        result = await create_huggingface_corpus(request, db=temp_db)

        # Assert
        assert result["name"] == "test_corpus"
        assert result["id"] is not None

        # Check database record
        corpus = temp_db.query(Corpus).filter(Corpus.id == result["id"]).first()
        assert corpus.source_config["num_samples"] == 2  # train split has 2 samples

    @pytest.mark.asyncio
    async def test_iterable_dataset_rejection(
        self, temp_db, mock_datasets, sample_iterable_dataset
    ):
        """Test that iterable datasets are rejected early"""
        # Arrange
        mock_datasets.return_value = sample_iterable_dataset

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            request = create_hf_request()
            await create_huggingface_corpus(request, db=temp_db)

        assert exc_info.value.status_code == 400
        assert "Iterable (streaming) datasets" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_dataset_dict_invalid_split(
        self, temp_db, mock_datasets, sample_dataset_dict
    ):
        """Test handling of invalid split in DatasetDict"""
        # Arrange
        mock_datasets.return_value = sample_dataset_dict

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            request = create_hf_request(split="invalid_split")
            await create_huggingface_corpus(request, db=temp_db)

        assert exc_info.value.status_code == 400
        assert "Split 'invalid_split' not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_missing_text_column(
        self, temp_db, mock_datasets, sample_dataset
    ):
        """Test handling of missing text column"""
        # Arrange
        mock_datasets.return_value = sample_dataset

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            request = create_hf_request(text_column="nonexistent_column")
            await create_huggingface_corpus(request, db=temp_db)

        assert exc_info.value.status_code == 400
        assert "Column 'nonexistent_column' not found" in exc_info.value.detail
        assert "Available columns: text, id" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_dataset_with_none_column_names(
        self, temp_db, mock_datasets
    ):
        """Test handling of dataset with None column names"""
        # Arrange
        mock_dataset = Mock()
        mock_dataset.column_names = None
        mock_datasets.return_value = mock_dataset

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            request = create_hf_request()
            await create_huggingface_corpus(request, db=temp_db)

        assert exc_info.value.status_code == 400
        assert "Column 'text' not found in dataset" in exc_info.value.detail
        assert "Available columns: none" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_config_name_missing_error(
        self, temp_db, mock_datasets
    ):
        """Test handling of config name missing error"""
        # Arrange
        mock_datasets.side_effect = ValueError("Config name is missing")

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            request = create_hf_request()
            await create_huggingface_corpus(request, db=temp_db)

        assert exc_info.value.status_code == 400
        assert "This dataset requires a config name" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_general_value_error(
        self, temp_db, mock_datasets
    ):
        """Test handling of general ValueError"""
        # Arrange
        mock_datasets.side_effect = ValueError("Some other error")

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            request = create_hf_request()
            await create_huggingface_corpus(request, db=temp_db)

        assert exc_info.value.status_code == 400
        assert "Some other error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_database_rollback_on_error(
        self, temp_db, mock_datasets, mock_aiofiles, sample_dataset
    ):
        """Test that database rollback occurs on error"""
        # Arrange
        mock_datasets.return_value = sample_dataset
        mock_aiofiles.open.return_value.__aenter__.return_value.write.side_effect = Exception("File write error")

        # Get initial corpus count
        initial_count = temp_db.query(Corpus).count()

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            request = create_hf_request()
            await create_huggingface_corpus(request, db=temp_db)

        assert exc_info.value.status_code == 400
        assert "File write error" in exc_info.value.detail

        # Verify no corpus was created (rollback occurred)
        final_count = temp_db.query(Corpus).count()
        assert final_count == initial_count

    @pytest.mark.asyncio
    async def test_file_creation_and_content(
        self, temp_db, mock_datasets, mock_aiofiles, sample_dataset, mock_os_makedirs
    ):
        """Test that file is created with correct content"""
        # Arrange
        mock_datasets.return_value = sample_dataset
        mock_write = AsyncMock()
        mock_aiofiles.open.return_value.__aenter__.return_value.write = mock_write

        # Act
        request = create_hf_request()
        await create_huggingface_corpus(request, db=temp_db)

        # Assert
        # Check that directory was created
        mock_os_makedirs.assert_called_once_with("data", exist_ok=True)

        # Check that file was opened for writing
        mock_aiofiles.open.assert_called_once_with("data/hf_corpus_test_corpus.txt", "w", encoding="utf-8")

        # Check that content was written
        mock_write.assert_called_once()
        written_content = mock_write.call_args[0][0]
        expected_content = "This is the first sample text.\n\nThis is the second sample text.\n\nThis is the third sample text with more content."
        assert written_content == expected_content

    @pytest.mark.asyncio
    async def test_empty_texts_handling(
        self, temp_db, mock_datasets, mock_aiofiles
    ):
        """Test handling of empty or None texts in dataset"""
        # Arrange
        dataset_with_empty = Dataset.from_dict({
            "text": ["Text 1", "", None, "Text 4"],
            "id": [1, 2, 3, 4]
        })
        mock_datasets.return_value = dataset_with_empty
        mock_write = AsyncMock()
        mock_aiofiles.open.return_value.__aenter__.return_value.write = mock_write

        # Act
        request = create_hf_request()
        await create_huggingface_corpus(request, db=temp_db)

        # Assert
        mock_write.assert_called_once()
        written_content = mock_write.call_args[0][0]
        expected_content = "Text 1\n\nText 4"  # Empty and None texts should be filtered out
        assert written_content == expected_content

    @pytest.mark.asyncio
    async def test_source_config_structure(
        self, temp_db, mock_datasets, mock_aiofiles, sample_dataset
    ):
        """Test that source_config contains all expected fields"""
        # Arrange
        mock_datasets.return_value = sample_dataset
        mock_aiofiles.open.return_value.__aenter__.return_value.write = AsyncMock()

        # Act
        request = create_hf_request(config_name="test-config", description="Test description")
        result = await create_huggingface_corpus(request, db=temp_db)

        # Assert
        corpus = temp_db.query(Corpus).filter(Corpus.id == result["id"]).first()
        source_config = corpus.source_config

        expected_keys = {
            "dataset_name", "config_name", "split", "text_column",
            "file_path", "num_samples"
        }
        assert set(source_config.keys()) == expected_keys

        assert source_config["dataset_name"] == "test-dataset"
        assert source_config["config_name"] == "test-config"
        assert source_config["split"] == "train"
        assert source_config["text_column"] == "text"
        assert source_config["file_path"] == "data/hf_corpus_test_corpus.txt"
        assert source_config["num_samples"] == 3

    @pytest.mark.asyncio
    async def test_response_structure(
        self, temp_db, mock_datasets, mock_aiofiles, sample_dataset
    ):
        """Test that response has correct structure"""
        # Arrange
        mock_datasets.return_value = sample_dataset
        mock_aiofiles.open.return_value.__aenter__.return_value.write = AsyncMock()

        # Act
        request = create_hf_request()
        result = await create_huggingface_corpus(request, db=temp_db)

        # Assert
        assert isinstance(result, dict)
        assert "id" in result
        assert "name" in result
        assert "message" in result

        assert result["name"] == "test_corpus"
        assert "successfully" in result["message"]
        assert "3 samples loaded" in result["message"]
        assert isinstance(result["id"], int)
