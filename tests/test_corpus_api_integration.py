import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.database import Corpus


class TestCorpusAPIEndpoints:
    """Integration tests for corpus API endpoints"""

    @pytest.mark.asyncio
    async def test_huggingface_endpoint_success(
        self, client, mock_datasets, mock_aiofiles, sample_dataset
    ):
        """Test successful HuggingFace corpus creation via API endpoint"""
        # Arrange
        mock_datasets.return_value = sample_dataset
        mock_aiofiles.open.return_value.__aenter__.return_value.write = AsyncMock()
        
        # Act
        response = client.post(
            "/corpus/huggingface",
            params={
                "name": "test_corpus",
                "dataset_name": "test-dataset",
                "split": "train",
                "text_column": "text",
                "description": "Test description"
            }
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_corpus"
        assert "successfully" in data["message"]
        assert data["id"] is not None

    @pytest.mark.asyncio
    async def test_huggingface_endpoint_with_config(
        self, client, mock_datasets, mock_aiofiles, sample_dataset
    ):
        """Test HuggingFace corpus creation with config name via API endpoint"""
        # Arrange
        mock_datasets.return_value = sample_dataset
        mock_aiofiles.open.return_value.__aenter__.return_value.write = AsyncMock()
        
        # Act
        response = client.post(
            "/corpus/huggingface",
            params={
                "name": "test_corpus",
                "dataset_name": "test-dataset",
                "config_name": "test-config",
                "split": "train",
                "text_column": "text"
            }
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_corpus"
        assert data["id"] is not None
        
        # Verify load_dataset was called with config
        mock_datasets.assert_called_once_with("test-dataset", "test-config", split="train")

    @pytest.mark.asyncio
    async def test_huggingface_endpoint_iterable_dataset_error(
        self, client, mock_datasets, sample_iterable_dataset
    ):
        """Test HuggingFace endpoint error handling for iterable datasets"""
        # Arrange
        mock_datasets.return_value = sample_iterable_dataset
        
        # Act
        response = client.post(
            "/corpus/huggingface",
            params={
                "name": "test_corpus",
                "dataset_name": "test-dataset",
                "split": "train",
                "text_column": "text"
            }
        )
        
        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Iterable (streaming) datasets" in data["detail"]

    @pytest.mark.asyncio
    async def test_huggingface_endpoint_missing_column_error(
        self, client, mock_datasets, sample_dataset
    ):
        """Test HuggingFace endpoint error handling for missing text column"""
        # Arrange
        mock_datasets.return_value = sample_dataset
        
        # Act
        response = client.post(
            "/corpus/huggingface",
            params={
                "name": "test_corpus",
                "dataset_name": "test-dataset",
                "split": "train",
                "text_column": "nonexistent_column"
            }
        )
        
        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "Column 'nonexistent_column' not found" in data["detail"]

    @pytest.mark.asyncio
    async def test_huggingface_endpoint_config_missing_error(
        self, client, mock_datasets
    ):
        """Test HuggingFace endpoint error handling for missing config"""
        # Arrange
        mock_datasets.side_effect = ValueError("Config name is missing")
        
        # Act
        response = client.post(
            "/corpus/huggingface",
            params={
                "name": "test_corpus",
                "dataset_name": "test-dataset",
                "split": "train",
                "text_column": "text"
            }
        )
        
        # Assert
        assert response.status_code == 400
        data = response.json()
        assert "This dataset requires a config name" in data["detail"]

    @pytest.mark.asyncio
    async def test_huggingface_endpoint_database_integration(
        self, client, mock_datasets, mock_aiofiles, sample_dataset, temp_db
    ):
        """Test that HuggingFace corpus creation properly saves to database"""
        # Arrange
        mock_datasets.return_value = sample_dataset
        mock_aiofiles.open.return_value.__aenter__.return_value.write = AsyncMock()
        
        # Act
        response = client.post(
            "/corpus/huggingface",
            params={
                "name": "test_corpus",
                "dataset_name": "test-dataset",
                "split": "train",
                "text_column": "text",
                "description": "Test description"
            }
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Check database record
        corpus = temp_db.query(Corpus).filter(Corpus.id == data["id"]).first()
        assert corpus is not None
        assert corpus.name == "test_corpus"
        assert corpus.description == "Test description"
        assert corpus.source == "huggingface"
        assert corpus.source_config["dataset_name"] == "test-dataset"
        assert corpus.source_config["split"] == "train"
        assert corpus.source_config["text_column"] == "text"
        assert corpus.source_config["num_samples"] == 3

    @pytest.mark.asyncio
    async def test_list_corpora_endpoint(
        self, client, mock_datasets, mock_aiofiles, sample_dataset, temp_db
    ):
        """Test listing corpora endpoint"""
        # Arrange - Create a corpus first
        mock_datasets.return_value = sample_dataset
        mock_aiofiles.open.return_value.__aenter__.return_value.write = AsyncMock()
        
        create_response = client.post(
            "/corpus/huggingface",
            params={
                "name": "test_corpus",
                "dataset_name": "test-dataset",
                "split": "train",
                "text_column": "text"
            }
        )
        assert create_response.status_code == 200
        
        # Act
        response = client.get("/corpus/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "test_corpus"
        assert data[0]["source"] == "huggingface"

    @pytest.mark.asyncio
    async def test_get_corpus_endpoint(
        self, client, mock_datasets, mock_aiofiles, sample_dataset, temp_db
    ):
        """Test getting a specific corpus endpoint"""
        # Arrange - Create a corpus first
        mock_datasets.return_value = sample_dataset
        mock_aiofiles.open.return_value.__aenter__.return_value.write = AsyncMock()
        
        create_response = client.post(
            "/corpus/huggingface",
            params={
                "name": "test_corpus",
                "dataset_name": "test-dataset",
                "split": "train",
                "text_column": "text"
            }
        )
        assert create_response.status_code == 200
        corpus_id = create_response.json()["id"]
        
        # Act
        response = client.get(f"/corpus/{corpus_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == corpus_id
        assert data["name"] == "test_corpus"
        assert data["source"] == "huggingface"

    @pytest.mark.asyncio
    async def test_get_corpus_not_found(
        self, client
    ):
        """Test getting a non-existent corpus"""
        # Act
        response = client.get("/corpus/999")
        
        # Assert
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]

    @pytest.mark.asyncio
    async def test_get_corpus_content_endpoint(
        self, client, mock_datasets, mock_aiofiles, sample_dataset, temp_db
    ):
        """Test getting corpus content endpoint"""
        # Arrange - Create a corpus first
        mock_datasets.return_value = sample_dataset
        mock_aiofiles.open.return_value.__aenter__.return_value.write = AsyncMock()
        
        create_response = client.post(
            "/corpus/huggingface",
            params={
                "name": "test_corpus",
                "dataset_name": "test-dataset",
                "split": "train",
                "text_column": "text"
            }
        )
        assert create_response.status_code == 200
        corpus_id = create_response.json()["id"]
        
        # Patch aiofiles.open for reading and os.path.exists
        with patch("app.api.corpus.aiofiles.open", create=True) as mock_file_open, \
             patch("app.api.corpus.os.path.exists", return_value=True):
            mock_file_open.return_value.__aenter__.return_value.read = AsyncMock(return_value="Test content from file")
            
            # Act
            response = client.get(f"/corpus/{corpus_id}/content")
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["corpus_id"] == corpus_id
            assert data["content"] == "Test content from file"
            assert data["length"] == 22

    @pytest.mark.asyncio
    async def test_delete_corpus_endpoint(
        self, client, mock_datasets, mock_aiofiles, sample_dataset, temp_db
    ):
        """Test deleting a corpus endpoint"""
        # Arrange - Create a corpus first
        mock_datasets.return_value = sample_dataset
        mock_aiofiles.open.return_value.__aenter__.return_value.write = AsyncMock()
        
        create_response = client.post(
            "/corpus/huggingface",
            params={
                "name": "test_corpus",
                "dataset_name": "test-dataset",
                "split": "train",
                "text_column": "text"
            }
        )
        assert create_response.status_code == 200
        corpus_id = create_response.json()["id"]
        
        # Verify corpus exists
        corpus = temp_db.query(Corpus).filter(Corpus.id == corpus_id).first()
        assert corpus is not None
        
        # Mock file deletion
        with patch('app.api.corpus.os.path.exists', return_value=True), \
             patch('app.api.corpus.os.remove') as mock_remove:
            
            # Act
            response = client.delete(f"/corpus/{corpus_id}")
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert "deleted successfully" in data["message"]
            
            # Verify file deletion was attempted
            mock_remove.assert_called_once()
            
            # Verify corpus was deleted from database
            corpus = temp_db.query(Corpus).filter(Corpus.id == corpus_id).first()
            assert corpus is None

    @pytest.mark.asyncio
    async def test_delete_corpus_not_found(
        self, client
    ):
        """Test deleting a non-existent corpus"""
        # Act
        response = client.delete("/corpus/999")
        
        # Assert
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]

    @pytest.mark.asyncio
    async def test_create_corpus_endpoint(
        self, client, temp_db
    ):
        """Test creating a corpus via the general create endpoint"""
        # Arrange
        corpus_data = {
            "name": "test_corpus",
            "description": "Test description",
            "source": "custom",
            "source_config": {"custom_field": "custom_value"}
        }
        
        # Act
        response = client.post("/corpus/", json=corpus_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_corpus"
        assert data["source"] == "custom"
        assert data["id"] is not None
        
        # Check database record
        corpus = temp_db.query(Corpus).filter(Corpus.id == data["id"]).first()
        assert corpus is not None
        assert corpus.name == "test_corpus"
        assert corpus.source_config["custom_field"] == "custom_value" 