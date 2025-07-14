from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_list_models_empty():
    """Test listing models when catalogue is empty"""
    response = client.get("/api/models")
    assert response.status_code == 200
    assert response.json() == []


def test_download_model():
    """Test downloading a model"""
    # Test with a small, fast model
    model_data = {
        "huggingface_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_type": "embedding",
        "name": "Test Embedding Model",
        "description": "A test embedding model"
    }

    response = client.post("/api/models/download", json=model_data)
    assert response.status_code == 200

    model = response.json()
    assert model["name"] == "Test Embedding Model"
    assert model["model_type"] == "embedding"
    assert model["huggingface_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert model["is_active"] is True


def test_download_duplicate_model():
    """Test that downloading the same model twice fails"""
    model_data = {
        "huggingface_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_type": "embedding",
        "name": "Test Embedding Model",
        "description": "A test embedding model"
    }

    # First download should succeed
    response1 = client.post("/api/models/download", json=model_data)
    assert response1.status_code == 200

    # Second download should fail
    response2 = client.post("/api/models/download", json=model_data)
    assert response2.status_code == 400
    assert "already exists" in response2.json()["detail"]


def test_list_models_by_type():
    """Test filtering models by type"""
    # First, add an embedding model
    embedding_data = {
        "huggingface_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_type": "embedding",
        "name": "Test Embedding",
        "description": "Test embedding model"
    }
    client.post("/api/models/download", json=embedding_data)

    # Add a reranker model
    reranker_data = {
        "huggingface_name": "BAAI/bge-reranker-v2-m3",
        "model_type": "reranker",
        "name": "Test Reranker",
        "description": "Test reranker model"
    }
    client.post("/api/models/download", json=reranker_data)

    # Test filtering by embedding type
    response = client.get("/api/models?model_type=embedding")
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 1
    assert all(model["model_type"] == "embedding" for model in models)

    # Test filtering by reranker type
    response = client.get("/api/models?model_type=reranker")
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 1
    assert all(model["model_type"] == "reranker" for model in models)


def test_delete_model():
    """Test deleting a model"""
    # First, add a model
    model_data = {
        "huggingface_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_type": "embedding",
        "name": "Test Model for Deletion",
        "description": "A model to be deleted"
    }

    create_response = client.post("/api/models/download", json=model_data)
    assert create_response.status_code == 200
    model_id = create_response.json()["id"]

    # Delete the model
    delete_response = client.delete(f"/api/models/{model_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["message"] == "Model deleted successfully"

    # Verify it's gone
    get_response = client.get(f"/api/models/{model_id}")
    assert get_response.status_code == 404


def test_get_model_info():
    """Test getting detailed model information"""
    # First, add a model
    model_data = {
        "huggingface_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_type": "embedding",
        "name": "Test Model for Info",
        "description": "A model to get info for"
    }

    create_response = client.post("/api/models/download", json=model_data)
    assert create_response.status_code == 200
    model_id = create_response.json()["id"]

    # Wait a bit for the background download to complete
    import time
    time.sleep(5)

    # Get model info
    info_response = client.get(f"/api/models/{model_id}/info")
    assert info_response.status_code == 200

    info = info_response.json()
    assert info["name"] == "Test Model for Info"
    assert info["model_type"] == "embedding"
    assert "huggingface_info" in info


def test_models_page_route():
    """Test that the models page route works"""
    response = client.get("/models")
    assert response.status_code == 200
    assert "Model Catalogue" in response.text
