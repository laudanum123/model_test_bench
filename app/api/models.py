import logging
import os
import shutil
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.database import ModelCatalogue, get_db
from app.models.schemas import (
    ModelCatalogue as ModelCatalogueSchema,
)
from app.models.schemas import (
    ModelDownloadRequest,
    ModelType,
    ModelUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/", response_model=list[ModelCatalogueSchema])
async def list_models(
    model_type: ModelType | None = None,
    db: Session = Depends(get_db)
):
    """List all models in the catalogue, optionally filtered by type"""
    logger.info(f"Listing models with filter: {model_type}")

    query = db.query(ModelCatalogue)
    if model_type:
        query = query.filter(ModelCatalogue.model_type == model_type.value)
        logger.debug(f"Applied model type filter: {model_type.value}")

    models = query.filter(ModelCatalogue.is_active == 1).all()
    logger.info(f"Found {len(models)} active models")
    return models


@router.get("/{model_id}", response_model=ModelCatalogueSchema)
async def get_model(model_id: int, db: Session = Depends(get_db)):
    """Get a specific model by ID"""
    logger.info(f"Retrieving model with id: {model_id}")

    model = db.query(ModelCatalogue).filter(ModelCatalogue.id == model_id).first()
    if not model:
        logger.error(f"Model not found with id: {model_id}")
        raise HTTPException(status_code=404, detail="Model not found")

    logger.info(f"Found model: {model.name} ({model.huggingface_name})")
    return model


@router.post("/download", response_model=ModelCatalogueSchema)
async def download_model(
    request: ModelDownloadRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Download a model from HuggingFace and add it to the catalogue"""
    logger.info(f"Starting model download request: {request.huggingface_name}")

    try:
        # Check if model already exists
        existing_model = db.query(ModelCatalogue).filter(
            ModelCatalogue.huggingface_name == request.huggingface_name
        ).first()

        if existing_model:
            logger.warning(f"Model {request.huggingface_name} already exists in catalogue")
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.huggingface_name} already exists in catalogue"
            )

        # Create model entry first
        model_name = request.name or request.huggingface_name
        provider = _determine_provider(request.model_type, request.huggingface_name)

        logger.debug(f"Creating model entry: name={model_name}, type={request.model_type}, provider={provider}")

        db_model = ModelCatalogue(
            name=model_name,
            model_type=request.model_type.value,
            provider=provider,
            huggingface_name=request.huggingface_name,
            description=request.description,
            is_active=1
        )

        db.add(db_model)
        db.commit()
        db.refresh(db_model)

        logger.info(f"Created model entry with id: {db_model.id}")

        # Start background download task
        background_tasks.add_task(
            _download_model_task,
            cast("int", db_model.id),
            request.huggingface_name,
            request.model_type,
            db
        )

        logger.info(f"Started background download task for model {request.huggingface_name}")

        return db_model

    except Exception as e:
        db.rollback()
        logger.error(f"Error creating model entry: {e!s}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.put("/{model_id}", response_model=ModelCatalogueSchema)
async def update_model(
    model_id: int,
    request: ModelUpdateRequest,
    db: Session = Depends(get_db)
):
    """Update model information"""
    logger.info(f"Updating model with id: {model_id}")

    model = db.query(ModelCatalogue).filter(ModelCatalogue.id == model_id).first()
    if not model:
        logger.error(f"Model not found with id: {model_id}")
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name
            logger.debug(f"Updating name to: {request.name}")
        if request.description is not None:
            update_data["description"] = request.description
            logger.debug(f"Updating description to: {request.description}")
        if request.is_active is not None:
            update_data["is_active"] = 1 if request.is_active else 0
            logger.debug(f"Updating is_active to: {update_data['is_active']}")

        db.query(ModelCatalogue).filter(ModelCatalogue.id == model_id).update(update_data)
        db.commit()
        db.refresh(model)

        logger.info(f"Successfully updated model {model.name}")

        return model

    except Exception as e:
        db.rollback()
        logger.error(f"Error updating model: {e!s}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/{model_id}")
async def delete_model(model_id: int, db: Session = Depends(get_db)):
    """Delete a model from the catalogue and remove local files"""
    logger.info(f"Deleting model with id: {model_id}")

    model = db.query(ModelCatalogue).filter(ModelCatalogue.id == model_id).first()
    if not model:
        logger.error(f"Model not found with id: {model_id}")
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # Remove local files if they exist
        if model.local_path and os.path.exists(model.local_path):
            logger.debug(f"Removing local files at: {model.local_path}")
            try:
                if os.path.isdir(model.local_path):
                    shutil.rmtree(model.local_path)
                    logger.debug(f"Removed directory: {model.local_path}")
                else:
                    os.remove(model.local_path)
                    logger.debug(f"Removed file: {model.local_path}")
                logger.info(f"Removed local files for model {model.huggingface_name}")
            except Exception as e:
                logger.warning(f"Could not remove local files for model {model.huggingface_name}: {e!s}")
        else:
            logger.debug(f"No local files to remove for model {model.huggingface_name}")

        # Delete from database
        db.query(ModelCatalogue).filter(ModelCatalogue.id == model_id).delete()
        db.commit()

        logger.info(f"Successfully deleted model {model.huggingface_name}")

        return {"message": "Model deleted successfully", "model_id": model_id}

    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting model: {e!s}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{model_id}/info")
async def get_model_info(model_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a model"""
    logger.info(f"Getting detailed info for model id: {model_id}")

    model = db.query(ModelCatalogue).filter(ModelCatalogue.id == model_id).first()
    if not model:
        logger.error(f"Model not found with id: {model_id}")
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # Get model info from HuggingFace
        logger.debug(f"Fetching HuggingFace info for: {model.huggingface_name}")
        info = model_info(model.huggingface_name)

        # Extract relevant information
        model_details = {
            "id": model.id,
            "name": model.name,
            "huggingface_name": model.huggingface_name,
            "model_type": model.model_type,
            "provider": model.provider,
            "description": model.description,
            "local_path": model.local_path,
            "is_active": bool(model.is_active),
            "created_at": model.created_at,
            "huggingface_info": {
                "tags": getattr(info, "tags", []),
                "downloads": getattr(info, "downloads", 0),
                "likes": getattr(info, "likes", 0),
                "last_modified": getattr(info, "lastModified", None),
                "author": getattr(info, "author", {}).get("name") if getattr(info, "author", None) else None,
            }
        }

        # Add model-specific information
        if model.model_type == "embedding":
            logger.debug(f"Getting embedding model info for: {model.huggingface_name}")
            model_details["embedding_info"] = _get_embedding_model_info(model.huggingface_name)
        elif model.model_type == "reranker":
            logger.debug(f"Getting reranker model info for: {model.huggingface_name}")
            model_details["reranker_info"] = _get_reranker_model_info(model.huggingface_name)

        logger.info(f"Successfully retrieved detailed info for model {model.name}")
        return model_details

    except Exception as e:
        logger.error(f"Error getting model info: {e!s}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _download_model_task(model_id: int, huggingface_name: str, model_type: ModelType, db: Session):
    """Background task to download model files"""
    logger.info(f"Starting download of model {huggingface_name} (id: {model_id})")

    try:
        # Create models directory
        models_dir = Path("data/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created models directory: {models_dir}")

        model_dir = models_dir / huggingface_name.replace("/", "_")
        model_dir.mkdir(exist_ok=True)
        logger.debug(f"Created model directory: {model_dir}")

        # Download model based on type
        if model_type == ModelType.EMBEDDING:
            logger.debug(f"Downloading embedding model: {huggingface_name}")
            local_path = await _download_embedding_model(huggingface_name, model_dir)
        elif model_type == ModelType.RERANKER:
            logger.debug(f"Downloading reranker model: {huggingface_name}")
            local_path = await _download_reranker_model(huggingface_name, model_dir)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Update database with local path and model info
        logger.debug(f"Extracting model info for: {huggingface_name}")
        model_info = _extract_model_info(huggingface_name, model_type, local_path)

        db.query(ModelCatalogue).filter(ModelCatalogue.id == model_id).update({
            "local_path": str(local_path),
            "model_info": model_info
        })
        db.commit()

        logger.info(f"Successfully downloaded model {huggingface_name} to {local_path}")

    except Exception as e:
        logger.error(f"Error downloading model {huggingface_name}: {e!s}", exc_info=True)
        # Update status to indicate failure
        db.query(ModelCatalogue).filter(ModelCatalogue.id == model_id).update({
            "is_active": 0
        })
        db.commit()
        logger.error(f"Marked model {huggingface_name} as inactive due to download failure")


async def _download_embedding_model(huggingface_name: str, model_dir: Path) -> Path:
    """Download embedding model files"""
    logger.debug(f"Downloading embedding model: {huggingface_name}")

    try:
        # Load model to trigger download
        _ = SentenceTransformer(huggingface_name, cache_folder=str(model_dir))
        logger.debug(f"Successfully loaded SentenceTransformer model: {huggingface_name}")

        # Get the actual cache directory
        cache_dir = model_dir / "sentence_transformers"
        if cache_dir.exists():
            logger.debug(f"Using cache directory: {cache_dir}")
            return cache_dir
        else:
            logger.debug(f"Using model directory: {model_dir}")
            return model_dir

    except Exception as e:
        logger.error(f"Failed to download embedding model {huggingface_name}: {e!s}")
        raise Exception(f"Failed to download embedding model: {e!s}") from e


async def _download_reranker_model(huggingface_name: str, model_dir: Path) -> Path:
    """Download reranker model files"""
    logger.debug(f"Downloading reranker model: {huggingface_name}")

    try:
        # Download tokenizer and model
        logger.debug(f"Downloading tokenizer for: {huggingface_name}")
        tokenizer = AutoTokenizer.from_pretrained(huggingface_name, cache_dir=str(model_dir))

        logger.debug(f"Downloading model for: {huggingface_name}")
        model = AutoModelForSequenceClassification.from_pretrained(huggingface_name, cache_dir=str(model_dir))

        # Save to local directory
        logger.debug(f"Saving tokenizer and model to: {model_dir}")
        tokenizer.save_pretrained(str(model_dir))
        model.save_pretrained(str(model_dir))

        logger.debug(f"Successfully downloaded reranker model: {huggingface_name}")
        return model_dir

    except Exception as e:
        logger.error(f"Failed to download reranker model {huggingface_name}: {e!s}")
        raise Exception(f"Failed to download reranker model: {e!s}") from e


def _determine_provider(model_type: ModelType, huggingface_name: str) -> str:
    """Determine the provider based on model type and name"""
    logger.debug(f"Determining provider for model type: {model_type}, name: {huggingface_name}")

    if model_type == ModelType.EMBEDDING:
        provider = "sentence_transformers"
    elif model_type == ModelType.RERANKER:
        if "cross-encoder" in huggingface_name.lower():
            provider = "sentence_transformers"
        else:
            provider = "transformers"
    else:
        provider = "transformers"

    logger.debug(f"Determined provider: {provider}")
    return provider


def _extract_model_info(huggingface_name: str, model_type: ModelType, local_path: Path) -> dict[str, Any]:
    """Extract model information like dimensions, etc."""
    logger.debug(f"Extracting model info for: {huggingface_name}, type: {model_type}")

    try:
        if model_type == ModelType.EMBEDDING:
            info = _get_embedding_model_info(huggingface_name)
        elif model_type == ModelType.RERANKER:
            info = _get_reranker_model_info(huggingface_name)
        else:
            info = {}

        logger.debug(f"Extracted model info: {info}")
        return info
    except Exception as e:
        logger.warning(f"Could not extract model info for {huggingface_name}: {e!s}")
        return {}


def _get_embedding_model_info(huggingface_name: str) -> dict[str, Any]:
    """Get embedding model information"""
    logger.debug(f"Getting embedding model info for: {huggingface_name}")

    try:
        model = SentenceTransformer(huggingface_name)
        info = {
            "embedding_dimension": model.get_sentence_embedding_dimension(),
            "max_seq_length": model.max_seq_length,
            "model_type": "sentence_transformer"
        }
        logger.debug(f"Embedding model info: {info}")
        return info
    except Exception as e:
        logger.warning(f"Could not get embedding model info: {e!s}")
        return {}


def _get_reranker_model_info(huggingface_name: str) -> dict[str, Any]:
    """Get reranker model information"""
    logger.debug(f"Getting reranker model info for: {huggingface_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(huggingface_name)
        model = AutoModelForSequenceClassification.from_pretrained(huggingface_name)

        info = {
            "vocab_size": tokenizer.vocab_size,
            "max_length": tokenizer.model_max_length,
            "num_labels": model.config.num_labels,
            "model_type": "sequence_classification"
        }
        logger.debug(f"Reranker model info: {info}")
        return info
    except Exception as e:
        logger.warning(f"Could not get reranker model info: {e!s}")
        return {}
