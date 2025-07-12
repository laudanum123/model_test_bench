from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional, cast
import aiofiles
import os
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset, IterableDatasetDict, get_dataset_config_names, get_dataset_split_names, get_dataset_infos
from app.database import get_db, Corpus
from app.models.schemas import CorpusCreate, Corpus as CorpusSchema, HuggingFaceCorpusRequest
from app.typing_helpers.vector_store_protocols import DatasetItem

router = APIRouter(prefix="/corpus", tags=["corpus"])


@router.get("/huggingface/metadata")
async def get_huggingface_dataset_metadata(dataset_name: str):
    """Get metadata for a HuggingFace dataset including available configs, splits, and columns"""
    print(f"DEBUG: Received request for dataset metadata: {dataset_name}")
    try:
        # Get available configs
        config_names = get_dataset_config_names(dataset_name)
        
        # If no configs (single config dataset), use None
        if not config_names:
            config_names = [None]
        
        metadata = {
            "dataset_name": dataset_name,
            "configs": [],
            "default_config": config_names[0] if config_names else None
        }
        
        # Get dataset infos for metadata
        try:
            dataset_infos = get_dataset_infos(dataset_name)
        except Exception as e:
            print(f"Warning: Could not get dataset infos: {e}")
            dataset_infos = {}
        
        for config_name in config_names:
            config_data = {"name": config_name, "splits": [], "columns": []}
            
            try:
                # Get splits for this config
                if config_name:
                    split_names = get_dataset_split_names(dataset_name, config_name)
                else:
                    split_names = get_dataset_split_names(dataset_name)
                
                config_data["splits"] = split_names
                
                # Get columns from dataset info if available
                if dataset_infos and config_name in dataset_infos:
                    info = dataset_infos[config_name]
                    if hasattr(info, 'features') and info.features:
                        config_data["columns"] = list(info.features.keys())
                elif dataset_infos and len(dataset_infos) == 1:
                    # Single config dataset
                    info = list(dataset_infos.values())[0]
                    if hasattr(info, 'features') and info.features:
                        config_data["columns"] = list(info.features.keys())
                else:
                    # Fallback: try to get columns from first split (but don't download full dataset)
                    try:
                        if config_name:
                            # Load just the first sample to get column names
                            dataset = load_dataset(dataset_name, config_name, split=split_names[0] if split_names else "train", streaming=True)
                            # Get first item to see structure
                            first_item = cast(DatasetItem, next(iter(dataset)))
                            if hasattr(first_item, 'keys'):
                                config_data["columns"] = list(first_item.keys())
                            else:
                                # For streaming datasets, try to get features safely
                                features = getattr(dataset, 'features', None)
                                config_data["columns"] = list(features.keys()) if features else ["text"]
                        else:
                            dataset = load_dataset(dataset_name, split=split_names[0] if split_names else "train", streaming=True)
                            first_item = cast(DatasetItem, next(iter(dataset)))
                            if hasattr(first_item, 'keys'):
                                config_data["columns"] = list(first_item.keys())
                            else:
                                # For streaming datasets, try to get features safely
                                features = getattr(dataset, 'features', None)
                                config_data["columns"] = list(features.keys()) if features else ["text"]
                    except Exception as e:
                        print(f"Could not get columns for config {config_name}: {e}")
                        # Set default columns
                        config_data["columns"] = ["text"]
                    
            except Exception as e:
                # If we can't get info for this config, skip it
                print(f"Error getting metadata for config {config_name}: {e}")
                continue
            
            metadata["configs"].append(config_data)
        
        return metadata
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching dataset metadata: {str(e)}")


@router.post("/", response_model=CorpusSchema)
async def create_corpus(
    corpus: CorpusCreate,
    db: Session = Depends(get_db)
):
    """Create a new corpus"""
    try:
        db_corpus = Corpus(
            name=corpus.name,
            description=corpus.description,
            source=corpus.source,
            source_config=corpus.source_config
        )
        db.add(db_corpus)
        db.commit()
        db.refresh(db_corpus)
        return db_corpus
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[CorpusSchema])
async def list_corpora(db: Session = Depends(get_db)):
    """List all corpora"""
    corpora = db.query(Corpus).all()
    return corpora


@router.get("/{corpus_id}", response_model=CorpusSchema)
async def get_corpus(corpus_id: int, db: Session = Depends(get_db)):
    """Get a specific corpus"""
    corpus = db.query(Corpus).filter(Corpus.id == corpus_id).first()
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    return corpus


@router.post("/upload")
async def upload_corpus(
    name: str,
    description: Optional[str] = None,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a text file as a corpus"""
    try:

        if file.filename is None:
            raise HTTPException(status_code=400, detail="File name is required")
        
        # Validate file type
        if not file.filename.endswith(('.txt', '.md', '.csv')):
            raise HTTPException(status_code=400, detail="Only .txt, .md, and .csv files are supported")
        
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Save file to data directory
        file_path = f"data/corpus_{name}_{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        # Create corpus record
        corpus = Corpus(
            name=name,
            description=description,
            source="upload",
            source_config={
                "file_path": file_path,
                "file_size": len(content),
                "file_type": file.filename.split('.')[-1]
            }
        )
        
        db.add(corpus)
        db.commit()
        db.refresh(corpus)
        
        return {
            "id": corpus.id,
            "name": corpus.name,
            "message": f"Corpus uploaded successfully. Text length: {len(text_content)} characters"
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/huggingface")
async def create_huggingface_corpus(
    request: HuggingFaceCorpusRequest,
    db: Session = Depends(get_db)
):
    """Create a corpus from a HuggingFace dataset"""
    try:
        # Load dataset from HuggingFace
        if request.config_name:
            dataset_raw = load_dataset(request.dataset_name, request.config_name, split=request.split)
        else:
            dataset_raw = load_dataset(request.dataset_name, split=request.split)
        
        # 1️⃣ Reject streaming datasets early
        if isinstance(dataset_raw, (IterableDataset, IterableDatasetDict)):
            raise HTTPException(
                status_code=400,
                detail="Iterable (streaming) datasets don't expose their length; "
                "remove `streaming=True` or count samples another way."
            )

        # 2️⃣ Resolve a specific split if the user gave the dataset (not a split) by mistake
        if isinstance(dataset_raw, DatasetDict):
            if request.split not in dataset_raw:
                raise HTTPException(
                    status_code=400,
                    detail=f"Split '{request.split}' not found in dataset")
            dataset: Dataset = dataset_raw[request.split]
        else:
            dataset = cast(Dataset, dataset_raw)

         # Extract text content
        column_names = dataset.column_names
        if column_names is None or request.text_column not in column_names:
            available_columns = ", ".join(column_names) if column_names else "none"
            raise HTTPException(
                status_code=400, 
                detail=f"Column '{request.text_column}' not found in dataset. Available columns: {available_columns}"
            )
        
        texts = dataset[request.text_column]
        combined_text = "\n\n".join([str(text) for text in texts if text])
        
        # Save to file
        file_path = f"data/hf_corpus_{request.name}.txt"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(combined_text)
        
        # Create corpus record
        corpus = Corpus(
            name=request.name,
            description=request.description,
            source="huggingface",
            source_config={
                "dataset_name": request.dataset_name,
                "config_name": request.config_name,
                "split": request.split,
                "text_column": request.text_column,
                "file_path": file_path,
                "num_samples": len(dataset)
            }
        )
        
        db.add(corpus)
        db.commit()
        db.refresh(corpus)
        
        return {
            "id": corpus.id,
            "name": corpus.name,
            "message": f"HuggingFace corpus created successfully. {len(dataset)} samples loaded."
        }
    
    except ValueError as e:
        # Handle config name missing error specifically
        if "Config name is missing" in str(e):
            raise HTTPException(
                status_code=400,
                detail=f"This dataset requires a config name. Please specify one of the available configs. Error: {str(e)}"
            )
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{corpus_id}/content")
async def get_corpus_content(corpus_id: int, db: Session = Depends(get_db)):
    """Get the text content of a corpus"""
    corpus = db.query(Corpus).filter(Corpus.id == corpus_id).first()
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    try:
        if corpus.source in ["upload", "huggingface"]:
            file_path = corpus.source_config.get("file_path")
            if not file_path or not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Corpus file not found")
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            return {
                "corpus_id": corpus_id,
                "content": content,
                "length": len(content)
            }
        else:
            raise HTTPException(status_code=400, detail="Unsupported corpus source")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{corpus_id}")
async def delete_corpus(corpus_id: int, db: Session = Depends(get_db)):
    """Delete a corpus"""
    corpus = db.query(Corpus).filter(Corpus.id == corpus_id).first()
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    try:
        # Delete associated file if it exists
        if corpus.source_config is not None and "file_path" in corpus.source_config:
            file_path = corpus.source_config["file_path"]
            if file_path is not None and os.path.exists(str(file_path)):
                os.remove(str(file_path))
        
        db.delete(corpus)
        db.commit()
        
        return {"message": "Corpus deleted successfully"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e)) 