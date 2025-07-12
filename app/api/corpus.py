from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
import aiofiles
import os
from datasets import load_dataset
from app.database import get_db, Corpus
from app.models.schemas import CorpusCreate, Corpus as CorpusSchema
from app.config import settings

router = APIRouter(prefix="/corpus", tags=["corpus"])


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
    name: str,
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
    description: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Create a corpus from a HuggingFace dataset"""
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_name, split=split)
        
        # Extract text content
        if text_column not in dataset.column_names:
            raise HTTPException(status_code=400, detail=f"Column '{text_column}' not found in dataset")
        
        texts = dataset[text_column]
        combined_text = "\n\n".join([str(text) for text in texts if text])
        
        # Save to file
        file_path = f"data/hf_corpus_{name}.txt"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(combined_text)
        
        # Create corpus record
        corpus = Corpus(
            name=name,
            description=description,
            source="huggingface",
            source_config={
                "dataset_name": dataset_name,
                "split": split,
                "text_column": text_column,
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
        if corpus.source == "upload" or corpus.source == "huggingface":
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
        if corpus.source_config and "file_path" in corpus.source_config:
            file_path = corpus.source_config["file_path"]
            if os.path.exists(file_path):
                os.remove(file_path)
        
        db.delete(corpus)
        db.commit()
        
        return {"message": "Corpus deleted successfully"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e)) 