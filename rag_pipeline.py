import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import asyncio

# Core libraries
import pymongo
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import torch.nn.functional as F


# PDF processing
import PyPDF2
import pdfplumber
from pdfminer.high_level import extract_text

# Text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from datetime import datetime

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

auto_embedding = True

class RAGPipeline:
    def __init__(self, 
                 mongodb_uri: str,
                 database_name: str,
                 collection_name: str,
                 embedding_model: str
                 ):
        """
        Initialize RAG Pipeline with MongoDB Vector Search
        
        Args:
            mongodb_uri: MongoDB connection string
            database_name: Name of the database
            collection_name: Name of the collection for vectors
        """
        
        # MongoDB setup
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]            
        
        # Embedding model setup
        if auto_embedding:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        else:
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            self.embedding_model = AutoModel.from_pretrained(embedding_model)        
            self.embedding_dimension = self.embedding_model.config.hidden_size

        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        logger.info(f"RAG Pipeline initialized with embedding dimension: {self.embedding_dimension}")

    def create_vector_index(self):
        """Create vector search index in MongoDB"""
        try:
            vector_index_name = "rag_search_index"
            
            # Correct structure for search index model
            search_index_model = {
            "name": vector_index_name,
            "definition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "embedding": {
                            "type": "vector",
                            "dimensions": self.embedding_dimension,
                            "similarity": "cosine"
                        }
                    }
                }
            }
        }

            # Check if index already exists
            existing_indexes = list(self.collection.list_search_indexes())
            if not any(idx.get('name') == vector_index_name for idx in existing_indexes):
                result = self.collection.create_search_index(search_index_model)
                logger.info(f"Vector search index created successfully: {result}")
            else:
                logger.info("Vector search index already exists")
            
        except Exception as e:
            logger.error(f"Error creating vector index: {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF using multiple methods for robustness
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        text_content = ""
        metadata = {
            "filename": pdf_path.name,
            "file_path": str(pdf_path),
            "extraction_method": None,
            "page_count": 0,
            "extraction_timestamp": datetime.utcnow()
        }
        
        # Method 1: Try pdfplumber (best for structured PDFs)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata["page_count"] = len(pdf.pages)
                for page in pdf.pages:
                    if page.extract_text():
                        text_content += page.extract_text() + "\n\n"
                
                if text_content.strip():
                    metadata["extraction_method"] = "pdfplumber"
                    logger.info(f"Successfully extracted text using pdfplumber from {pdf_path.name}")
                    return {"text": text_content.strip(), "metadata": metadata}
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path.name}: {str(e)}")
        
        # Method 2: Try PyPDF2 (fallback)
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata["page_count"] = len(pdf_reader.pages)
                
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n\n"
                
                if text_content.strip():
                    metadata["extraction_method"] = "PyPDF2"
                    logger.info(f"Successfully extracted text using PyPDF2 from {pdf_path.name}")
                    return {"text": text_content.strip(), "metadata": metadata}
        except Exception as e:
            logger.warning(f"PyPDF2 failed for {pdf_path.name}: {str(e)}")
        
        # Method 3: Try pdfminer (last resort)
        try:
            text_content = extract_text(pdf_path)
            if text_content.strip():
                metadata["extraction_method"] = "pdfminer"
                logger.info(f"Successfully extracted text using pdfminer from {pdf_path.name}")
                return {"text": text_content.strip(), "metadata": metadata}
        except Exception as e:
            logger.error(f"All PDF extraction methods failed for {pdf_path.name}: {str(e)}")
            raise Exception(f"Could not extract text from PDF: {pdf_path.name}")

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        
        # Remove page numbers and headers (basic patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text, flags=re.IGNORECASE)
        
        return text.strip()

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text to be chunked
            metadata: Original document metadata
            
        Returns:
            List of chunks with metadata
        """
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(cleaned_text)
        
        chunk_documents = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:  # Filter out very small chunks
                chunk_doc = {
                    "text": chunk.strip(),
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "source_metadata": metadata.copy(),
                    "created_at": datetime.utcnow()
                }
                chunk_documents.append(chunk_doc)
        
        logger.info(f"Created {len(chunk_documents)} chunks from {metadata['filename']}")
        return chunk_documents

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embeddings as lists of floats
        """
        if auto_embedding:
            return self.generate_embeddings_Auto(texts)
        else:
            return self.generate_embeddings_Manual(texts)
                
    def generate_embeddings_Auto(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def generate_embeddings_Manual(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            if isinstance(texts, str):
                texts = [texts]
        
            # Tokenize
            inputs = self.embedding_tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )
        
            # Get embeddings
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                # Normalize
                embeddings = F.normalize(embeddings, p=2, dim=1)
                # Convert to Python list of lists
                embeddings_list = embeddings.cpu().numpy().tolist()
            
            return embeddings_list

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def insert_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Insert documents with embeddings into MongoDB
        
        Args:
            documents: List of document chunks
            
        Returns:
            List of inserted document IDs
        """
        if not documents:
            return []
        
        # Extract texts for embedding generation
        texts = [doc["text"] for doc in documents]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.generate_embeddings(texts)
        
        # Prepare documents for insertion
        docs_to_insert = []
        for doc, embedding in zip(documents, embeddings):
            doc_with_embedding = doc.copy()
            doc_with_embedding["embedding"] = embedding
            docs_to_insert.append(doc_with_embedding)
        
        # Insert into MongoDB
        try:
            result = self.collection.insert_many(docs_to_insert)
            logger.info(f"Successfully inserted {len(result.inserted_ids)} documents")
            return [str(id) for id in result.inserted_ids]
        except Exception as e:
            logger.error(f"Error inserting documents: {str(e)}")
            raise

    def process_pdf(self, pdf_path: str) -> List[str]:
        """
        Complete pipeline to process a single PDF
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of inserted document IDs
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Step 1: Extract text from PDF
        extracted_data = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Chunk the text
        chunks = self.chunk_text(extracted_data["text"], extracted_data["metadata"])
        
        # Step 3: Insert chunks with embeddings
        inserted_ids = self.insert_documents(chunks)
        
        logger.info(f"Completed processing {pdf_path}. Inserted {len(inserted_ids)} chunks.")
        return inserted_ids

    def process_multiple_pdfs(self, pdf_directory: str) -> Dict[str, List[str]]:
        """
        Process multiple PDFs from a directory
        
        Args:
            pdf_directory: Directory containing PDF files
            
        Returns:
            Dictionary mapping PDF filenames to inserted document IDs
        """
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_directory}")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return {}
        
        results = {}
        for pdf_file in pdf_files:
            try:
                inserted_ids = self.process_pdf(str(pdf_file))
                results[pdf_file.name] = inserted_ids
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                results[pdf_file.name] = []
        
        return results

    def search_similar_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector search
        
        Args:
            query: Search query
            limit: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        # Generate embedding for query
        query_embedding = self.generate_embeddings([query])[0]
        
        # Perform vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "source_metadata": 1,
                    "chunk_id": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        try:
            results = list(self.collection.aggregate(pipeline))
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error performing vector search: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            stats = {
                "total_documents": self.collection.count_documents({}),
                "unique_sources": len(self.collection.distinct("source_metadata.filename")),
                "sample_document": self.collection.find_one({}, {"embedding": 0})
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}

# Usage Example
def main():
    # Configuration
    MONGODB_URI = os.getenv("MONGODB_URI")
    DATABASE_NAME = os.getenv("DATABASE_NAME") 
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")  
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
        
    # Initialize RAG Pipeline
    rag = RAGPipeline(
        mongodb_uri=MONGODB_URI,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME,
        embedding_model = EMBEDDING_MODEL
    )
    
    # Create vector index
    rag.create_vector_index()
    
    # Process single PDF
    pdf_path = "pdfs/document.pdf"
    inserted_ids = rag.process_pdf(pdf_path)
    print(f"Inserted {len(inserted_ids)} chunks")
        
    # Search for similar documents
    query = "What is Carbon Footprint?"
    similar_docs = rag.search_similar_documents(query, limit=3)
    for doc in similar_docs:
         print(f"Score: {doc['score']:.4f}")
         print(f"Source: {doc['source_metadata']['filename']}")
         print(f"Text: {doc['text'][:200]}...")
         print("-" * 50)
    
    # Get collection statistics
    stats = rag.get_collection_stats()
    print("Collection Statistics:", stats)

if __name__ == "__main__":
    main()