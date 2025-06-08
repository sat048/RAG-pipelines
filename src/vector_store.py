"""
Vector Store for Smart Policy Assistant
Handles FAISS vector database operations
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import faiss
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self, vector_db_path: str = "./data/vector_db", dimension: int = 1536):
        self.vector_db_path = Path(vector_db_path)
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Create directory if it doesn't exist
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        index_path = self.vector_db_path / "faiss_index.bin"
        metadata_path = self.vector_db_path / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.metadata = data.get('metadata', [])
                
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Error loading existing index: {str(e)}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []
        logger.info("Created new FAISS index")
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of document texts
            embeddings: Numpy array of embeddings (shape: [n_docs, embedding_dim])
            metadata: List of metadata dictionaries
        """
        if len(documents) != len(embeddings) or len(documents) != len(metadata):
            raise ValueError("Documents, embeddings, and metadata must have the same length")
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of dictionaries containing document text, metadata, and similarity scores
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):  # Valid index
                result = {
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(distance),
                    'similarity_score': 1.0 / (1.0 + distance),  # Convert distance to similarity
                    'rank': i + 1
                }
                results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.dimension,
            'vector_db_path': str(self.vector_db_path)
        }
    
    def save(self):
        """Save the FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            index_path = self.vector_db_path / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = self.vector_db_path / "metadata.pkl"
            data = {
                'documents': self.documents,
                'metadata': self.metadata
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved vector store to {self.vector_db_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
    
    def clear(self):
        """Clear all documents from the vector store"""
        self._create_new_index()
        logger.info("Cleared vector store")
    
    def rebuild_from_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Rebuild the entire vector store from chunks
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Numpy array of chunk embeddings
        """
        # Clear existing data
        self.clear()
        
        # Prepare data
        documents = [chunk['text'] for chunk in chunks]
        metadata = [
            {
                'chunk_id': chunk['id'],
                'source_file': chunk['source_file'],
                'source_filename': chunk['source_filename'],
                'chunk_index': chunk['chunk_index'],
                'token_count': chunk['token_count']
            }
            for chunk in chunks
        ]
        
        # Add to vector store
        self.add_documents(documents, embeddings, metadata)
        
        # Save to disk
        self.save()
        
        logger.info(f"Rebuilt vector store with {len(chunks)} chunks")

def main():
    """Test the vector store"""
    # Create a simple test
    store = VectorStore()
    
    # Test data
    test_docs = [
        "This is a test document about company policies.",
        "Remote work policies and guidelines for employees.",
        "Health and safety protocols in the workplace."
    ]
    
    # Mock embeddings (random for testing)
    test_embeddings = np.random.rand(len(test_docs), 1536).astype('float32')
    
    test_metadata = [
        {'source': 'policy1.pdf', 'page': 1},
        {'source': 'policy2.pdf', 'page': 2},
        {'source': 'policy3.pdf', 'page': 3}
    ]
    
    # Add documents
    store.add_documents(test_docs, test_embeddings, test_metadata)
    
    # Search
    query_embedding = np.random.rand(1, 1536).astype('float32')
    results = store.search(query_embedding, k=2)
    
    print(f"Search results: {len(results)}")
    for result in results:
        print(f"- {result['metadata']['source']}: {result['similarity_score']:.3f}")
    
    # Save
    store.save()
    print("Vector store saved successfully")

if __name__ == "__main__":
    main()

