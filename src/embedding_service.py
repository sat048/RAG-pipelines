"""
Embedding Service for Smart Policy Assistant
Handles OpenAI embeddings and text processing
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using OpenAI"""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Set up OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        logger.info(f"Initialized EmbeddingService with model: {model}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embeddings
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = np.array(response.data[0].embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        all_embeddings = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts
                )
                
                # Extract embeddings
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                # Add zero embeddings for failed batch
                batch_embeddings = [np.zeros(1536) for _ in batch_texts]
                all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def embed_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Number of chunks to process in each batch
            
        Returns:
            Numpy array of chunk embeddings
        """
        texts = [chunk['text'] for chunk in chunks]
        return self.embed_texts(texts, batch_size)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from the model"""
        try:
            # Generate a test embedding to get dimension
            test_embedding = self.embed_text("test")
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {str(e)}")
            return 1536  # Default for text-embedding-3-small

class MockEmbeddingService:
    """Mock embedding service for testing without API calls"""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        logger.info(f"Initialized MockEmbeddingService with dimension: {dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate random embedding for testing"""
        # Use text hash for consistent random embeddings
        seed = hash(text) % 2**32
        np.random.seed(seed)
        return np.random.rand(self.dimension).astype('float32')
    
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate random embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return np.array(embeddings)
    
    def embed_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> np.ndarray:
        """Generate random embeddings for chunks"""
        texts = [chunk['text'] for chunk in chunks]
        return self.embed_texts(texts, batch_size)
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.dimension

def create_embedding_service(use_mock: bool = False) -> EmbeddingService:
    """
    Factory function to create embedding service
    
    Args:
        use_mock: If True, creates a mock service for testing
        
    Returns:
        EmbeddingService instance
    """
    if use_mock:
        return MockEmbeddingService()
    else:
        return EmbeddingService()

def main():
    """Test the embedding service"""
    # Test with mock service first
    print("Testing with MockEmbeddingService...")
    mock_service = MockEmbeddingService()
    
    test_texts = [
        "Company policies and procedures",
        "Remote work guidelines",
        "Health and safety protocols"
    ]
    
    # Test single embedding
    single_embedding = mock_service.embed_text(test_texts[0])
    print(f"Single embedding dimension: {single_embedding.shape}")
    
    # Test batch embeddings
    batch_embeddings = mock_service.embed_texts(test_texts)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    
    # Test with real service if API key is available
    if os.getenv("OPENAI_API_KEY"):
        print("\nTesting with real OpenAI API...")
        try:
            real_service = EmbeddingService()
            real_embedding = real_service.embed_text("Test text")
            print(f"Real embedding dimension: {real_embedding.shape}")
        except Exception as e:
            print(f"Error with real API: {str(e)}")
    else:
        print("\nSkipping real API test (no API key)")

if __name__ == "__main__":
    main()

