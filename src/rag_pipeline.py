"""
RAG Pipeline for Smart Policy Assistant
Main orchestration class that combines all components
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .embedding_service import EmbeddingService, create_embedding_service

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline that orchestrates document processing, embedding, and retrieval"""
    
    def __init__(
        self,
        documents_path: str = "./data/documents",
        vector_db_path: str = "./data/vector_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_mock_embeddings: bool = False
    ):
        self.documents_path = Path(documents_path)
        self.vector_db_path = Path(vector_db_path)
        
        # Initialize components
        self.document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = VectorStore(str(vector_db_path))
        self.embedding_service = create_embedding_service(use_mock=use_mock_embeddings)
        
        logger.info("RAG Pipeline initialized")
    
    def ingest_documents(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Ingest documents from the documents directory
        
        Args:
            force_rebuild: If True, rebuild the vector store even if it exists
            
        Returns:
            Dictionary with ingestion statistics
        """
        if not self.documents_path.exists():
            logger.error(f"Documents directory not found: {self.documents_path}")
            return {"error": "Documents directory not found"}
        
        # Check if vector store already exists and has data
        if not force_rebuild and self.vector_store.index and self.vector_store.index.ntotal > 0:
            logger.info("Vector store already exists. Use force_rebuild=True to rebuild.")
            return {
                "message": "Vector store already exists",
                "total_documents": len(self.vector_store.documents),
                "index_size": self.vector_store.index.ntotal
            }
        
        logger.info("Starting document ingestion...")
        
        # Process documents
        chunks = self.document_processor.process_directory(str(self.documents_path))
        
        if not chunks:
            logger.warning("No chunks created from documents")
            return {"error": "No chunks created from documents"}
        
        logger.info(f"Created {len(chunks)} chunks from documents")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_service.embed_chunks(chunks)
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Store in vector database
        logger.info("Storing in vector database...")
        self.vector_store.rebuild_from_chunks(chunks, embeddings)
        
        # Get statistics
        stats = self.vector_store.get_stats()
        
        logger.info("Document ingestion completed successfully")
        
        return {
            "success": True,
            "total_chunks": len(chunks),
            "total_documents": stats["total_documents"],
            "index_size": stats["index_size"],
            "embedding_dimension": stats["embedding_dimension"]
        }
    
    def search(self, query: str, k: int = 5, min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            k: Number of results to return
            min_similarity: Minimum similarity score threshold
            
        Returns:
            List of search results with documents and metadata
        """
        if not query.strip():
            return []
        
        logger.info(f"Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k)
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results 
            if result['similarity_score'] >= min_similarity
        ]
        
        logger.info(f"Found {len(filtered_results)} relevant results")
        
        return filtered_results
    
    def get_context_for_query(self, query: str, k: int = 3) -> str:
        """
        Get relevant context documents for a query
        
        Args:
            query: Search query
            k: Number of context documents to retrieve
            
        Returns:
            Combined context string
        """
        results = self.search(query, k=k)
        
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"Document {i} (Source: {result['metadata']['source_filename']}):\n"
                f"{result['document']}\n"
                f"Similarity Score: {result['similarity_score']:.3f}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        vector_stats = self.vector_store.get_stats()
        
        return {
            "documents_path": str(self.documents_path),
            "vector_db_path": str(self.vector_db_path),
            "vector_store_stats": vector_stats,
            "embedding_model": self.embedding_service.model if hasattr(self.embedding_service, 'model') else "mock",
            "chunk_size": self.document_processor.chunk_size,
            "chunk_overlap": self.document_processor.chunk_overlap
        }
    
    def clear_database(self):
        """Clear the vector database"""
        self.vector_store.clear()
        logger.info("Vector database cleared")

def main():
    """Test the RAG pipeline"""
    # Create pipeline with mock embeddings for testing
    pipeline = RAGPipeline(use_mock_embeddings=True)
    
    # Test ingestion (will only work if documents exist)
    print("Testing RAG Pipeline...")
    
    # Check if documents exist
    docs_path = Path("./data/documents")
    if docs_path.exists() and any(docs_path.iterdir()):
        print("Documents found, testing ingestion...")
        result = pipeline.ingest_documents()
        print(f"Ingestion result: {result}")
        
        # Test search
        test_queries = [
            "remote work policy",
            "health and safety",
            "company benefits"
        ]
        
        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            results = pipeline.search(query, k=2)
            print(f"Found {len(results)} results")
            for result in results:
                print(f"- {result['metadata']['source_filename']}: {result['similarity_score']:.3f}")
    else:
        print("No documents found in ./data/documents")
        print("Creating sample documents for testing...")
        
        # Create sample documents
        docs_path.mkdir(parents=True, exist_ok=True)
        
        sample_docs = {
            "remote_work_policy.txt": """
            Remote Work Policy
            
            Employees may work remotely up to 3 days per week with manager approval.
            Remote workers must maintain regular communication with their team.
            All remote work must be conducted in a secure, distraction-free environment.
            Equipment and internet costs for remote work are reimbursable up to $500 per year.
            """,
            
            "health_safety.txt": """
            Health and Safety Guidelines
            
            All employees must complete safety training within 30 days of hire.
            Personal protective equipment must be worn in designated areas.
            Report all incidents immediately to the safety officer.
            Emergency procedures are posted throughout the building.
            """,
            
            "benefits_package.txt": """
            Employee Benefits Package
            
            Health insurance coverage begins on the first day of employment.
            Dental and vision insurance are available at employee cost.
            401(k) matching up to 6% of salary after 90 days.
            Paid time off accrues at 1.25 days per month.
            """
        }
        
        for filename, content in sample_docs.items():
            with open(docs_path / filename, 'w') as f:
                f.write(content)
        
        print("Sample documents created. Testing ingestion...")
        result = pipeline.ingest_documents()
        print(f"Ingestion result: {result}")
        
        # Test search with sample data
        test_query = "remote work benefits"
        results = pipeline.search(test_query, k=2)
        print(f"\nSearch results for '{test_query}':")
        for result in results:
            print(f"- {result['metadata']['source_filename']}: {result['similarity_score']:.3f}")
            print(f"  Text: {result['document'][:100]}...")

if __name__ == "__main__":
    main()

