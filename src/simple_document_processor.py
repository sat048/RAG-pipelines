"""
Simplified Document Processor - No tiktoken dependency
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class SimpleDocumentProcessor:
    """Simple document processor without tiktoken"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single document"""
        try:
            file_path = Path(file_path)
            
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                return []
            
            # Create simple chunks by splitting on sentences
            sentences = text.split('. ')
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) < self.chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Convert to chunk format
            result = []
            for i, chunk_text in enumerate(chunks):
                result.append({
                    'id': f"{file_path.stem}_chunk_{i}",
                    'text': chunk_text,
                    'source_file': str(file_path),
                    'source_filename': file_path.name,
                    'chunk_index': i,
                    'token_count': len(chunk_text.split()),
                    'start_token': 0,
                    'end_token': len(chunk_text.split())
                })
            
            logger.info(f"Processed {file_path.name}: {len(result)} chunks created")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return []
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Process all txt files in directory"""
        directory = Path(directory_path)
        if not directory.exists():
            return []
        
        all_chunks = []
        
        for file_path in directory.glob('*.txt'):
            logger.info(f"Processing: {file_path}")
            chunks = self.process_document(file_path)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

def main():
    """Test the simple document processor"""
    processor = SimpleDocumentProcessor()
    
    test_dir = Path("./data/documents")
    if test_dir.exists():
        chunks = processor.process_directory(str(test_dir))
        print(f"Processed {len(chunks)} chunks")
        
        if chunks:
            print("\nFirst chunk:")
            print(f"ID: {chunks[0]['id']}")
            print(f"Source: {chunks[0]['source_filename']}")
            print(f"Text: {chunks[0]['text'][:200]}...")

if __name__ == "__main__":
    main()
