"""
Document Processor for Smart Policy Assistant
Handles ingestion and chunking of various document formats
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes various document formats and creates text chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single document and return list of chunks with metadata
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        try:
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            # Extract text based on file type
            if file_extension == '.pdf':
                text = self._extract_pdf_text(file_path)
            elif file_extension in ['.docx', '.doc']:
                text = self._extract_docx_text(file_path)
            elif file_extension in ['.txt', '.md']:
                text = self._extract_text_file(file_path)
            elif file_extension in ['.html', '.htm']:
                text = self._extract_html_text(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
            
            if not text.strip():
                logger.warning(f"No text extracted from {file_path}")
                return []
            
            # Create chunks
            chunks = self._create_chunks(text, file_path)
            logger.info(f"Processed {file_path.name}: {len(chunks)} chunks created")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return []
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
        return text
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {str(e)}")
        return text
    
    def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {str(e)}")
            return ""
    
    def _extract_html_text(self, file_path: Path) -> str:
        """Extract text from HTML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                return soup.get_text()
        except Exception as e:
            logger.error(f"Error reading HTML file {file_path}: {str(e)}")
            return ""
    
    def _create_chunks(self, text: str, file_path: Path) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: The full text content
            file_path: Path to the source file
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # Tokenize the text
        tokens = self.encoding.encode(text)
        
        # Create overlapping chunks
        start = 0
        chunk_id = 0
        
        while start < len(tokens):
            # Get chunk tokens
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            
            # Decode tokens back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Skip empty chunks
            if chunk_text.strip():
                chunk_data = {
                    'id': f"{file_path.stem}_chunk_{chunk_id}",
                    'text': chunk_text.strip(),
                    'source_file': str(file_path),
                    'source_filename': file_path.name,
                    'chunk_index': chunk_id,
                    'token_count': len(chunk_tokens),
                    'start_token': start,
                    'end_token': end
                }
                chunks.append(chunk_data)
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= len(tokens):
                break
        
        return chunks
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all supported documents in a directory
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of all chunks from all processed documents
        """
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        all_chunks = []
        supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.htm'}
        
        # Process all supported files in directory
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                logger.info(f"Processing: {file_path}")
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

def main():
    """Test the document processor"""
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    
    # Test with sample documents if they exist
    test_dir = Path("./data/documents")
    if test_dir.exists():
        chunks = processor.process_directory(str(test_dir))
        print(f"Processed {len(chunks)} chunks")
        
        # Print first chunk as example
        if chunks:
            print("\nFirst chunk:")
            print(f"ID: {chunks[0]['id']}")
            print(f"Source: {chunks[0]['source_filename']}")
            print(f"Text: {chunks[0]['text'][:200]}...")
    else:
        print("No documents found in ./data/documents")

if __name__ == "__main__":
    main()

