"""
Simple Smart Policy Assistant - Working Version
"""

import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from pathlib import Path

from src.simple_document_processor import SimpleDocumentProcessor
from src.vector_store import VectorStore
from src.embedding_service import MockEmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
processor = SimpleDocumentProcessor(chunk_size=500, chunk_overlap=100)
vector_store = VectorStore("./data/vector_db", dimension=1536)
embedding_service = MockEmbeddingService(dimension=1536)

@app.route('/')
def index():
    """Main page"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Policy Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .search-box { width: 100%; padding: 15px; font-size: 16px; border: 2px solid #ddd; border-radius: 5px; margin-bottom: 20px; }
        .btn { background: #007bff; color: white; padding: 15px 30px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; margin-right: 10px; }
        .btn:hover { background: #0056b3; }
        .result { background: #f8f9fa; padding: 20px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
        .result-header { font-weight: bold; color: #333; margin-bottom: 10px; }
        .result-text { color: #666; line-height: 1.6; }
        .stats { background: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 Smart Policy Assistant</h1>
            <p>Ask questions about your company policies</p>
        </div>
        
        <div class="stats">
            <strong>System Status:</strong> <span id="status">Loading...</span>
        </div>
        
        <div>
            <button class="btn" onclick="ingestDocs()">Ingest Documents</button>
            <button class="btn" onclick="clearDb()">Clear Database</button>
        </div>
        
        <div style="margin-top: 30px;">
            <input type="text" class="search-box" id="query" placeholder="Ask a question about policies..." onkeypress="handleKeyPress(event)">
            <button class="btn" onclick="search()">Search</button>
        </div>
        
        <div id="results"></div>
    </div>

    <script>
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                search();
            }
        }
        
        async function search() {
            const query = document.getElementById('query').value.trim();
            if (!query) {
                alert('Please enter a question');
                return;
            }
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<div class="result">Searching...</div>';
            
            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, k: 3 })
                });
                
                const data = await response.json();
                
                if (data.results && data.results.length > 0) {
                    let html = '<h3>Search Results:</h3>';
                    data.results.forEach((result, i) => {
                        html += `
                            <div class="result">
                                <div class="result-header">Document ${i+1}: ${result.metadata.source_filename} (Score: ${(result.similarity_score * 100).toFixed(1)}%)</div>
                                <div class="result-text">${result.document}</div>
                            </div>
                        `;
                    });
                    resultsDiv.innerHTML = html;
                } else {
                    resultsDiv.innerHTML = '<div class="result">No relevant documents found.</div>';
                }
            } catch (error) {
                resultsDiv.innerHTML = '<div class="result">Error: ' + error.message + '</div>';
            }
        }
        
        async function ingestDocs() {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = 'Ingesting documents...';
            
            try {
                const response = await fetch('/api/ingest', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    statusDiv.textContent = `Ready - ${data.total_chunks} chunks, ${data.total_documents} documents`;
                } else {
                    statusDiv.textContent = 'Error: ' + (data.error || 'Unknown error');
                }
            } catch (error) {
                statusDiv.textContent = 'Error: ' + error.message;
            }
        }
        
        async function clearDb() {
            if (confirm('Clear database?')) {
                try {
                    await fetch('/api/clear', { method: 'POST' });
                    document.getElementById('status').textContent = 'Database cleared';
                    document.getElementById('results').innerHTML = '';
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
        }
        
        // Load stats on page load
        window.onload = function() {
            document.getElementById('status').textContent = 'Ready to use';
        };
    </script>
</body>
</html>
    """

@app.route('/api/search', methods=['POST'])
def search():
    """Search API"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', 3)
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Generate query embedding
        query_embedding = embedding_service.embed_text(query)
        
        # Search vector store
        results = vector_store.search(query_embedding, k=k)
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = {
                'document': result['document'],
                'metadata': result['metadata'],
                'distance': float(result['distance']),
                'similarity_score': float(result['similarity_score']),
                'rank': int(result['rank'])
            }
            serializable_results.append(serializable_result)
        
        return jsonify({
            'query': query,
            'results': serializable_results,
            'total_results': len(serializable_results)
        })
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ingest', methods=['POST'])
def ingest_documents():
    """Ingest documents"""
    try:
        # Process documents
        chunks = processor.process_directory("./data/documents")
        
        if not chunks:
            return jsonify({'error': 'No documents found'}), 400
        
        # Generate embeddings
        embeddings = embedding_service.embed_chunks(chunks)
        
        # Store in vector database
        vector_store.rebuild_from_chunks(chunks, embeddings)
        
        return jsonify({
            'success': True,
            'total_chunks': len(chunks),
            'total_documents': len(set(chunk['source_filename'] for chunk in chunks))
        })
        
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_database():
    """Clear database"""
    try:
        vector_store.clear()
        return jsonify({'message': 'Database cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Simple Smart Policy Assistant on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
