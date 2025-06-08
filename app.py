"""
Flask Web Application for Smart Policy Assistant
Provides a web interface for querying the RAG system
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize RAG pipeline
# Use mock embeddings if no API key is provided
use_mock = not bool(os.getenv("OPENAI_API_KEY"))
pipeline = RAGPipeline(use_mock_embeddings=use_mock)

@app.route('/')
def index():
    """Main page with search interface"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """API endpoint for document search"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', 5)
        min_similarity = data.get('min_similarity', 0.0)
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Search the RAG pipeline
        results = pipeline.search(query, k=k, min_similarity=min_similarity)
        
        return jsonify({
            'query': query,
            'results': results,
            'total_results': len(results)
        })
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/context', methods=['POST'])
def get_context():
    """API endpoint for getting context for a query"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', 3)
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Get context from RAG pipeline
        context = pipeline.get_context_for_query(query, k=k)
        
        return jsonify({
            'query': query,
            'context': context
        })
        
    except Exception as e:
        logger.error(f"Context error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """API endpoint for pipeline statistics"""
    try:
        stats = pipeline.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ingest', methods=['POST'])
def ingest_documents():
    """API endpoint for ingesting documents"""
    try:
        data = request.get_json() or {}
        force_rebuild = data.get('force_rebuild', False)
        
        result = pipeline.ingest_documents(force_rebuild=force_rebuild)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_database():
    """API endpoint for clearing the database"""
    try:
        pipeline.clear_database()
        return jsonify({'message': 'Database cleared successfully'})
    except Exception as e:
        logger.error(f"Clear database error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

def create_templates():
    """Create the HTML template if it doesn't exist"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    template_path = templates_dir / "index.html"
    
    if not template_path.exists():
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Policy Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .search-section {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #1e3c72;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .btn {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            margin-top: 30px;
        }
        
        .result-item {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border-left: 4px solid #1e3c72;
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .result-source {
            font-weight: 600;
            color: #1e3c72;
        }
        
        .result-score {
            background: #e3f2fd;
            color: #1976d2;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .result-text {
            color: #555;
            line-height: 1.6;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .stats {
            background: #e8f5e8;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .stats h3 {
            color: #2e7d32;
            margin-bottom: 10px;
        }
        
        .admin-section {
            background: #fff3cd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .admin-section h3 {
            color: #856404;
            margin-bottom: 15px;
        }
        
        .admin-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .btn-secondary {
            background: #6c757d;
        }
        
        .btn-danger {
            background: #dc3545;
        }
        
        @media (max-width: 768px) {
            .form-row {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 Smart Policy Assistant</h1>
            <p>Ask questions about your company policies and get instant answers</p>
        </div>
        
        <div class="content">
            <!-- Stats Section -->
            <div class="stats">
                <h3>System Status</h3>
                <div id="stats-content">Loading...</div>
            </div>
            
            <!-- Admin Section -->
            <div class="admin-section">
                <h3>Admin Controls</h3>
                <div class="admin-buttons">
                    <button class="btn btn-secondary" onclick="refreshStats()">Refresh Stats</button>
                    <button class="btn btn-secondary" onclick="ingestDocuments()">Ingest Documents</button>
                    <button class="btn btn-danger" onclick="clearDatabase()">Clear Database</button>
                </div>
            </div>
            
            <!-- Search Section -->
            <div class="search-section">
                <h2>Search Policies</h2>
                <form id="searchForm">
                    <div class="form-row">
                        <div class="form-group">
                            <label for="query">What would you like to know?</label>
                            <input type="text" id="query" name="query" placeholder="e.g., What's the remote work policy?" required>
                        </div>
                        <div class="form-group">
                            <label for="maxResults">Max Results</label>
                            <select id="maxResults" name="maxResults">
                                <option value="3">3 results</option>
                                <option value="5" selected>5 results</option>
                                <option value="10">10 results</option>
                            </select>
                        </div>
                    </div>
                    <button type="submit" class="btn" id="searchBtn">🔍 Search</button>
                </form>
            </div>
            
            <!-- Results Section -->
            <div id="results" class="results" style="display: none;"></div>
            
            <!-- Loading Section -->
            <div id="loading" class="loading" style="display: none;">
                <h3>Searching documents...</h3>
                <p>This may take a moment</p>
            </div>
        </div>
    </div>
    
    <script>
        // Load stats on page load
        window.onload = function() {
            refreshStats();
        };
        
        // Search form handling
        document.getElementById('searchForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const query = document.getElementById('query').value.trim();
            const maxResults = parseInt(document.getElementById('maxResults').value);
            
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('searchBtn').disabled = true;
            
            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        query: query,
                        k: maxResults
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResults(data);
                } else {
                    throw new Error(data.error || 'Search failed');
                }
                
            } catch (error) {
                console.error('Search error:', error);
                document.getElementById('results').innerHTML = `
                    <div class="error">
                        <strong>Search Error:</strong> ${error.message}
                    </div>
                `;
                document.getElementById('results').style.display = 'block';
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('searchBtn').disabled = false;
            }
        });
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            if (data.results.length === 0) {
                resultsDiv.innerHTML = `
                    <div class="error">
                        No relevant documents found for your query.
                    </div>
                `;
            } else {
                const resultsHtml = data.results.map((result, index) => `
                    <div class="result-item">
                        <div class="result-header">
                            <div class="result-source">${result.metadata.source_filename}</div>
                            <div class="result-score">Similarity: ${(result.similarity_score * 100).toFixed(1)}%</div>
                        </div>
                        <div class="result-text">${result.document}</div>
                    </div>
                `).join('');
                
                resultsDiv.innerHTML = `
                    <h3>Search Results for: "${data.query}" (${data.total_results} results)</h3>
                    ${resultsHtml}
                `;
            }
            
            resultsDiv.style.display = 'block';
        }
        
        async function refreshStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                const statsHtml = `
                    <div>
                        <strong>Total Documents:</strong> ${data.vector_store_stats.total_documents}<br>
                        <strong>Index Size:</strong> ${data.vector_store_stats.index_size}<br>
                        <strong>Embedding Model:</strong> ${data.embedding_model}<br>
                        <strong>Status:</strong> ${data.vector_store_stats.index_size > 0 ? 'Ready' : 'No documents indexed'}
                    </div>
                `;
                
                document.getElementById('stats-content').innerHTML = statsHtml;
            } catch (error) {
                console.error('Stats error:', error);
                document.getElementById('stats-content').innerHTML = 'Error loading stats';
            }
        }
        
        async function ingestDocuments() {
            if (!confirm('This will process all documents in the data/documents folder. Continue?')) {
                return;
            }
            
            try {
                const response = await fetch('/api/ingest', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        force_rebuild: true
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    alert(`Documents ingested successfully!\\n\\nTotal chunks: ${data.total_chunks}\\nTotal documents: ${data.total_documents}`);
                    refreshStats();
                } else {
                    throw new Error(data.error || 'Ingestion failed');
                }
            } catch (error) {
                console.error('Ingestion error:', error);
                alert('Error ingesting documents: ' + error.message);
            }
        }
        
        async function clearDatabase() {
            if (!confirm('This will clear all indexed documents. This action cannot be undone. Continue?')) {
                return;
            }
            
            try {
                const response = await fetch('/api/clear', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    alert('Database cleared successfully');
                    refreshStats();
                } else {
                    throw new Error(data.error || 'Clear failed');
                }
            } catch (error) {
                console.error('Clear error:', error);
                alert('Error clearing database: ' + error.message);
            }
        }
    </script>
</body>
</html>
        """
        
        with open(template_path, 'w') as f:
            f.write(html_content)
        
        logger.info("Created HTML template")

if __name__ == '__main__':
    # Create templates directory and HTML file
    from pathlib import Path
    create_templates()
    
    # Get configuration from environment
    host = os.getenv('APP_HOST', '0.0.0.0')
    port = int(os.getenv('APP_PORT', 5000))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting Smart Policy Assistant on {host}:{port}")
    logger.info(f"Using {'mock' if use_mock else 'OpenAI'} embeddings")
    
    app.run(host=host, port=port, debug=debug)

