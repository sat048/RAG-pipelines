# 📚 Smart Policy Assistant

Currently experimenting with a lightweight RAG (Retrieval-Augmented Generation) system that helps employees quickly find answers to questions about company policies, HR guidelines, and procedures. Work in progress!

## 🚀 Features

- **Document Ingestion**: Supports PDF, DOCX, TXT, HTML, and Markdown files
- **Intelligent Search**: Semantic search using OpenAI embeddings
- **Vector Database**: FAISS-based storage for fast similarity search
- **Web Interface**: Clean, responsive web UI for querying
- **Real-time Processing**: Live document ingestion and indexing
- **Admin Controls**: Easy document management and system monitoring

## 🏗️ Architecture

```
Smart Policy Assistant
├── Document Processor     # Ingests and chunks documents
├── Embedding Service      # Generates text embeddings (OpenAI)
├── Vector Store          # FAISS-based similarity search
├── RAG Pipeline          # Orchestrates retrieval and generation
└── Web Interface         # Flask-based API and UI
```

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key (optional - mock embeddings available for testing)

## 🛠️ Installation

1. **Clone and Setup**
   ```bash
   cd smart-policy-assistant
   python3 -m virtualenv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   ```bash
   cp env.example .env
   # Edit .env and add your OpenAI API key (optional)
   ```

3. **Add Your Documents**
   ```bash
   # Place your policy documents in data/documents/
   # Supported formats: PDF, DOCX, TXT, HTML, MD
   ```

## 🚀 Quick Start

1. **Run the Application**
   ```bash
   source venv/bin/activate
   python app.py
   ```

2. **Open Your Browser**
   ```
   http://localhost:5000
   ```

3. **Ingest Documents**
   - Click "Ingest Documents" in the admin section
   - Wait for processing to complete

4. **Start Searching**
   - Ask questions like:
     - "What's the remote work policy?"
     - "How much vacation time do I get?"
     - "What are the safety requirements?"

## 📖 Usage Examples

### Web Interface
- **Search**: Type natural language questions
- **Filter Results**: Adjust number of results and similarity thresholds
- **View Sources**: See which documents contain relevant information

### API Endpoints

```python
# Search for documents
POST /api/search
{
    "query": "remote work policy",
    "k": 5,
    "min_similarity": 0.0
}

# Get context for a query
POST /api/context
{
    "query": "health insurance coverage",
    "k": 3
}

# Ingest new documents
POST /api/ingest
{
    "force_rebuild": false
}

# Get system statistics
GET /api/stats
```

## 🔧 Configuration

### Environment Variables
```env
# OpenAI API (optional - uses mock embeddings if not provided)
OPENAI_API_KEY=your_openai_api_key_here

# Application Settings
APP_HOST=0.0.0.0
APP_PORT=5000
DEBUG=True

# Vector Database Settings
VECTOR_DB_PATH=./data/vector_db
DOCUMENTS_PATH=./data/documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Document Processing
- **Chunk Size**: 1000 tokens (adjustable)
- **Chunk Overlap**: 200 tokens (adjustable)
- **Supported Formats**: PDF, DOCX, TXT, HTML, MD

## 📁 Project Structure

```
smart-policy-assistant/
├── src/
│   ├── document_processor.py    # Document ingestion and chunking
│   ├── vector_store.py          # FAISS vector database
│   ├── embedding_service.py     # OpenAI embeddings
│   └── rag_pipeline.py          # Main RAG orchestration
├── data/
│   ├── documents/               # Place your policy docs here
│   └── vector_db/              # FAISS index storage
├── templates/
│   └── index.html              # Web interface
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🧪 Testing

### Run with Mock Embeddings (No API Key Required)
```bash
python app.py
# The app will automatically use mock embeddings
```

### Test Individual Components
```bash
# Test documents processing
python src/document_processor.py

# Test vector store
python src/vector_store.py

# Test embedding service
python src/embedding_service.py

# Test RAG pipeline
python src/rag_pipeline.py
```

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker (create Dockerfile)
docker build -t smart-policy-assistant .
docker run -p 5000:5000 smart-policy-assistant
```

## 💡 Commercial Applications

This RAG system is perfect for:

- **HR Departments**: Employee policy queries
- **Legal Teams**: Contract and compliance document search
- **Customer Support**: Knowledge base search
- **Training**: Onboarding and procedure lookup
- **Compliance**: Regulatory document retrieval

## 🔒 Security Considerations

- Store API keys securely (use environment variables)
- Implement authentication for production use
- Validate and sanitize user inputs
- Consider data encryption for sensitive documents
- Regular security updates for dependencies

## 🛠️ Customization

### Adding New Document Types
1. Extend `DocumentProcessor` class
2. Add new file type handler
3. Update supported extensions list

### Changing Embedding Models
1. Modify `EmbeddingService` class
2. Update model parameters
3. Adjust vector dimensions in `VectorStore`

### Customizing the UI
1. Edit `templates/index.html`
2. Modify CSS styling
3. Add new API endpoints in `app.py`
