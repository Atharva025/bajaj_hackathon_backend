# Intelligent Document Query System - RAG Pipeline

## 🚀 Project Overview

A sophisticated Retrieval-Augmented Generation (RAG) system designed for precise document analysis and question-answering. Built as a comprehensive full-stack solution with an optimized backend API and intuitive frontend interface, this system excels at extracting accurate information from complex policy documents and technical materials.

## 🏗️ Architecture

### Backend (bajaj_backend)
**Technology Stack:**
- **Framework:** FastAPI with Uvicorn server
- **Document Processing:** PyMuPDF for high-performance PDF parsing
- **Vector Search:** FAISS (Facebook AI Similarity Search) with Inner Product indexing
- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2) for semantic understanding
- **LLM Integration:** Mixtral-8x7B-Instruct via OpenRouter API
- **Caching:** Multi-layer caching system for documents and Q&A pairs

### Frontend (bajaj_hackathon)  
**Technology Stack:**
- **UI Framework:** Gradio with custom CSS styling
- **API Integration:** RESTful communication with backend services
- **User Experience:** Interactive interface with real-time query processing

## 🔧 Core Features

### Advanced RAG Pipeline
- **Intelligent Chunking:** Context-aware text segmentation with configurable overlap (400 words/chunk, 50-word overlap)
- **Semantic Retrieval:** L2-normalized vector embeddings with cosine similarity search
- **Hybrid Ranking:** Combines semantic similarity with lexical keyword matching
- **MMR Re-ranking:** Maximal Marginal Relevance algorithm for diverse, relevant results
- **Domain-Aware Processing:** Specialized handling for insurance/policy terminology

### Optimization Features
- **Concurrent Processing:** ThreadPoolExecutor for parallel question processing (6 workers)
- **Multi-level Caching:** Document embeddings and Q&A response caching with SHA1 hashing
- **Smart Filtering:** Similarity thresholds and penalty systems for generic responses
- **Streaming Support:** Async streaming responses for improved performance

### Document Intelligence
- **Section Detection:** Automatic heading recognition and context preservation
- **Neighbor Window Selection:** Context expansion for comprehensive answers
- **Fallback Mechanisms:** Lexical retrieval when semantic search fails
- **Answer Formatting:** Structured responses limited to 3 sentences for clarity

## 📊 Technical Implementation

### Backend API Endpoints
```
POST /api/v1/hackrx/run
```
**Request Format:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": ["Question 1", "Question 2"]
}
```

**Response Format:**
```json
{
  "answers": ["Answer 1", "Answer 2"]
}
```

### Performance Optimizations
- **Vector Indexing:** FAISS IndexFlatIP for normalized embeddings
- **Batch Processing:** Efficient embedding computation (64 batch size)
- **Memory Management:** In-memory index storage with disk persistence
- **Rate Limiting:** Configurable concurrency controls
- **Error Handling:** Comprehensive exception management with fallback strategies

### Security & Configuration
- **API Authentication:** Bearer token authentication
- **Environment Variables:** Secure configuration management
- **CORS Support:** Cross-origin resource sharing
- **GZip Compression:** Response compression for bandwidth optimization

## 🛠️ Key Dependencies

### Backend Dependencies
- `fastapi` - Modern web framework for APIs
- `sentence-transformers` - State-of-the-art embeddings
- `faiss-cpu` - High-performance vector search
- `pymupdf` - PDF processing and text extraction
- `langchain` - LLM integration framework
- `uvicorn` - ASGI server implementation

### Frontend Dependencies
- `gradio` - Machine learning web interfaces
- `requests` - HTTP client for API communication
- `python-dotenv` - Environment configuration

## 🎯 Performance Metrics

- **Retrieval Accuracy:** Top-K retrieval with MMR diversity (λ=0.8)
- **Processing Speed:** Concurrent question processing with 6-worker thread pool
- **Cache Efficiency:** SHA1-based caching for repeated queries
- **Response Quality:** 3-sentence limit with domain-specific optimizations
- **Similarity Threshold:** 0.35 minimum for relevance filtering

## 🔄 Workflow Process

1. **Document Ingestion:** PDF parsing and text extraction per page
2. **Chunking Strategy:** Sentence-aware segmentation with overlap
3. **Embedding Generation:** Batch embedding computation and normalization
4. **Index Creation:** FAISS index construction and persistence
5. **Query Processing:** Parallel question handling with caching
6. **Retrieval Pipeline:** Semantic search + lexical filtering + MMR ranking
7. **Answer Generation:** LLM prompting with curated context
8. **Response Formatting:** Structured output with source attribution

## 🌟 Advanced Features

### Domain-Specific Enhancements
- **Insurance Terminology:** Specialized synonym expansion for policy terms
- **Numerical Awareness:** Enhanced scoring for dates, percentages, and periods
- **Section Boosting:** Contextual relevance based on document structure
- **Plan-Specific Logic:** Targeted retrieval for specific policy plans

### User Experience
- **Custom Styling:** Professional dark theme with gradient headers
- **Example Queries:** Pre-loaded examples for user guidance
- **Real-time Processing:** Progress tracking and status updates
- **Error Handling:** Comprehensive error messaging and recovery

## 🚀 Deployment

Both applications are deployed on Hugging Face Spaces with:
- **Automatic Scaling:** Container-based deployment
- **Environment Management:** Secure secret handling
- **API Integration:** Cross-service communication
- **Monitoring:** Built-in logging and error tracking

## 📈 Technical Achievements

- **Modular Architecture:** Separation of concerns between frontend and backend
- **Scalable Design:** Concurrent processing and efficient caching
- **Production-Ready:** Comprehensive error handling and monitoring
- **Performance Optimization:** Multi-level caching and efficient algorithms
- **Domain Expertise:** Specialized handling for complex document types

This project demonstrates advanced skills in machine learning system architecture, API development, document processing, and user interface design, showcasing the ability to build production-ready AI applications with real-world applicability.

## 🔗 Repository Links

- **Backend Repository:** [bajaj_backend](https://huggingface.co/spaces/Atharva025/bajaj_backend)
- **Frontend Repository:** [bajaj_hackathon](https://huggingface.co/spaces/Atharva025/bajaj_hackathon)
- **Live Demo:** Available on Hugging Face Spaces
