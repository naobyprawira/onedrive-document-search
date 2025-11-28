# OneDrive Document Search System

A comprehensive document search system that syncs PDF files from OneDrive, performs OCR extraction, generates AI-powered summaries, and enables semantic search with a user-friendly Streamlit interface.

## 🌟 Features

- **Automatic OneDrive Sync** - Scheduled synchronization of PDF documents from OneDrive/SharePoint
- **OCR Processing** - Advanced text extraction from PDF files via external OCR service
- **AI Summarization** - Automatic document summaries using Google Gemini or OpenRouter
- **Hybrid Search** - Combines semantic (vector) and keyword (BM25) search for better results
- **Hierarchical Storage** - Document and chunk-level vectors for precise retrieval
- **Web Interface** - Clean Streamlit UI for searching and browsing documents
- **Real-time Updates** - Detects new and modified files automatically

## 🏗️ Architecture

The system consists of five main components:

```
                    ┌──────────────┐
                    │  OCR Service │ ← External PDF Text Extraction
                    └──────▲───────┘
                           │
┌─────────────────┐        │
│  Streamlit App  │ ← User Interface (Port 8501)
└────────┬────────┘        │
         │                 │
┌────────▼────────┐        │
│ Search Service  │ ← FastAPI Search API (Port 8000)
└────────┬────────┘        │
         │                 │
┌────────▼─────────┐       │
│     Qdrant       │ ← Vector Database (Port 6333)
└──────────────────┘       │
         ▲                 │
         │                 │
┌────────┴─────────┐       │
│ Ingestion Service│ ──────┘ OneDrive Sync & Processing (Port 8001)
└──────────────────┘
```

### Services

1. **OCR Service** (External) - Extracts text from PDF files page by page
2. **Ingestion Service** - Syncs files from OneDrive, processes PDFs with OCR, generates embeddings and summaries, stores in Qdrant
3. **Search Service** - Provides search API with hybrid dense + sparse vector search
4. **Streamlit App** - Web UI for searching documents and viewing results
5. **Qdrant** - Vector database storing document and chunk embeddings with BM25 sparse vectors

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))
- Azure AD app with Microsoft Graph API permissions (Files.Read.All)
- OCR service running (see setup below)

### Setup

1. **Set up OCR Service (Required)**
   
   The document search system requires a separate OCR service for PDF text extraction.
   
   ```bash
   # Clone the OCR service repository
   git clone https://github.com/naobyprawira/ocr-service.git
   cd ocr-service
   
   # Follow the setup instructions in the OCR service README
   # Start the OCR service and note the endpoint URL
   ```
   
   Once running, you'll need the OCR service endpoint URL (e.g., `https://your-ocr-service.ngrok-free.app/ocr`)

2. **Clone this repository**
   ```bash
   git clone https://github.com/naobyprawira/OneDrive-Document-Search.git
   cd OneDrive-Document-Search
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials and OCR service URL
   ```

4. **Start the services**
   ```bash
   docker-compose up -d
   ```

5. **Access the application**
   - Streamlit UI: http://localhost:8501
   - Search API: http://localhost:8012
   - Ingestion API: http://localhost:8011
   - Qdrant UI: http://localhost:6333/dashboard

## 📝 Configuration

Key environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `MS_TENANT_ID` | Azure AD tenant ID | Required |
| `MS_CLIENT_ID` | Azure AD client ID | Required |
| `MS_CLIENT_SECRET` | Azure AD client secret | Required |
| `ONEDRIVE_DRIVE_ID` | OneDrive drive ID | Required |
| `ONEDRIVE_ROOT_PATH` | Folder path to sync | `AI/Document Filing` |
| `OCR_SERVICE_URL` | OCR endpoint URL | Required |
| `DOC_COLLECTION` | Qdrant documents collection name | `documents` |
| `CHUNK_COLLECTION` | Qdrant chunks collection name | `chunks` |
| `EMBED_DIM` | Embedding dimension | `3072` |
| `CHUNK_SIZE` | Text chunk size (chars) | `2000` |
| `SCHEDULE_CRON` | Sync schedule | `0 0 * * *` (daily) |

See `.env.example` for complete list of configuration options.

## 🔧 Usage

### Manual Ingestion

Trigger immediate ingestion via API:

```bash
curl -X POST http://localhost:8011/admin/ingest-now
```

### Search via API

```bash
curl "http://localhost:8012/search?query=laporan+keuangan&top_k=5"
```

### Search via UI

1. Open http://localhost:8501
2. Enter your search query
3. Adjust parameters in sidebar (number of results, chunk candidates)
4. Click "Search" or press Enter
5. View results with summaries and links to OneDrive

## 📂 Project Structure

```
OneDrive-Document-Search/
├── ingestion_service/     # OneDrive sync and document processing
│   ├── main.py           # FastAPI app with scheduler
│   ├── pipeline.py       # Document processing pipeline
│   ├── graph.py          # Microsoft Graph API integration
│   ├── ocr.py            # OCR service integration
│   ├── embeddings.py     # Gemini embedding & summarization
│   ├── storage.py        # Qdrant operations
│   └── config.py         # Configuration
├── search_service/        # Search API
│   └── main.py           # FastAPI search endpoint
├── streamlit_app/         # Web UI
│   └── app.py            # Streamlit interface
├── docker-compose.yml     # Service orchestration
├── .env.example          # Environment template
└── README.md             # This file
```

## 🔍 How It Works

### Ingestion Pipeline

1. **Discovery** - Lists all PDF files from OneDrive recursively
2. **Change Detection** - Compares with local inventory, identifies new/modified files
3. **Download** - Downloads files to temp directory
4. **OCR** - Extracts text from PDF pages in parallel
5. **Summarization** - Generates document summary using AI
6. **Chunking** - Splits text into overlapping chunks
7. **Embedding** - Creates dense vectors (semantic) and sparse vectors (BM25)
8. **Storage** - Stores documents and chunks with vectors in Qdrant

### Search Pipeline

1. **Query Processing** - User enters search query
2. **Dual Embedding** - Generates both semantic and BM25 vectors
3. **Hybrid Search** - Searches chunks using RRF (Reciprocal Rank Fusion)
4. **Deduplication** - Groups chunks by document, keeps best match per document
5. **Document Retrieval** - Fetches document metadata
6. **Ranking** - Sorts by relevance score
7. **Results** - Returns top K documents with summaries and snippets

## 🛠️ Development

### Rebuild a service

```bash
docker-compose up -d --build <service-name>
```

Example:
```bash
docker-compose up -d --build ingestion-service
```

### View logs

```bash
docker logs -f <container-name>
```

Example:
```bash
docker logs -f ingestion-service
```

### Stop services

```bash
docker-compose down
```

## 📊 Collections

The system uses two Qdrant collections (configurable via environment variables):

- **documents** (or custom name via `DOC_COLLECTION`) - Document-level vectors and metadata
- **chunks** (or custom name via `CHUNK_COLLECTION`) - Chunk-level vectors for fine-grained search

Both collections use:
- Dense vectors (3072-dim) for semantic search
- Sparse vectors (BM25) for keyword search

## 🐛 Troubleshooting

**Ingestion not processing files:**
- Check ingestion service logs: `docker logs -f ingestion-service`
- Verify OCR service is accessible
- Check OneDrive credentials and permissions

**Search returns no results:**
- Ensure ingestion has completed successfully
- Check Qdrant UI to verify documents exist
- Verify search service can connect to Qdrant

**"v_bm25 not found" error:**
- Collections need BM25 support (sparse vectors)
- Ensure collections are created with sparse vector configuration
- Check that `DOC_COLLECTION` and `CHUNK_COLLECTION` environment variables match your Qdrant collections
## Deployment Guide (Rocky Linux)

This guide details how to deploy the Document Search Stack from your local machine to a Rocky Linux server.

### Prerequisites

1.  **Server Access**: SSH access to your Rocky Linux server.
2.  **Docker & Docker Compose**: Installed on the server.
    ```bash
    # Install Docker
    sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    sudo dnf install docker-ce docker-ce-cli containerd.io
    sudo systemctl start docker
    sudo systemctl enable docker

    # Install Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.29.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    ```
3.  **Git**: Installed on the server (`sudo dnf install git`).

### Deployment Steps

1.  **Clone the Repository**
    On your server, clone the project repository:
    ```bash
    git clone <your-repo-url> document_search_stack
    cd document_search_stack
    ```
    *Alternatively, you can copy your local files using `scp`:*
    ```bash
    scp -r /path/to/local/project user@server_ip:/path/to/destination
    ```

2.  **Configure Environment Variables**
    Create a `.env` file in the project root (you can copy `.env.example`):
    ```bash
    cp .env.example .env
    nano .env
    ```
    **Crucial**: Update the following variables in `.env`:
    *   `QDRANT_API_KEY`: Set to your Qdrant API key (e.g., `iniadalahapikeyqdrant`).
    *   `QDRANT_HOST`: Set to `qdrant` (if using the external network setup below).
    *   `GEMINI_API_KEY`: Your Google Gemini API key.
    *   `MS_TENANT_ID`, `MS_CLIENT_ID`, `MS_CLIENT_SECRET`: Microsoft Graph credentials.
    *   `ONEDRIVE_DRIVE_ID`: Target OneDrive Drive ID.

3.  **Setup External Network**
    If you are connecting to an existing Qdrant instance running in another Docker stack (as per your configuration), ensure the external network exists.
    
    Check existing networks:
    ```bash
    docker network ls
    ```
    
    If the network `my_services_my_network` does not exist (and your other stack hasn't created it yet), create it:
    ```bash
    docker network create my_services_my_network
    ```
    *Note: Ensure your existing Qdrant container is attached to this network.*

4.  **Build and Run**
    Start the services:
    ```bash
    docker-compose up -d --build
    ```

5.  **Verify Deployment**
    *   **Streamlit UI**: Access `http://<server_ip>:8501`.
    *   **Logs**: Check logs in the `./logs` directory on the server.
        ```bash
        tail -f logs/ingestion.log
        tail -f logs/search.log
        ```

### Troubleshooting
*   **Network Issues**: If services can't connect to Qdrant, ensure both stacks are on `my_services_my_network` and `QDRANT_HOST` matches the container name of your Qdrant instance.
*   **Permissions**: Ensure the `./logs` directory is writable.
