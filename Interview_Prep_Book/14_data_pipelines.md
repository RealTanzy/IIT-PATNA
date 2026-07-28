# Chapter 14: Data Pipelines & ETL

## 14.1 Document Ingestion

Every RAG system begins with the same challenge: turning heterogeneous documents into uniform text chunks. In production, you never get clean data — you get PDFs with embedded tables, Word documents with tracked changes, Excel files where column headers are on row 3, and HTML pages with navigation noise. The ingestion layer must handle all of this gracefully.

### PDF: PyMuPDF (fitz)

PyMuPDF is the fastest Python PDF library. It extracts text with layout preservation and handles embedded images:

```python
import fitz  # PyMuPDF

def extract_pdf(file_path: Path) -> list[dict]:
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")  # Plain text extraction
        tables = page.find_tables()   # Table detection (PyMuPDF 1.23+)
        images = page.get_images()    # Embedded image references
        pages.append({
            "text": text,
            "tables": [t.extract() for t in tables],
            "images": images,
            "page_num": page_num + 1,
        })
    return pages
```

**Key decisions:**
- Use `get_text("blocks")` for layout-aware extraction (preserves reading order)
- For scanned PDFs: detect low text content, route to OCR pipeline
- For images: extract and send to vision API for description

### DOCX: python-docx

Word documents have structure — paragraphs, headings, tables, and embedded media:

```python
from docx import Document

def extract_docx(file_path: Path) -> list[dict]:
    doc = Document(file_path)
    sections = []
    current_section = {"heading": "", "content": []}
    
    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            if current_section["content"]:
                sections.append(current_section)
            current_section = {"heading": para.text, "content": []}
        else:
            current_section["content"].append(para.text)
    
    # Extract tables separately
    for table in doc.tables:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        sections.append({"heading": "Table", "content": rows})
    
    return sections
```

### Excel: pandas + openpyxl

Excel files are deceptive — headers might not be on row 1, multiple sheets contain different data, and cells might contain formulas:

```python
import pandas as pd

def extract_excel(file_path: Path) -> list[dict]:
    xls = pd.ExcelFile(file_path)
    sheets = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # Auto-detect header row (first row with >50% non-null values)
        # Convert to natural language description
        text = f"Sheet: {sheet_name}\nColumns: {', '.join(df.columns)}\n"
        text += df.to_string(index=False, max_rows=100)
        sheets.append({"sheet": sheet_name, "text": text, "shape": df.shape})
    return sheets
```

### HTML: BeautifulSoup

Web content is noisy — navigation, footers, ads. Extract the signal:

```python
from bs4 import BeautifulSoup

def extract_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove noise
    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()
    # Extract tables as structured text
    tables = []
    for table in soup.find_all("table"):
        rows = [[td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                for tr in table.find_all("tr")]
        tables.append(rows)
        table.decompose()  # Remove from soup to avoid double-counting
    # Get remaining text
    text = soup.get_text(separator="\n", strip=True)
    return text, tables
```

### CSV: pandas with Detection

```python
def extract_csv(file_path: Path) -> str:
    # Detect encoding
    with open(file_path, "rb") as f:
        encoding = chardet.detect(f.read(10000))["encoding"]
    # Detect delimiter
    df = pd.read_csv(file_path, encoding=encoding, sep=None, engine="python")
    return df.to_string(index=False)
```

### Images: Vision API

For images embedded in documents or uploaded directly:

```python
async def describe_image(image_bytes: bytes) -> str:
    response = await llm.invoke(
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "data": b64encode(image_bytes)}},
                {"type": "text", "text": "Describe this image in detail for search indexing."}
            ]
        }]
    )
    return response.content
```

---

## 14.2 Chunking Pipelines

Chunking is the most underrated component of RAG. Bad chunking causes retrieval failures no matter how good your embedding model is. The goal: create chunks that are self-contained, semantically coherent, and the right size for your embedding model's context window.

### The Pipeline

```
Raw Document
    │
    ▼
┌───────────────────────────┐
│ Step 1: Section Split     │ ← Split on headers (H1, H2, H3)
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│ Step 2: Size Check        │ ← If section > max_chunk_size (512 tokens)
└────────────┬──────────────┘   split by paragraphs
             │
             ▼
┌───────────────────────────┐
│ Step 3: Merge Short       │ ← Merge paragraphs until target_size (300 tokens)
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│ Step 4: Add Overlap       │ ← 50 token overlap between adjacent chunks
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│ Step 5: Inject Metadata   │ ← source, section_path, chunk_type, page_num
└───────────────────────────┘
```

### Implementation

```python
class ChunkingPipeline:
    def __init__(self, max_size=512, target_size=300, overlap=50):
        self.max_size = max_size
        self.target_size = target_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_document(self, doc: Document) -> list[Chunk]:
        # Step 1: Split by sections
        sections = self._split_sections(doc.text)
        
        chunks = []
        for section in sections:
            token_count = len(self.tokenizer.encode(section.text))
            
            if token_count <= self.max_size:
                # Section fits in one chunk
                chunks.append(section)
            else:
                # Step 2: Split by paragraphs
                paragraphs = section.text.split("\n\n")
                # Step 3: Merge short paragraphs
                merged = self._merge_paragraphs(paragraphs)
                chunks.extend(merged)
        
        # Step 4: Add overlap
        chunks = self._add_overlap(chunks)
        
        # Step 5: Inject metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                "source": doc.source,
                "section_path": chunk.section_path,
                "chunk_index": i,
                "chunk_type": self._classify_chunk(chunk.text),
            }
        
        return chunks
```

### Table-Aware Chunking

Tables require special handling — splitting a table mid-row destroys its meaning:

```python
def chunk_with_tables(self, text: str, tables: list) -> list[Chunk]:
    chunks = []
    for table in tables:
        # Each table becomes its own chunk with header row repeated
        header = table[0]
        for batch in batched(table[1:], 20):  # 20 rows per chunk
            chunk_text = format_table([header] + batch)
            chunks.append(Chunk(text=chunk_text, chunk_type="table"))
    return chunks
```

---

## 14.3 Embedding & Indexing

### Batch Embedding

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dimensions

def embed_chunks(chunks: list[Chunk], batch_size: int = 64) -> np.ndarray:
    texts = [c.text for c in chunks]
    # Batch encode — much faster than one-by-one
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # For cosine similarity via dot product
    )
    return embeddings  # Shape: (num_chunks, 384)
```

### Vector Store Indexing (LanceDB)

```python
import lancedb

db = lancedb.connect("./vector_store")

# Create table with schema
table = db.create_table("documents", data=[
    {"text": chunk.text, "vector": embedding, **chunk.metadata}
    for chunk, embedding in zip(chunks, embeddings)
])

# Create FTS index for BM25
table.create_fts_index("text")
```

### ChromaDB Alternative

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("documents")

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=embeddings.tolist(),
    documents=[c.text for c in chunks],
    metadatas=[c.metadata for c in chunks],
)
```

---

## 14.4 Delta Sync

### The Problem

Re-embedding an entire knowledge base on every update is wasteful:
- 10,000 documents x 384-dim embeddings = significant compute
- API-based embeddings cost money per token
- Users expect near-real-time updates

### The Solution: Content Hashing

```python
import hashlib

class DeltaSyncManager:
    def __init__(self, state_file: Path):
        self.state = self._load_state(state_file)  # {doc_id: hash}
    
    def get_changed_docs(self, documents: list[Document]) -> list[Document]:
        changed = []
        for doc in documents:
            content_hash = hashlib.sha256(doc.text.encode()).hexdigest()
            if doc.id not in self.state or self.state[doc.id] != content_hash:
                changed.append(doc)
                self.state[doc.id] = content_hash
        self._save_state()
        return changed
    
    def sync(self, documents: list[Document]):
        changed = self.get_changed_docs(documents)
        if not changed:
            logger.info("No changes detected, skipping re-embedding")
            return
        
        logger.info(f"Re-embedding {len(changed)}/{len(documents)} changed docs")
        # Delete old chunks for changed docs
        vector_store.delete(filter={"doc_id": {"$in": [d.id for d in changed]}})
        # Re-chunk and re-embed only changed docs
        new_chunks = [chunk for doc in changed for chunk in chunker.chunk(doc)]
        new_embeddings = embed_chunks(new_chunks)
        vector_store.add(new_chunks, new_embeddings)
```

---

## 14.5 Multi-Source Integration

### The Challenge

Real systems pull from multiple sources, each with different:
- Authentication (PAT, OAuth, Basic Auth, API keys)
- APIs (REST v2 vs v3, GraphQL, CQL)
- Pagination (offset-based, cursor-based, token-based)
- Rate limits (different thresholds per source)

### Normalization Layer

```python
class SourceConnector(ABC):
    @abstractmethod
    async def fetch_documents(self) -> AsyncIterator[RawDocument]: ...

class ConfluenceConnector(SourceConnector):
    async def fetch_documents(self):
        async for page in self._cql_search("space = SSV"):
            yield RawDocument(
                id=f"confluence:{page['id']}",
                text=page["body"]["storage"]["value"],
                source="confluence",
                last_modified=page["version"]["when"],
            )

class JiraConnector(SourceConnector):
    async def fetch_documents(self):
        async for issue in self._paginated_fetch("/rest/api/2/search"):
            yield RawDocument(
                id=f"jira:{issue['key']}",
                text=self._format_issue(issue),
                source="jira",
                last_modified=issue["fields"]["updated"],
            )
```

### Deduplication

When multiple sources contain the same content (e.g., a Jira ticket linked from Confluence):

```python
def deduplicate(chunks: list[Chunk]) -> list[Chunk]:
    seen_hashes = set()
    unique = []
    for chunk in chunks:
        h = hashlib.md5(chunk.text.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(chunk)
    return unique
```

---

## 14.6 Case Study: SSV RAG Ingestion

The SSV RAG system integrates three distinct sources into a unified knowledge base:

### Confluence

```python
class ConfluenceIngester:
    """CQL search + recursive child page crawl + image extraction."""
    
    async def ingest(self):
        # Search for all pages in SSV space
        pages = await self.cql_search('space = "SSV" AND type = "page"')
        
        for page in pages:
            # Recursively fetch child pages
            children = await self.get_children(page["id"])
            
            # Extract text (HTML → clean text)
            text = self.html_to_text(page["body"]["storage"]["value"])
            
            # Extract and describe images
            for img in page.get("images", []):
                img_bytes = await self.download_attachment(img["url"])
                description = await self.vision_api.describe(img_bytes)
                text += f"\n[Image: {description}]"
            
            yield RawDocument(id=page["id"], text=text)
```

### Jira SWF (REST v2)

```python
class JiraSWFIngester:
    """PAT authentication, REST API v2, offset pagination."""
    
    async def ingest(self):
        start_at = 0
        batch_size = 200
        while True:
            response = await self.client.get(
                "/rest/api/2/search",
                params={"jql": self.jql, "startAt": start_at, "maxResults": batch_size},
                headers={"Authorization": f"Bearer {self.pat}"},
            )
            issues = response["issues"]
            if not issues:
                break
            for issue in issues:
                yield self._format_issue(issue)
            start_at += batch_size
```

### Jira MCP (REST v3)

```python
class JiraMCPIngester:
    """Basic auth, REST API v3, nextPageToken pagination."""
    
    async def ingest(self):
        next_token = None
        while True:
            params = {"jql": self.jql, "maxResults": 100}
            if next_token:
                params["nextPageToken"] = next_token
            response = await self.client.get(
                "/rest/api/3/search",
                params=params,
                auth=(self.email, self.api_token),
            )
            for issue in response["issues"]:
                yield self._format_issue(issue)
            next_token = response.get("nextPageToken")
            if not next_token:
                break
```

### Orchestration

```python
class KBSyncManager:
    """Background daemon thread, 30-minute sync interval."""
    
    def __init__(self):
        self.ingesters = [ConfluenceIngester(), JiraSWFIngester(), JiraMCPIngester()]
        self.chunker = TableAwareChunker(max_size=512, overlap=50)
        self.vector_store = lancedb.connect("./kb")
    
    def start_daemon(self):
        thread = threading.Thread(target=self._sync_loop, daemon=True)
        thread.start()
    
    def _sync_loop(self):
        while True:
            self._full_sync()
            time.sleep(1800)  # 30 minutes
    
    def _full_sync(self):
        all_docs = []
        for ingester in self.ingesters:
            docs = asyncio.run(self._collect(ingester))
            all_docs.extend(docs)
        
        changed = self.delta_manager.get_changed_docs(all_docs)
        if changed:
            chunks = [c for doc in changed for c in self.chunker.chunk(doc)]
            chunks = deduplicate(chunks)
            embeddings = embed_chunks(chunks)
            self.vector_store.upsert(chunks, embeddings)
            logger.info(f"Synced {len(changed)} docs, {len(chunks)} chunks")
```

This architecture processes thousands of documents from three sources into a single unified vector store, with delta sync ensuring only changed content is re-embedded on each 30-minute cycle.
