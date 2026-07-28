# Chapter 9: Compliance & Quality Assessment with AI

> "A compliance audit is just a structured question: show me the evidence that you did what you said you would do."

---

## 9.1 The Problem: Manual Compliance Audits

### What Is ASPICE?

Automotive SPICE (Software Process Improvement and Capability dEtermination) is the de facto process maturity framework for automotive software development. Originally derived from ISO/IEC 15504, it defines a set of process areas that organizations must demonstrate capability in to achieve supplier qualification with OEMs like Mercedes-Benz, BMW, and Volkswagen.

ASPICE defines **process areas** across the engineering lifecycle:

| Process Area | ID | Focus |
|---|---|---|
| Software Requirements Analysis | SWE.1 | Are requirements complete, consistent, traceable? |
| Software Architectural Design | SWE.2 | Is the architecture documented and justified? |
| Software Detailed Design | SWE.3 | Are modules designed with interfaces defined? |
| Software Unit Construction | SWE.4 | Is code written per standards? |
| Software Unit Verification | SWE.5 | Are units tested with adequate coverage? |
| **Software Qualification Testing** | **SWE.6** | **Is system-level qualification testing adequate?** |

Each process area is assessed at capability levels (0-5), with most OEM requirements targeting Level 2 or Level 3.

### SWE.6: Software Qualification Testing

SWE.6 specifically addresses whether the software qualification testing process is:
- **Planned** (test strategy exists, test environment defined)
- **Executed** (tests run against qualification criteria)
- **Evidenced** (traceability from requirements to test cases to results)
- **Reviewed** (results analyzed, deviations handled)

An ASPICE SWE.6 assessment typically involves 16 checkpoints (Q1-Q16), each with 3-11 sub-points. Assessors must examine documentation, interview teams, and determine compliance status per checkpoint.

### The Pain of Manual Audits

**Time:** A manual SWE.6 assessment takes 2-3 full working days per team. An assessor must read through test strategies, test plans, test reports, traceability matrices, and defect logs. For an organization with 30+ teams, this means a single assessment cycle consumes months.

**Subjectivity:** Different assessors interpret the same evidence differently. One assessor might rate a test strategy as "compliant" while another marks it "partially compliant" because the risk-based approach is implied but not explicitly stated.

**Inconsistency:** Without a standardized rubric applied mechanically, the same team might receive different scores in successive audits despite no process changes. This erodes trust in the assessment process itself.

**Scale:** As automotive software complexity grows (modern vehicles run 100M+ lines of code across 100+ ECUs), the volume of documentation that must be assessed grows proportionally. Manual approaches simply cannot keep pace.

### The Opportunity

Large Language Models can reason about evidence quality. Given a checkpoint definition ("Does the test strategy define a risk-based approach to test prioritization?") and a set of documents (test strategy PDF, test plan DOCX, traceability matrix XLSX), an LLM can:

1. Parse and understand the documents
2. Identify relevant evidence for each checkpoint
3. Evaluate whether the evidence satisfies the requirement
4. Produce a structured assessment with citations

This is not replacing human judgment — it is providing a consistent, rapid first pass that surfaces gaps and evidence before human assessors review.

---

## 9.2 Document Understanding Pipeline

### The Challenge: Format Diversity

ASPICE evidence comes in every format imaginable:
- **PDF** — test reports, signed approvals, process descriptions
- **DOCX** — test strategies, test plans, review minutes
- **Excel** — traceability matrices, test case databases, defect logs
- **HTML** — tool-generated reports (e.g., from DOORS, Polarion)
- **CSV** — exported data from test management tools
- **Images** — architecture diagrams, test environment setups (embedded in documents)

A compliance assessment tool must handle all of these uniformly.

### The Adapter Pattern

The system uses a classic **Adapter Pattern** to normalize diverse inputs into a common representation:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class InputSource:
    """Represents any input document for assessment."""
    path: Path
    filename: str
    mime_type: str
    content_bytes: bytes

@dataclass
class ExtractedContent:
    """Normalized output from any adapter."""
    text: str
    tables: list[dict]          # structured table data
    images: list[bytes]         # extracted images for vision processing
    metadata: dict              # source-specific metadata
    sections: list[dict]        # hierarchical section structure

class AbstractAdapter(ABC):
    """Base class for all document adapters."""
    
    @abstractmethod
    def can_handle(self, source: InputSource) -> bool:
        """Return True if this adapter can process the given source."""
        ...
    
    @abstractmethod
    def extract(self, source: InputSource) -> ExtractedContent:
        """Extract structured content from the source document."""
        ...
```

### Concrete Adapters

Each format gets a dedicated adapter:

```python
class PDFAdapter(AbstractAdapter):
    """Extracts text, tables, and images from PDF documents using PyMuPDF."""
    
    def can_handle(self, source: InputSource) -> bool:
        return source.mime_type == "application/pdf" or source.filename.endswith(".pdf")
    
    def extract(self, source: InputSource) -> ExtractedContent:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=source.content_bytes, filetype="pdf")
        
        full_text = []
        images = []
        tables = []
        
        for page in doc:
            # Text extraction with layout preservation
            full_text.append(page.get_text("text"))
            
            # Image extraction for vision processing
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                images.append(base_image["image"])
            
            # Table detection via heuristics (grid lines + text positioning)
            page_tables = self._detect_tables(page)
            tables.extend(page_tables)
        
        return ExtractedContent(
            text="\n".join(full_text),
            tables=tables,
            images=images,
            metadata={"page_count": len(doc), "title": doc.metadata.get("title", "")},
            sections=self._parse_sections(full_text)
        )


class DOCXAdapter(AbstractAdapter):
    """Extracts content from Word documents using python-docx."""
    
    def can_handle(self, source: InputSource) -> bool:
        return source.filename.endswith((".docx", ".doc"))
    
    def extract(self, source: InputSource) -> ExtractedContent:
        from docx import Document
        from io import BytesIO
        
        doc = Document(BytesIO(source.content_bytes))
        
        sections = []
        current_section = {"heading": "Root", "level": 0, "content": []}
        
        for para in doc.paragraphs:
            if para.style.name.startswith("Heading"):
                level = int(para.style.name.split()[-1])
                sections.append(current_section)
                current_section = {"heading": para.text, "level": level, "content": []}
            else:
                current_section["content"].append(para.text)
        
        sections.append(current_section)
        
        # Extract tables
        tables = []
        for table in doc.tables:
            rows = []
            for row in table.rows:
                rows.append([cell.text for cell in row.cells])
            tables.append({"headers": rows[0] if rows else [], "rows": rows[1:]})
        
        return ExtractedContent(
            text="\n".join(p.text for p in doc.paragraphs),
            tables=tables,
            images=self._extract_images(doc),
            metadata={"author": doc.core_properties.author},
            sections=sections
        )


class ExcelAdapter(AbstractAdapter):
    """Extracts tabular data from Excel files using openpyxl."""
    
    def can_handle(self, source: InputSource) -> bool:
        return source.filename.endswith((".xlsx", ".xls"))
    
    def extract(self, source: InputSource) -> ExtractedContent:
        from openpyxl import load_workbook
        from io import BytesIO
        
        wb = load_workbook(BytesIO(source.content_bytes), read_only=True)
        tables = []
        
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = list(ws.iter_rows(values_only=True))
            if rows:
                tables.append({
                    "sheet": sheet_name,
                    "headers": [str(c) if c else "" for c in rows[0]],
                    "rows": [[str(c) if c else "" for c in row] for row in rows[1:]]
                })
        
        # Flatten to text for embedding
        text_parts = []
        for table in tables:
            text_parts.append(f"Sheet: {table['sheet']}")
            for row in table["rows"]:
                text_parts.append(" | ".join(row))
        
        return ExtractedContent(
            text="\n".join(text_parts),
            tables=tables,
            images=[],
            metadata={"sheet_count": len(wb.sheetnames)},
            sections=[]
        )
```

### The Registry Pattern

An **AdapterRegistry** routes each input to the correct adapter:

```python
class AdapterRegistry:
    """Routes InputSource to the correct adapter using first-match semantics."""
    
    def __init__(self):
        self._adapters: list[AbstractAdapter] = []
    
    def register(self, adapter: AbstractAdapter) -> None:
        self._adapters.append(adapter)
    
    def get_adapter(self, source: InputSource) -> AbstractAdapter:
        for adapter in self._adapters:
            if adapter.can_handle(source):
                return adapter
        raise UnsupportedFormatError(f"No adapter for: {source.filename}")
    
    def extract(self, source: InputSource) -> ExtractedContent:
        adapter = self.get_adapter(source)
        return adapter.extract(source)

# Initialization
registry = AdapterRegistry()
registry.register(PDFAdapter())
registry.register(DOCXAdapter())
registry.register(ExcelAdapter())
registry.register(HTMLAdapter())
registry.register(CSVAdapter())
registry.register(ImageAdapter())
```

### LLM-Enhanced Extraction

Raw extraction is often insufficient. A PDF table might have merged cells, irregular formatting, or implicit headers. The system sends raw extracted text to an LLM for intelligent structuring:

```python
def llm_enhanced_extraction(raw_text: str, source_context: str) -> str:
    """Use LLM to clean and structure poorly-formatted extracted text."""
    prompt = f"""You are a document parsing assistant. Below is raw text extracted 
from a {source_context}. Clean it up:
- Fix table alignments
- Identify implicit section headers
- Resolve abbreviations where obvious
- Preserve ALL information — do not summarize

Raw text:
{raw_text}

Structured output:"""
    
    return llm.invoke(prompt)
```

### Image Extraction via Claude Vision API

Embedded diagrams (test architecture, traceability views) contain evidence that text extraction misses entirely. The system extracts images and sends them to Claude's vision capabilities:

```python
def extract_evidence_from_image(image_bytes: bytes, context: str) -> str:
    """Use Claude Vision to extract textual evidence from diagrams."""
    response = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", 
                    "media_type": "image/png",
                    "data": base64.b64encode(image_bytes).decode()}},
                {"type": "text", "text": f"""Extract all textual information from 
this diagram. Context: this is from an ASPICE SWE.6 assessment document about 
{context}. List all labels, connections, and structured information visible."""}
            ]
        }]
    )
    return response.content[0].text
```

---

## 9.3 Semantic Chunking for Compliance

### Why Chunking Matters for Assessment

In a compliance assessment, each chunk represents a **potential piece of evidence**. When the system evaluates "Does the test strategy define entry and exit criteria?", it needs to retrieve the specific paragraph or table that contains entry/exit criteria — not the entire 40-page test strategy document.

Bad chunking means:
- Relevant evidence split across chunks (retrieval finds half the answer)
- Irrelevant content diluting relevant chunks (noise in context)
- Lost section hierarchy (cannot attribute evidence to specific document sections)

### Section-Aware Splitting

The chunker respects document structure. It splits at headings and preserves the full heading hierarchy as metadata:

```python
@dataclass
class Chunk:
    """A single chunk of text with metadata for provenance tracking."""
    text: str
    chunk_type: str              # "section", "table_row", "paragraph"
    source_name: str             # original document name
    section_path: list[str]      # e.g., ["Test Strategy", "Scope", "Entry Criteria"]
    char_start: int              # character offset in original document
    char_end: int
    metadata: dict               # additional context

class SectionAwareChunker:
    def __init__(self, max_chunk_size: int = 512, overlap_tokens: int = 50):
        self.max_chunk_size = max_chunk_size  # in tokens
        self.overlap_tokens = overlap_tokens
    
    def chunk(self, content: ExtractedContent, source_name: str) -> list[Chunk]:
        chunks = []
        
        for section in content.sections:
            section_text = "\n".join(section["content"])
            section_path = self._build_path(section)
            
            if self._token_count(section_text) <= self.max_chunk_size:
                # Section fits in one chunk
                chunks.append(Chunk(
                    text=section_text,
                    chunk_type="section",
                    source_name=source_name,
                    section_path=section_path,
                    char_start=section.get("start", 0),
                    char_end=section.get("end", 0),
                    metadata={"heading": section["heading"], "level": section["level"]}
                ))
            else:
                # Split at paragraph boundaries with overlap
                chunks.extend(self._split_with_overlap(
                    section_text, section_path, source_name
                ))
        
        return chunks
```

### Table-Aware Chunking

Tables are common in compliance documentation (traceability matrices, test case lists, defect logs). Each table row is treated as a separate chunk, with the header row prepended for context:

```python
def chunk_table(self, table: dict, source_name: str, section_path: list[str]) -> list[Chunk]:
    """Each table row becomes one chunk. Max size = 2x normal to preserve row integrity."""
    header_text = " | ".join(table["headers"])
    chunks = []
    
    for i, row in enumerate(table["rows"]):
        row_text = " | ".join(row)
        # Prepend header for context
        chunk_text = f"[Table Header: {header_text}]\n[Row {i+1}]: {row_text}"
        
        chunks.append(Chunk(
            text=chunk_text,
            chunk_type="table_row",
            source_name=source_name,
            section_path=section_path + [f"Row {i+1}"],
            char_start=0,
            char_end=0,
            metadata={"row_index": i, "header": table["headers"]}
        ))
    
    return chunks
```

**Why 2x max size for table rows?** Table rows often contain dense information (requirement ID + description + test case + status). Splitting a row mid-cell destroys its meaning. Allowing double the normal chunk size preserves row integrity while keeping chunks manageable.

### Overlap for Context Continuity

When a section exceeds `max_chunk_size` and must be split, the chunker applies 50 tokens of overlap between consecutive chunks. This ensures that a sentence at a chunk boundary is fully captured in at least one chunk:

```python
def _split_with_overlap(self, text: str, section_path: list, source_name: str) -> list[Chunk]:
    tokens = self._tokenize(text)
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + self.max_chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        
        chunks.append(Chunk(
            text=self._detokenize(chunk_tokens),
            chunk_type="section_fragment",
            source_name=source_name,
            section_path=section_path,
            char_start=start,
            char_end=end,
            metadata={"fragment_index": len(chunks)}
        ))
        
        # Advance with overlap
        start = end - self.overlap_tokens
    
    return chunks
```

### Metadata Preservation

Every chunk carries provenance metadata:
- **section_path:** Full breadcrumb trail (e.g., `["Test Strategy v2.1.pdf", "Section 4", "Risk-Based Approach", "Priority Matrix"]`)
- **source_name:** Original filename for citation in the report
- **chunk_type:** Enables type-specific retrieval (e.g., prefer `table_row` chunks when looking for traceability data)

---

## 9.4 RAG for Evidence Retrieval

### ChromaDB as Per-Session Vector Store

Each assessment session gets its own isolated ChromaDB collection:

```python
import chromadb
from chromadb.config import Settings

class SessionVectorStore:
    def __init__(self, session_id: str):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=f"./sessions/{session_id}/chroma",
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name=f"assessment_{session_id}",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: list[Chunk]):
        self.collection.add(
            documents=[c.text for c in chunks],
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            metadatas=[{
                "source_name": c.source_name,
                "section_path": " > ".join(c.section_path),
                "chunk_type": c.chunk_type
            } for c in chunks]
        )
    
    def query(self, query_text: str, top_k: int = 5, 
              filter_type: Optional[str] = None) -> list[dict]:
        where_filter = {"chunk_type": filter_type} if filter_type else None
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter
        )
        return [{
            "text": doc,
            "metadata": meta,
            "distance": dist
        } for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )]
```

### Embeddings: all-MiniLM-L6-v2

The system uses `all-MiniLM-L6-v2` for embedding generation:
- **384 dimensions** — compact, fast to store and query
- **Trained on 1B+ sentence pairs** — good general-purpose semantic understanding
- **5x faster than large models** — critical when embedding hundreds of chunks per session
- **Good enough for compliance** — domain-specific embeddings were tested but offered marginal improvement for the added complexity

### Retrieval Strategy: One Query Per Sub-Point

Rather than a single monolithic query per checkpoint, the system generates one targeted query per sub-point:

```python
class EvidenceRetriever:
    def __init__(self, vector_store: SessionVectorStore):
        self.store = vector_store
    
    def retrieve_for_sub_point(self, sub_point_description: str, 
                                top_k: int = 5) -> list[dict]:
        """Retrieve evidence chunks relevant to a specific sub-point."""
        # Generate a retrieval-optimized query from the sub-point description
        retrieval_query = self._optimize_query(sub_point_description)
        
        results = self.store.query(retrieval_query, top_k=top_k)
        
        # Filter out low-relevance results (distance > threshold)
        filtered = [r for r in results if r["distance"] < 0.75]
        
        return filtered
    
    def _optimize_query(self, description: str) -> str:
        """Rewrite sub-point description as a retrieval query."""
        # Example: "Entry criteria for test execution are defined" 
        # becomes: "entry criteria test execution conditions prerequisites"
        return description  # In practice, may use LLM-based query expansion
```

### Why Per-Session (Not Global)?

Each assessment evaluates a different team's documents. A global vector store would:
- Mix evidence from different teams (cross-contamination)
- Grow unboundedly (millions of chunks over time)
- Return irrelevant evidence from other assessments
- Create data isolation concerns (Team A's documents visible during Team B's assessment)

Per-session stores are created at assessment start and optionally persisted for audit trail purposes. They can be deleted after the assessment report is finalized.

---

## 9.5 Structured Evaluation with LLMs

### The Challenge: Structured Output

The LLM must not return free-form text. Each sub-point evaluation must be a structured JSON object with specific fields. Free-form answers are impossible to aggregate into a scorecard.

### Anti-Hallucination System Prompts

The single most important design decision: the LLM is explicitly instructed to never generate information not present in the retrieved evidence.

```python
ASSESSOR_SYSTEM_PROMPT = """You are an ASPICE SWE.6 compliance assessor. Your role is to 
evaluate whether provided evidence satisfies specific checkpoint requirements.

CRITICAL RULES:
1. You MUST NOT generate, infer, or assume any information not explicitly present 
   in the provided evidence chunks.
2. If the evidence does not clearly address a sub-point, mark it as "missing".
3. If the evidence partially addresses a sub-point, mark it as "partial" and explain 
   what is present and what is missing.
4. BLANK IS BETTER THAN WRONG. It is far better to say "no evidence found" than to 
   hallucinate evidence that does not exist.
5. Always cite the specific source document and section where you found evidence.
6. Your confidence score (0.0-1.0) reflects how certain you are about YOUR assessment, 
   not how good the evidence is.

You will receive:
- A sub-point description (what you are evaluating)
- Retrieved evidence chunks (what you are evaluating against)
- Evaluation guidance (criteria and examples for this specific sub-point)

Respond ONLY in the specified JSON format."""
```

### The "Blank > Wrong" Policy

In compliance assessment, a false positive (claiming evidence exists when it does not) is far more damaging than a false negative (missing evidence that exists). A false positive could lead a team to believe they are compliant when they are not, potentially affecting safety-critical software certification.

The system enforces this through:
1. **System prompt emphasis** (as shown above)
2. **Low confidence threshold** — if confidence < 0.6, status is automatically downgraded to "partial"
3. **Evidence citation requirement** — the LLM must quote the specific text; if it cannot quote, it cannot claim compliance

### Pydantic v2 Schema Enforcement

The output schema is enforced via Pydantic v2 models:

```python
from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum

class ComplianceStatus(str, Enum):
    FOUND = "found"
    PARTIAL = "partial"
    MISSING = "missing"

class SubPointEvaluation(BaseModel):
    """Evaluation result for a single sub-point."""
    sub_point_id: str = Field(description="e.g., 'Q1.a'")
    status: ComplianceStatus
    score: float = Field(ge=0.0, le=1.0, description="1.0=found, 0.5=partial, 0.0=missing")
    evidence_text: str = Field(description="Direct quote from evidence, or empty if missing")
    evidence_source: str = Field(description="Source document and section")
    reasoning: str = Field(description="Why this status was assigned")
    confidence: float = Field(ge=0.0, le=1.0, description="Assessor confidence in this evaluation")
    follow_up_question: str = Field(default="", description="Question to ask if partial/missing")

class CheckpointResult(BaseModel):
    """Aggregated result for a full checkpoint."""
    checkpoint_id: str
    checkpoint_title: str
    sub_point_evaluations: list[SubPointEvaluation]
    overall_status: Literal["Yes", "Partial", "No"]
    overall_score: float
    summary: str
```

### Confidence Scoring

Every evaluation carries a confidence score (0.0-1.0) that reflects the assessor's certainty:

- **0.9-1.0:** Clear, unambiguous evidence directly addressing the sub-point
- **0.7-0.9:** Evidence present but requires interpretation
- **0.5-0.7:** Tangentially related evidence; uncertain if it satisfies the requirement
- **< 0.5:** Essentially guessing; system auto-downgrades to "missing"

---

## 9.6 Hierarchical Scoring

### Three-Level Aggregation

The scoring system operates at three nested levels:

```
Overall Scorecard (completeness %)
  └── Checkpoint Level (Q1-Q16: Yes/Partial/No)
       └── Sub-Point Level (a, b, c, ...: found/partial/missing)
```

### Sub-Point Level

Each sub-point receives a numeric score:

| Status | Score | Meaning |
|--------|-------|---------|
| Found | 1.0 | Evidence clearly satisfies the requirement |
| Partial | 0.5 | Some evidence exists but incomplete |
| Missing | 0.0 | No evidence found |

### Checkpoint Level

A checkpoint's score is the arithmetic mean of its sub-point scores:

$$\text{CheckpointScore} = \frac{1}{N} \sum_{i=1}^{N} \text{SubPointScore}_i$$

The numeric score maps to a categorical status:

| Score Range | Status | Meaning |
|-------------|--------|---------|
| >= 0.80 | **Yes** | Compliant — evidence is comprehensive |
| >= 0.40 | **Partial** | Partially compliant — significant gaps exist |
| < 0.40 | **No** | Non-compliant — evidence is largely absent |

### Overall Scorecard

The overall completeness is the average across all checkpoints:

$$\text{Completeness} = \frac{1}{16} \sum_{q=1}^{16} \text{CheckpointScore}_q \times 100\%$$

```python
def compute_scorecard(checkpoint_results: list[CheckpointResult]) -> dict:
    """Compute the overall assessment scorecard."""
    total_score = 0.0
    status_counts = {"Yes": 0, "Partial": 0, "No": 0}
    
    for result in checkpoint_results:
        # Compute checkpoint score from sub-points
        sub_scores = [sp.score for sp in result.sub_point_evaluations]
        checkpoint_score = sum(sub_scores) / len(sub_scores) if sub_scores else 0.0
        
        # Apply thresholds
        if checkpoint_score >= 0.80:
            status = "Yes"
        elif checkpoint_score >= 0.40:
            status = "Partial"
        else:
            status = "No"
        
        status_counts[status] += 1
        total_score += checkpoint_score
    
    completeness = (total_score / len(checkpoint_results)) * 100
    
    return {
        "completeness_percentage": round(completeness, 1),
        "checkpoints_compliant": status_counts["Yes"],
        "checkpoints_partial": status_counts["Partial"],
        "checkpoints_non_compliant": status_counts["No"],
        "overall_readiness": "READY" if status_counts["No"] == 0 else "NOT READY"
    }
```

### Why These Thresholds?

The thresholds (0.80 / 0.40) were calibrated against historical manual assessments:
- **0.80:** In manual audits, assessors marked "Yes" when 4 out of 5 (or more) sub-points were clearly evidenced. 80% captures this.
- **0.40:** Below 40%, manual assessors consistently rated checkpoints as non-compliant. The 40-80% range captured cases where assessors debated between Partial and No.

---

## 9.7 Agent-to-Agent (A2A) Architecture

### Dual-Agent Design

SPOT CHECK V2 uses two specialized agents:

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentCoordinator                           │
│                    (Mediator Pattern)                         │
│                                                              │
│   ┌────────────────────┐     ┌────────────────────────┐    │
│   │  Ingestion Agent   │     │   Assessor Agent        │    │
│   │  (Claude Opus 4.5) │     │   (Claude Opus 4.6)     │    │
│   │                    │     │                          │    │
│   │  - Parse documents │     │  - Evaluate evidence    │    │
│   │  - Extract content │     │  - Apply rubric         │    │
│   │  - Chunk text      │     │  - Score sub-points     │    │
│   │  - Build vectors   │     │  - Generate reasoning   │    │
│   └────────────────────┘     └────────────────────────┘    │
│              │                           │                    │
│              └─────────┬─────────────────┘                   │
│                        │                                     │
│                        ▼                                     │
│              ┌──────────────────┐                            │
│              │  Shared State     │                            │
│              │  (ChromaDB +      │                            │
│              │   Session Store)  │                            │
│              └──────────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### Why Two Agents?

**Ingestion Agent (Claude Opus 4.5):** Handles the mechanical work of document parsing, content extraction, chunking, and vector store population. This is a large-context, patient task — processing dozens of documents, handling format quirks, and building a comprehensive evidence base. Opus 4.5's extended thinking and large context window make it ideal.

**Assessor Agent (Claude Opus 4.6):** Handles the evaluative work of analyzing evidence quality, applying ASPICE criteria, and producing structured judgments. This requires domain expertise, nuanced reasoning, and the ability to distinguish "adequate evidence" from "almost evidence." Opus 4.6's superior reasoning capabilities are critical here.

### AgentCoordinator as Mediator

The agents never communicate directly. All communication flows through the `AgentCoordinator`:

```python
class AgentCoordinator:
    """Mediates between Ingestion and Assessor agents.
    Implements the Mediator pattern — agents are decoupled."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_store = SessionStore(session_id)
        self.vector_store = SessionVectorStore(session_id)
        self.ingestion_agent = IngestionAgent(model="claude-opus-4-5-20250514")
        self.assessor_agent = AssessorAgent(model="claude-opus-4-20250514")
    
    async def run_assessment(self, documents: list[InputSource], 
                              checkpoints: list[Checkpoint]) -> AssessmentResult:
        # Phase 1: Ingestion
        ingestion_result = await self.ingestion_agent.process(
            documents=documents,
            vector_store=self.vector_store
        )
        
        # Store ingestion metadata in shared state
        self.session_store.set("ingestion_complete", True)
        self.session_store.set("chunk_count", ingestion_result.chunk_count)
        self.session_store.set("document_summaries", ingestion_result.summaries)
        
        # Phase 2: Assessment (uses the populated vector store)
        assessment_result = await self.assessor_agent.evaluate(
            checkpoints=checkpoints,
            retriever=EvidenceRetriever(self.vector_store),
            document_context=ingestion_result.summaries
        )
        
        return assessment_result
```

### Communication via Shared State

The coordinator passes data between agents through two mechanisms:
1. **ChromaDB vector store** — ingestion agent populates it; assessor agent queries it
2. **Session store** — key-value metadata (document summaries, chunk counts, ingestion status)

This avoids the complexity of message queues or direct RPC between agents.

### HTTP Protocol to GenAI Nexus

Both agents communicate with their respective LLM backends via the organization's GenAI Nexus gateway:

```python
class GenAINexusClient:
    """Client for the GenAI Nexus API gateway."""
    
    def __init__(self, base_url: str, bearer_token: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        }
    
    async def invoke(self, model: str, messages: list[dict], 
                     tools: list[dict] = None) -> dict:
        """POST /model/{model}/invoke"""
        payload = {
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.0  # Deterministic for compliance
        }
        if tools:
            payload["tools"] = tools
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/model/{model}/invoke",
                json=payload,
                headers=self.headers,
                timeout=120.0
            )
            response.raise_for_status()
            return response.json()
```

---

## 9.8 The Checkpoint Evaluator Pattern

### 16 Checkpoints with Plugin Architecture

Each of the 16 checkpoints (Q1-Q16) has its own evaluator class, auto-discovered via a decorator-based plugin system:

```python
# Registry for checkpoint evaluators
_EVALUATOR_REGISTRY: dict[int, type] = {}

def register(checkpoint_number: int):
    """Decorator to auto-register checkpoint evaluators."""
    def decorator(cls):
        _EVALUATOR_REGISTRY[checkpoint_number] = cls
        return cls
    return decorator

def get_all_evaluators() -> dict[int, "CheckpointEvaluator"]:
    """Return all registered evaluators, instantiated."""
    return {n: cls() for n, cls in sorted(_EVALUATOR_REGISTRY.items())}
```

### Base Evaluator Class

```python
class CheckpointEvaluator(ABC):
    """Base class for all checkpoint evaluators."""
    
    checkpoint_id: str           # e.g., "Q1"
    checkpoint_title: str        # e.g., "Test Strategy"
    sub_points: list[str]        # e.g., ["a", "b", "c", ..., "k"]
    
    @abstractmethod
    def get_evaluation_guidance(self, sub_point: str) -> str:
        """Return specific evaluation criteria for this sub-point.
        Includes few-shot examples and pass/fail criteria."""
        ...
    
    async def evaluate(self, session: SessionStore, 
                        retriever: EvidenceRetriever,
                        assessor: AssessorAgent) -> CheckpointResult:
        """Evaluate all sub-points for this checkpoint."""
        evaluations = []
        
        for sp in self.sub_points:
            sp_id = f"{self.checkpoint_id}.{sp}"
            description = self.get_sub_point_description(sp)
            guidance = self.get_evaluation_guidance(sp)
            
            # Retrieve evidence for this specific sub-point
            evidence_chunks = retriever.retrieve_for_sub_point(
                description, top_k=5
            )
            
            # Evaluate via LLM
            result = await assessor.evaluate_sub_point(
                sub_point_id=sp_id,
                description=description,
                evidence=evidence_chunks,
                guidance=guidance
            )
            
            evaluations.append(result)
        
        # Aggregate
        overall_score = sum(e.score for e in evaluations) / len(evaluations)
        overall_status = self._score_to_status(overall_score)
        
        return CheckpointResult(
            checkpoint_id=self.checkpoint_id,
            checkpoint_title=self.checkpoint_title,
            sub_point_evaluations=evaluations,
            overall_status=overall_status,
            overall_score=overall_score,
            summary=self._generate_summary(evaluations)
        )
```

### Concrete Evaluator Example

```python
@register(1)
class Q01TestStrategy(CheckpointEvaluator):
    """Q1: Is there a documented test strategy?"""
    
    checkpoint_id = "Q1"
    checkpoint_title = "Test Strategy"
    sub_points = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    
    def get_sub_point_description(self, sp: str) -> str:
        descriptions = {
            "a": "Test strategy defines the overall test approach and methodology",
            "b": "Test strategy identifies test levels (unit, integration, system, qualification)",
            "c": "Test strategy defines entry and exit criteria for each test level",
            "d": "Test strategy identifies test environment requirements",
            "e": "Test strategy defines test data requirements",
            "f": "Test strategy addresses risk-based test prioritization",
            "g": "Test strategy defines test metrics and reporting",
            "h": "Test strategy identifies required tools and infrastructure",
            "i": "Test strategy defines roles and responsibilities",
            "j": "Test strategy is reviewed and approved",
            "k": "Test strategy is version-controlled and accessible"
        }
        return descriptions[sp]
    
    def get_evaluation_guidance(self, sp: str) -> str:
        guidance = {
            "a": """CRITERIA: The test strategy document must explicitly state the 
overall approach (e.g., V-model based, risk-based, requirements-based).

EXAMPLES OF COMPLIANT:
- "The testing approach follows a risk-based methodology where test priority 
  is determined by ASIL rating and feature complexity."
- "Testing is structured according to the V-model with defined test levels."

EXAMPLES OF NON-COMPLIANT:
- A document titled "Test Strategy" that only contains test cases (no approach).
- A test plan with no overarching strategy discussion.

LOOK FOR: explicit mention of methodology, approach, philosophy, or framework.""",
            
            "c": """CRITERIA: Entry and exit criteria must be explicitly defined for 
EACH test level mentioned in the strategy.

EXAMPLES OF COMPLIANT:
- A table showing Entry Criteria and Exit Criteria columns for each test level.
- "Integration testing entry criteria: all unit tests pass, code review complete."
- "Exit criteria for qualification testing: 100% requirement coverage, 0 open 
  critical defects."

EXAMPLES OF NON-COMPLIANT:
- "Testing begins when code is ready" (too vague, not per-level).
- Exit criteria mentioned for only one test level.

LOOK FOR: tables with entry/exit columns, or sections titled "Entry Criteria" 
and "Exit Criteria".""",
            # ... similar guidance for other sub-points
        }
        return guidance.get(sp, "Evaluate whether evidence satisfies this sub-point.")
```

### Why Plugin Architecture?

1. **Extensibility:** Adding checkpoint Q17 means creating one new file with `@register(17)` — no changes to core framework
2. **Isolation:** Each evaluator encapsulates its own criteria; changes to Q3 cannot affect Q7
3. **Testing:** Each evaluator can be unit-tested independently
4. **Domain expertise:** Different domain experts can author different evaluators without merge conflicts

---

## 9.9 Follow-Up Questions & Action Items

### Automatic Follow-Up Generation

When a sub-point is evaluated as `partial` or `missing`, the system automatically generates a targeted follow-up question:

```python
async def generate_follow_up(self, sub_point_id: str, status: str, 
                              description: str, existing_evidence: str) -> str:
    """Generate a specific follow-up question for partial/missing sub-points."""
    
    if status == "missing":
        prompt = f"""The assessment found NO evidence for: {description}
        
Generate a specific question to ask the team to provide this evidence.
The question should be actionable and specific, not generic.
Example: Instead of "Do you have a test strategy?" ask 
"Can you provide the document that defines your risk-based test prioritization approach?"

Question:"""
    
    elif status == "partial":
        prompt = f"""The assessment found PARTIAL evidence for: {description}
        
Evidence found: {existing_evidence}

What is missing: Generate a specific question asking for the MISSING part.
The question should acknowledge what was found and ask for what is still needed.

Question:"""
    
    response = await self.llm.invoke(prompt)
    return response.strip()
```

### User-Driven Re-Evaluation

Follow-up questions can be answered by the team being assessed. Their answers are fed back into the evaluation:

```python
async def re_evaluate_with_answer(self, sub_point_id: str, 
                                    original_evidence: list[dict],
                                    follow_up_answer: str) -> SubPointEvaluation:
    """Re-evaluate a sub-point with additional context from team's answer."""
    
    # Combine original evidence with the new answer
    combined_evidence = original_evidence + [{
        "text": f"[Team Response]: {follow_up_answer}",
        "metadata": {"source_name": "follow_up_response", "chunk_type": "answer"}
    }]
    
    # Re-run evaluation with expanded context
    result = await self.assessor.evaluate_sub_point(
        sub_point_id=sub_point_id,
        description=self.get_description(sub_point_id),
        evidence=combined_evidence,
        guidance=self.get_guidance(sub_point_id)
    )
    
    return result
```

### Action Item Consolidation

After all 16 checkpoints are evaluated, the system generates consolidated action items:

```python
async def generate_action_items(self, all_results: list[CheckpointResult]) -> list[ActionItem]:
    """Generate deduplicated, prioritized action items from all gaps."""
    
    # Collect all partial/missing sub-points
    gaps = []
    for result in all_results:
        for sp in result.sub_point_evaluations:
            if sp.status in ("partial", "missing"):
                gaps.append({
                    "checkpoint": result.checkpoint_title,
                    "sub_point": sp.sub_point_id,
                    "status": sp.status,
                    "description": sp.reasoning,
                    "follow_up": sp.follow_up_question
                })
    
    # Use LLM to consolidate and deduplicate
    prompt = f"""Given the following compliance gaps, generate a consolidated list 
of action items. Rules:
- Deduplicate: if multiple gaps point to the same missing document, create ONE action item
- Prioritize: Critical gaps (multiple checkpoints affected) first
- Be specific: "Create entry/exit criteria table in Test Strategy v2.1" not "Fix test strategy"
- Link to gaps: reference which checkpoints are affected

Gaps:
{json.dumps(gaps, indent=2)}

Return as JSON array of action items with fields: 
title, description, priority (high/medium/low), affected_checkpoints, effort_estimate"""
    
    response = await self.llm.invoke(prompt)
    return [ActionItem(**item) for item in json.loads(response)]
```

---

## 9.10 Output Generation

### Multi-Format Reporting

The assessment results are rendered in four formats, each serving a different audience:

### Excel: ASPICE Template

Fills the standard ASPICE assessment template with Yes/Partial/No per checkpoint:

```python
from openpyxl import load_workbook

def generate_excel_report(results: list[CheckpointResult], template_path: str) -> bytes:
    """Fill ASPICE Excel template with assessment results."""
    wb = load_workbook(template_path)
    ws = wb["Assessment"]
    
    # Column mapping: B=Checkpoint, C=Status, D=Evidence, E=Gaps
    for i, result in enumerate(results):
        row = i + 3  # Start after header rows
        ws[f"B{row}"] = result.checkpoint_title
        ws[f"C{row}"] = result.overall_status
        ws[f"D{row}"] = "; ".join(
            sp.evidence_source for sp in result.sub_point_evaluations 
            if sp.status == "found"
        )
        ws[f"E{row}"] = "; ".join(
            sp.reasoning for sp in result.sub_point_evaluations 
            if sp.status == "missing"
        )
    
    # Summary sheet
    ws_summary = wb["Summary"]
    scorecard = compute_scorecard(results)
    ws_summary["B2"] = f"{scorecard['completeness_percentage']}%"
    ws_summary["B3"] = scorecard["checkpoints_compliant"]
    ws_summary["B4"] = scorecard["checkpoints_partial"]
    ws_summary["B5"] = scorecard["checkpoints_non_compliant"]
    
    output = BytesIO()
    wb.save(output)
    return output.getvalue()
```

### PDF: Branded Report

Generates a branded PDF with executive summary, detailed findings, and evidence citations:

```python
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer

def generate_pdf_report(results: list[CheckpointResult], 
                         metadata: dict) -> bytes:
    """Generate MBRDI-branded PDF assessment report."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    
    story = []
    
    # Header with branding
    story.append(Paragraph("ASPICE SWE.6 Assessment Report", styles["Title"]))
    story.append(Paragraph(f"Team: {metadata['team_name']}", styles["Subtitle"]))
    story.append(Paragraph(f"Date: {metadata['date']}", styles["Normal"]))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    scorecard = compute_scorecard(results)
    story.append(Paragraph("Executive Summary", styles["Heading1"]))
    story.append(Paragraph(
        f"Overall completeness: {scorecard['completeness_percentage']}%. "
        f"{scorecard['checkpoints_compliant']} checkpoints compliant, "
        f"{scorecard['checkpoints_partial']} partial, "
        f"{scorecard['checkpoints_non_compliant']} non-compliant.",
        styles["Normal"]
    ))
    
    # Detailed findings per checkpoint
    for result in results:
        story.append(Paragraph(
            f"{result.checkpoint_id}: {result.checkpoint_title} — {result.overall_status}",
            styles["Heading2"]
        ))
        # Sub-point table
        table_data = [["Sub-Point", "Status", "Evidence", "Gap"]]
        for sp in result.sub_point_evaluations:
            table_data.append([
                sp.sub_point_id,
                sp.status.value,
                sp.evidence_text[:100] + "..." if len(sp.evidence_text) > 100 else sp.evidence_text,
                sp.reasoning if sp.status != "found" else ""
            ])
        story.append(Table(table_data))
    
    doc.build(story)
    return buffer.getvalue()
```

### Markdown: Full Text Report

```python
def generate_markdown_report(results: list[CheckpointResult]) -> str:
    """Generate full Markdown report with tables."""
    lines = ["# ASPICE SWE.6 Assessment Report\n"]
    
    scorecard = compute_scorecard(results)
    lines.append(f"**Overall Completeness:** {scorecard['completeness_percentage']}%\n")
    
    # Summary table
    lines.append("| Checkpoint | Status | Score |")
    lines.append("|---|---|---|")
    for result in results:
        emoji = {"Yes": "Pass", "Partial": "Partial", "No": "Fail"}[result.overall_status]
        lines.append(f"| {result.checkpoint_id}: {result.checkpoint_title} | "
                     f"{emoji} | {result.overall_score:.2f} |")
    
    # Detailed sections
    for result in results:
        lines.append(f"\n## {result.checkpoint_id}: {result.checkpoint_title}\n")
        lines.append(f"**Status:** {result.overall_status} | **Score:** {result.overall_score:.2f}\n")
        lines.append("| Sub-Point | Status | Evidence Source | Reasoning |")
        lines.append("|---|---|---|---|")
        for sp in result.sub_point_evaluations:
            lines.append(f"| {sp.sub_point_id} | {sp.status.value} | "
                         f"{sp.evidence_source} | {sp.reasoning} |")
    
    return "\n".join(lines)
```

### JSON: Raw Evidence Log

The JSON output preserves the complete provenance chain for audit purposes:

```python
def generate_json_report(results: list[CheckpointResult], 
                          session_metadata: dict) -> str:
    """Generate raw JSON with full provenance chains."""
    report = {
        "session_id": session_metadata["session_id"],
        "timestamp": session_metadata["timestamp"],
        "model_versions": {
            "ingestion": "claude-opus-4-5-20250514",
            "assessor": "claude-opus-4-20250514"
        },
        "documents_processed": session_metadata["documents"],
        "scorecard": compute_scorecard(results),
        "checkpoints": [result.model_dump() for result in results],
        "provenance": {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_strategy": "section-aware with 50-token overlap",
            "retrieval_top_k": 5,
            "confidence_threshold": 0.6
        }
    }
    return json.dumps(report, indent=2, default=str)
```

---

## 9.11 Case Study: SPOT CHECK V2

### System Overview

SPOT CHECK V2 is the production implementation of the architecture described in this chapter, deployed at Mercedes-Benz Research & Development India (MBRDI) for automated ASPICE SWE.6 compliance assessment.

### Scale

- **16 checkpoints** (Q1-Q16), each with 3-11 sub-points
- **60+ evaluation dimensions** total
- **Per assessment:** typically 10-30 documents (PDF, DOCX, XLSX)
- **Per session:** 200-2000 chunks in ChromaDB
- **Assessment time:** sub-hour (compared to 2-3 days manual)

### Architecture Decisions in Practice

**Per-Session ChromaDB:**

Each assessment starts with an empty ChromaDB collection. This provides:
- Complete data isolation between teams
- No "stale evidence" from previous assessments
- Simple cleanup (delete session directory when done)
- Corruption recovery (if ChromaDB corrupts, restart session — no global state lost)

```python
class ChromaDBRecovery:
    """Handle ChromaDB corruption gracefully."""
    
    @staticmethod
    def safe_init(session_id: str) -> SessionVectorStore:
        try:
            store = SessionVectorStore(session_id)
            # Verify collection is accessible
            store.collection.count()
            return store
        except Exception as e:
            logger.warning(f"ChromaDB corrupted for session {session_id}: {e}")
            # Nuke and recreate
            shutil.rmtree(f"./sessions/{session_id}/chroma", ignore_errors=True)
            return SessionVectorStore(session_id)
```

**Dual-Agent with Model Specialization:**

The ingestion agent uses Claude Opus 4.5 because:
- Document parsing requires patience with messy formatting
- Large context window handles multi-page documents
- The task is mechanical — extract and structure, not judge

The assessor agent uses Claude Opus 4.6 because:
- Compliance evaluation requires nuanced reasoning
- Must distinguish "adequate evidence" from "related but insufficient evidence"
- Better at following complex evaluation rubrics
- Superior structured output adherence

**Cross-Model Fallback with Exponential Backoff:**

```python
class ResilientModelClient:
    """Handles model unavailability with fallback and retry."""
    
    def __init__(self):
        self.primary_models = {
            "ingestion": "claude-opus-4-5-20250514",
            "assessor": "claude-opus-4-20250514"
        }
        self.fallback_models = {
            "ingestion": "claude-opus-4-20250514",  # Opus 4.6 as fallback
            "assessor": "claude-sonnet-4-20250514"    # Sonnet as fallback
        }
    
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((TimeoutError, HTTPStatusError))
    )
    async def invoke_with_fallback(self, role: str, messages: list, 
                                     tools: list = None) -> dict:
        try:
            return await self.nexus_client.invoke(
                model=self.primary_models[role],
                messages=messages,
                tools=tools
            )
        except (RateLimitError, ServiceUnavailableError):
            logger.warning(f"Primary model for {role} unavailable, using fallback")
            return await self.nexus_client.invoke(
                model=self.fallback_models[role],
                messages=messages,
                tools=tools
            )
```

### End-to-End Flow

```
1. User uploads documents (PDF, DOCX, XLSX)
         │
         ▼
2. AdapterRegistry routes each document to correct adapter
         │
         ▼
3. Ingestion Agent (Opus 4.5) processes documents:
   - Extract text, tables, images
   - LLM-enhanced extraction for poor formatting
   - Vision API for embedded diagrams
         │
         ▼
4. SectionAwareChunker produces chunks with metadata
         │
         ▼
5. Chunks embedded (all-MiniLM-L6-v2) and stored in session ChromaDB
         │
         ▼
6. AgentCoordinator triggers Assessor Agent (Opus 4.6)
         │
         ▼
7. For each checkpoint Q1-Q16:
   - Load evaluator via @register decorator
   - For each sub-point:
     - Retrieve top-5 evidence chunks
     - Evaluate with guidance + anti-hallucination prompt
     - Produce SubPointEvaluation (status, score, evidence, reasoning)
   - Aggregate to CheckpointResult
         │
         ▼
8. Generate follow-up questions for partial/missing sub-points
         │
         ▼
9. Compute hierarchical scorecard
         │
         ▼
10. Generate outputs: Excel (template), PDF (branded), Markdown, JSON
```

### Results

| Metric | Manual Assessment | SPOT CHECK V2 | Improvement |
|--------|-------------------|---------------|-------------|
| Time per assessment | 2-3 days | < 1 hour | **~85% reduction** |
| Assessor consistency | Variable | Deterministic | **Eliminated subjectivity** |
| Evidence coverage | Depends on assessor | All documents scanned | **Complete coverage** |
| Audit trail | Notes in spreadsheet | Full JSON provenance | **Complete traceability** |
| Scalability | Linear with headcount | Parallel sessions | **Unlimited scale** |

### Limitations and Human Role

SPOT CHECK V2 is a **first-pass tool**, not a replacement for human assessors:

1. **Final judgment remains human.** The tool surfaces evidence and gaps; a human assessor makes the final compliance determination.
2. **Implicit evidence.** Some compliance evidence is cultural or tacit (e.g., "we always do code review" is not documented). The tool cannot assess undocumented practices.
3. **Context sensitivity.** The tool cannot judge whether a particular risk mitigation is adequate for a safety-critical vs. comfort feature — that requires domain expertise.
4. **Interview-based evidence.** Some ASPICE evidence comes from interviews with team members. The tool assesses only documented artifacts.

---

## 9.12 Summary & Interview Tips

### Key Architecture Patterns

| Pattern | Where Used | Why |
|---------|-----------|-----|
| **Adapter Pattern** | Document parsing (PDF, DOCX, Excel) | Uniform interface over diverse formats |
| **Registry Pattern** | AdapterRegistry, Evaluator registry | Auto-discovery, extensibility without core changes |
| **Mediator Pattern** | AgentCoordinator | Decouples agents, centralizes communication |
| **Plugin Architecture** | @register(N) for checkpoint evaluators | Add/modify checkpoints without touching framework |
| **Strategy Pattern** | Different evaluation_guidance per sub-point | Same evaluation flow, different criteria |

### How to Explain A2A in Interviews

**The 30-second version:**

"We split the system into two agents — an Ingestion Agent that parses and indexes documents, and an Assessor Agent that evaluates evidence against compliance criteria. They communicate through a shared vector store mediated by a coordinator. This separation lets us use different models optimized for each task: a patient, large-context model for ingestion, and a strong reasoning model for evaluation."

**Follow-up: "Why not one agent?"**

"One agent would need to hold all documents in context AND apply evaluation criteria simultaneously. By separating, we can process documents once, build a searchable evidence base, and then query it precisely per sub-point. It also makes the system testable — you can verify ingestion quality independently from assessment quality."

**Follow-up: "Why not direct communication?"**

"Direct agent-to-agent communication creates coupling — if we change the ingestion agent's output format, the assessor breaks. The coordinator as mediator means we can swap either agent independently. It also gives us a natural point for logging, retry logic, and session management."

### Common Interview Questions

**Q: "How do you prevent the LLM from hallucinating compliance evidence?"**

A: Three mechanisms. First, an explicit system prompt that says "Blank is better than wrong" — the model is instructed to mark sub-points as "missing" rather than generating evidence. Second, citation requirement — the model must quote exact text from retrieved chunks; if it cannot quote, it cannot claim compliance. Third, confidence scoring with automatic downgrade — if confidence is below 0.6, the status is mechanically downgraded regardless of what the model claims.

**Q: "How does the system handle documents it cannot parse?"**

A: Graceful degradation. The AdapterRegistry raises `UnsupportedFormatError` if no adapter matches. The coordinator logs this, notifies the user which documents could not be processed, and continues assessment with available documents. The final report notes which documents were excluded and why.

**Q: "Why ChromaDB and not a more production-grade vector database?"**

A: Two reasons. First, per-session isolation — we need fresh, empty collections per assessment, not a shared production database. ChromaDB's lightweight nature makes creating and destroying collections trivial. Second, data volume — each assessment involves hundreds to low-thousands of chunks, not millions. ChromaDB is perfectly adequate at this scale and avoids the operational overhead of a distributed vector database.

**Q: "How do you validate that the AI assessment is accurate?"**

A: We ran parallel assessments — the same documents were assessed both manually (by certified ASPICE assessors) and by SPOT CHECK V2. We measured agreement at the sub-point level. Where disagreements occurred, we analyzed whether the tool missed evidence (false negative) or hallucinated evidence (false positive). False negatives are acceptable (the tool is conservative); false positives trigger system prompt refinement.

**Q: "What happens when ASPICE criteria change?"**

A: The plugin architecture makes this straightforward. Each checkpoint evaluator is a self-contained class with its own criteria and guidance. Updating criteria means modifying the `get_evaluation_guidance()` method of the affected evaluator. Adding new checkpoints means creating a new `@register(N)` class. The framework itself never changes.

**Q: "How do you handle large documents that exceed context limits?"**

A: The chunking pipeline ensures no single LLM call receives more than it can process. Documents are chunked into 512-token pieces with overlap. The assessor agent only sees the top-5 most relevant chunks per sub-point, never entire documents. The ingestion agent processes documents page-by-page for extraction, never loading the full document into a single prompt.

### Key Takeaways for SPOT CHECK V2

1. **Compliance assessment is a structured RAG problem** — retrieve evidence, evaluate against criteria, produce structured output.

2. **Anti-hallucination is the #1 priority** — in compliance, a false positive is catastrophic. Every design decision prioritizes precision over recall.

3. **Hierarchical scoring enables both detailed and summary views** — executives see percentages; assessors see per-sub-point evidence gaps.

4. **Agent separation follows the Single Responsibility Principle** — parse/index is fundamentally different from evaluate/judge.

5. **Plugin architecture future-proofs against process changes** — ASPICE evolves; the framework stays stable.

6. **Per-session state eliminates cross-contamination** — each assessment is isolated, reproducible, and auditable.

7. **The tool augments, not replaces, human judgment** — it provides a consistent, comprehensive first pass that human assessors can review and override.

---

*Next chapter: Chapter 10 — Search Algorithms*
