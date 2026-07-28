# iFAST — Intelligent Failure Analysis through Signal Tracing

## For Resume Agent: Read This First
  
This document describes a production system I designed and built at Mercedes-Benz R&D India. Translate this into resume bullets that a **non-automotive** hiring manager (Google, Amazon, fintech, AI startup) can immediately understand. Avoid deep automotive jargon — frame as systems engineering, AI/ML, and data engineering achievements.

---

## The Problem (Universal Framing)

A modern car's brain (ECU) runs **67,000+ real-time signals** across **1,128 software modules** — updated weekly. When a signal fails during testing, an engineer must determine: **"Was this caused by a code change, or is it a test environment issue?"**

This is equivalent to: *Given a microservice failure in a 1,128-service distributed system with 67K inter-service data flows, identify which code deploy caused the regression — without access to runtime logs, only static code analysis.*

**Before iFAST:** Engineers manually investigated using Simulink diagrams, git diffs, and spreadsheets. **4-8 hours per failure.** With 50+ failures per week, the backlog was growing 10x faster than the team could investigate.

---

## What I Built

A **fully autonomous AI agent** that performs the entire investigation in **under 60 seconds** — from parsing the failure report to generating a production-grade root cause analysis with cited evidence.

Think of it as: **"An AI SRE that does incident root-cause analysis by fusing 8 independent data sources, running Bayesian inference, and producing a confidence-scored verdict with a causal explanation."**

---

## Impact (Hard Numbers)

| Metric | Before | After | Impact |
|--------|--------|-------|--------|
| Investigation time per failure | 4-8 hours | **< 60 seconds** | **99.7% faster** |
| Daily throughput | 3-5 signals (human-limited) | **Unlimited** | **Removed bottleneck** |
| False positive rate | ~20% | **< 5%** | **4x more precise** |
| Data freshness | Days-to-weeks stale | **Real-time** | **Eliminated drift** |
| Dependency graph coverage | 50K nodes | **67,637 nodes** | **+35% completeness** |
| Evidence sources cross-validated | 1-2 (manual) | **8 (automated)** | **4x more thorough** |
| Root cause explanation | "Probably software" | **"Function X deleted from Module Y at line 3809"** | **Actionable vs vague** |

---

## Technical Architecture (Transferable Design Patterns)

### 1. Autonomous AI Agent with Tool-Use

Built a **single-agent system** that orchestrates 14 specialized tools via MCP (Model Context Protocol) over JSON-RPC 2.0. The agent decides which tools to call, in what order, handles missing data autonomously, and produces structured output — no human in the loop.

**Industry equivalent:** Like building a custom LangChain/CrewAI agent, but production-hardened with session state, caching, error recovery, and real-time progress feedback.

**Tech:** Claude (Anthropic) via AWS Bedrock, custom tool schemas, streaming responses, token budget management (200K context window).

### 2. Bayesian Evidence Fusion Engine

Designed a **probabilistic inference system** that combines 8 independent evidence sources using Log-Likelihood Ratios (LLR). Each source contributes a mathematically grounded weight — positive for "code change detected," negative for "code confirmed unchanged."

```
P(CodeChange | Evidence) = sigmoid(Σ LLR_i)

Where each LLR_i comes from an independent automated checker:
  +3.5  Visual structural diff (strongest)
  +3.0  Source code interface deleted (ground truth)
  +2.5  Git confirms module file changed
  +2.5  Configuration parameter missing
  -2.5  Source code confirms signal UNCHANGED (cancels false positive)
  -4.0  All 600+ modules confirmed unchanged (definitive negative)
```

**Industry equivalent:** Like building a fraud detection ensemble that fuses behavioral signals, network patterns, and device fingerprints — each with calibrated log-odds — into a single score.

### 3. Novel Source Code Dependency Graph Engine

Invented a method to extract a **complete service dependency graph** from auto-generated source code using regex pattern matching. Parses 1,128 modules across 3 git repositories in 86 seconds:

- **10,738 output interfaces** (what each module produces)
- **21,257 input interfaces** (what each module consumes)
- **28,097 configuration parameters** (tuning values)
- **17,563 internal state variables** (computation intermediates)
- **1,655,171 dependency edges** (who-feeds-who)

**Key innovation:** This replaced a manual export process (5-30 min + human intervention, results stale within days) with a **fully automated, version-specific** pipeline that runs from any git tag. The old manual process missed 35% of signals.

**Industry equivalent:** Like building a service mesh dependency map by parsing gRPC proto files + API gateway configs across 1,128 microservices — automated, versioned, always fresh.

### 4. Multi-Source Cross-Validation

Implemented an evidence verification system that detects when multiple independent sources agree (TRIPLE/DOUBLE confirmed) or contradict each other. This catches:
- False positives (module changed, but THIS specific signal didn't)
- False negatives (signal changed, but one source missed it)
- Data quality issues (stale exports, parser errors)

**Industry equivalent:** Like consensus verification in distributed systems — if 3 out of 4 independent monitors agree, confidence is high; if they contradict, flag for investigation.

### 5. Real-Time Progress Architecture

Solved a **long-running task + UI responsiveness** challenge:
- Background data generation (10-15 min subprocess)
- Non-blocking agent responses (returns immediately)
- Live progress streaming to frontend (file-based IPC, 5s polling)
- Graceful degradation (agent works with partial data, full results arrive later)

**Industry equivalent:** Like designing async job processing with live status updates (think: GitHub Actions UI showing step-by-step progress while builds run).

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **AI/LLM** | Claude (Anthropic), AWS Bedrock, MCP Protocol, Tool-use pattern, 200K context |
| **Backend** | Python 3.11, Bayesian inference, Regex engines, ProcessPool/ThreadPool parallelism |
| **Data Processing** | Custom parsers (C source, XML/SVG, binary formats), JSON-RPC 2.0, SQLite |
| **Frontend** | Streamlit (multi-page), custom design system, real-time progress widgets |
| **Infrastructure** | Git (4 repos × 236 tags), subprocess orchestration, file-based IPC |
| **Algorithms** | Bayesian LLR, graph traversal (BFS, 3-hop ancestor trace), content hashing (SHA-256), proximity matching |
| **Domain** | Embedded systems, AUTOSAR, signal processing, automotive safety (ISO 26262 context) |

---

## Engineering Challenges (Interview-Ready Stories)

### Challenge 1: "How do you determine causation from correlation?"

**Situation:** A module contains 16 output signals. If the module changed in git, naive approach says ALL 16 are affected.  
**Action:** Built signal-level verification — parse the actual source code to confirm which specific interfaces were added/removed/unchanged.  
**Result:** Eliminated 75% of false positives. Signals confirmed unchanged get -2.5 LLR (strong negative evidence), mathematically cancelling the module-level +2.5 positive signal.

### Challenge 2: "How do you make a 15-minute process feel instant?"

**Situation:** Full data generation takes 10-15 min (extracting + comparing 15,000 files). Agent session times out at 5 min.  
**Action:** Decomposed into fast-path (JSON evidence, 2-3 min) and slow-path (visual reports, 10+ min background). Agent returns verdict immediately with JSON evidence; visual reports stream in asynchronously.  
**Result:** User gets actionable verdict in <60s for pre-generated pairs, 3-4 min for new pairs. No timeouts.

### Challenge 3: "How do you replace a manual process without losing accuracy?"

**Situation:** Existing dependency data came from a manual GUI tool (Movetto) that traced signal paths through SVG block diagrams. 55MB of static CSV/JSON files, outdated within days.  
**Action:** Reverse-engineered the naming conventions of auto-generated code (TargetLink pattern: `Rte_IWrite_{Module}_P_{Signal}_{Signal}`). Built regex parser that extracts the same dependency information directly from source code at any git tag.  
**Result:** 35% more signal coverage, zero staleness, zero junk entries, 86-second build time (vs 5-30 min manual). Validated: byte-for-byte identical reports when compared against manual process output.

### Challenge 4: "How do you handle conflicting evidence?"

**Situation:** One source says "module changed" (+2.5), another says "signal is unchanged" (-2.5), a third says "parameter was added" (+2.5). What's the verdict?  
**Action:** Each source contributes independent LLR to a running sum. Cross-validation detects agreement/contradiction. Final probability = sigmoid(total_LLR). Threshold rule: LLR < 0 = always INCONCLUSIVE (evidence leans away from code change).  
**Result:** Mathematically grounded verdicts. No "vibes" — every decision traces to weighted evidence. Engineers trust it because it shows its work.

---

## What Makes This Stand Out

| Dimension | Typical Industry Approach | What I Built |
|-----------|--------------------------|-------------|
| Root cause analysis | Log-based, reactive, manual | Code-structure-based, proactive, automated |
| Dependency mapping | Static docs, outdated | Live from git, version-specific, 86s rebuild |
| Confidence scoring | Binary (pass/fail) or heuristic | Bayesian LLR with 8 sources, cross-validated |
| AI agent design | Single LLM call, no tools | 14-tool autonomous agent with state management |
| Evidence presentation | "We think it's X" | "Here's 8 sources of proof, mathematically scored" |
| Data freshness | Export once, hope it's right | Real-time from source of truth (git) |

---

## Resume Bullets (Ranked by Impact)

### For Senior/Staff Engineer Roles:

> **1.** Architected an **autonomous AI agent system** (Claude/Bedrock + 14 MCP tools) that performs **probabilistic root-cause analysis** across 67K+ data signals using **Bayesian evidence fusion from 8 independent sources**, reducing failure investigation time from 4-8 hours to <60 seconds with 97% accuracy

> **2.** Designed a **novel dependency graph engine** that extracts 1.6M+ inter-module relationships from auto-generated source code via regex parsing across 1,128 modules in 86 seconds — replacing a manual process that took 30 min, was 35% incomplete, and became stale within days

> **3.** Built a **real-time evidence fusion pipeline** combining code analysis, configuration diffs, binary comparisons, visual structural diffs, and model-based path tracing into a unified **Bayesian Log-Likelihood Ratio score** with cross-source validation (TRIPLE/DOUBLE confirmation)

> **4.** Eliminated **75% of false positives** in automated failure triage by implementing signal-level source code verification — confirming whether specific interfaces were modified between versions (not just module-level "something changed")

### For AI/ML Engineer Roles:

> **1.** Built a **production AI agent** with autonomous tool orchestration (14 tools, JSON-RPC 2.0), session state management, non-blocking async execution, and real-time progress streaming — zero human intervention from input to final report

> **2.** Designed a **Bayesian inference engine** fusing 8 independent evidence sources with calibrated Log-Likelihood Ratios, achieving 97% accuracy on binary classification with mathematically grounded confidence thresholds and cross-validation

### For Data/Platform Engineer Roles:

> **1.** Built an **automated data pipeline** that extracts, transforms, and indexes 67,637 signals from 1,128 source code modules across 3 repositories in 86 seconds — providing version-specific dependency graphs cached permanently for instant retrieval

> **2.** Designed a **dual-mode data architecture** (primary: live git extraction, fallback: static exports) with per-version caching, automatic graph construction on first access, and zero-downtime source switching

---

## Keywords (ATS-Optimized, Broad Coverage)

**AI/ML:** LLM, Claude, Anthropic, AWS Bedrock, AI Agent, Autonomous Agent, Tool-use, MCP Protocol, Bayesian Inference, Probabilistic Reasoning, Evidence Fusion, Cross-validation, Confidence Scoring, NLP

**Engineering:** Python, Distributed Systems, Microservices Architecture, Data Pipeline, ETL, Real-time Processing, Graph Algorithms, BFS, Dependency Resolution, Regex, Pattern Matching, Parallel Processing, ThreadPool, ProcessPool, Async

**Data:** SQLite, JSON, CSV, Git, Data Freshness, Cache Architecture, Graph Database (conceptual), 1.6M+ edges, 67K+ nodes

**Infrastructure:** AWS, Subprocess Orchestration, File-based IPC, Non-blocking I/O, Progress Streaming, Background Jobs, Session Management

**Frontend:** Streamlit, Real-time UI, Progress Widgets, HTML Report Generation, Data Visualization

**Domain:** Automotive, Embedded Systems, AUTOSAR, Signal Processing, Safety-Critical Software, ISO 26262, Root Cause Analysis, Incident Response

---

## Context for Resume Agent

- **Company:** Mercedes-Benz Research & Development India Pvt. Ltd.
- **Team:** Software Systems Validation (SSV)
- **Duration:** ~6 months (Q1-Q2 2026)
- **Role:** Lead Developer & System Architect (team of engineers)
- **Scale:** Production system used by SSV engineering team for daily triage
- **Complexity:** 8 evidence sources, 1,128 modules, 67K signals, 4 git repos, 14 AI tools
- **Innovation:** Novel C-source dependency extraction method (potential patent), Bayesian multi-source fusion for code change detection
