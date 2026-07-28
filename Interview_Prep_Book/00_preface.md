# Preface

## Why This Book Exists

I wrote this book because I was tired of the gap between what interviews ask and what engineers actually build.

Every technical concept in this book was used in a real system that shipped — not a Kaggle competition, not a course project, not a "tutorial I followed." When I explain Bayesian Log-Likelihood Ratios, I show you the exact engine that achieved 97% accuracy diagnosing failures across 67,000 signals at Mercedes-Benz. When I walk through A* search, I derive the admissibility proof that's in my TMLR paper.

This is the book I wished I had before my own interviews.

## Who This Is For

- M.Tech / MS students preparing for AI/ML roles
- Engineers transitioning into LLM/GenAI engineering
- Anyone preparing for interviews at companies building AI agents, RAG systems, or reasoning engines
- Researchers who want to understand production systems
- Builders who want to understand research rigor

## How This Book Is Organized

**Part I (Chapters 1-5):** Foundations — the theory behind everything. Read this if you need to refresh fundamentals or explain WHY something works.

**Part II (Chapters 6-9):** Systems — real production architectures, broken down component by component. Each chapter ends with a case study from my own projects.

**Part III (Chapters 10-12):** Research — how to do rigorous experiments, prove things formally, and publish.

**Part IV (Chapters 13-15):** Engineering — the Python, infra, and deployment skills that separate "proof of concept" from "production system."

**Part V (Chapter 16):** 210+ interview questions with detailed answers. Organized by topic. Start here if your interview is tomorrow.

## The Projects

Everything in this book traces back to five real systems:

1. **iFAST** — An autonomous AI agent that diagnoses root causes of software failures using Bayesian evidence fusion. Built at Mercedes-Benz R&D.

2. **SSV RAG Chatbot** — A production retrieval system serving real-time Q&A over enterprise data (Jira + Confluence) with hybrid retrieval and 5.5× latency optimization.

3. **SPOT CHECK V2** — A dual-agent compliance assessment system that automates quality audits using structured evaluation with mandatory evidence citation.

4. **MTP Research (A* + Topology Router)** — A search algorithm over LLM reasoning traces with formal optimality guarantees, submitted to TMLR.

5. **Moshi Moshi Tools** — Sales automation (Selenium + LLMs) and a GenAI design assistant for a branding agency.

---

*Md Tanzeel Adam Khan*  
*M.Tech AI, IIT Patna*  
*July 2026*
