# Chapter 1: The AI/ML Landscape — Where Do You Fit?

*Before you answer a single technical question, you need to know what role you are interviewing for, what that company actually values, and how to present your experience so it lands.*

---

The biggest mistake candidates make is preparing "for ML interviews" as if that were one thing. A research scientist at DeepMind, an ML engineer at Uber, an AI engineer at a Series A startup, and a data scientist at a bank are all "AI roles" — but they evaluate completely different skills, ask completely different questions, and want completely different narratives from you.

This chapter helps you figure out where you fit, what each type of company actually screens for, how automated systems filter you before a human ever sees your resume, and how to build a coherent interview narrative from your actual projects.

---

## 1.1 Types of AI Roles

### Research Scientist

The job: publish papers, advance the state of the art, prove things formally, run experiments that answer open questions.

What they evaluate: mathematical maturity, novelty of ideas, experimental rigor, ability to identify important problems. They will ask you to derive things on a whiteboard, critique a paper's methodology, or propose how you would extend a published result.

Day-to-day: reading papers, writing proofs, designing experiments, running training jobs for weeks, writing papers, peer review, conference presentations.

My connection: The MTP thesis (A* search over reasoning traces, topology-based routing) lives squarely in this world. The TMLR submission required formal admissibility proofs, ablation studies, and claims backed by statistical tests — not just "it worked."

### ML Engineer

The job: take models and make them work in production. Latency budgets, reliability, scale, monitoring, graceful degradation. The model is one component in a larger system.

What they evaluate: system design (how do you serve a model at 100K requests per second?), algorithms, data pipelines, monitoring (how do you know when your model is degrading?), incident response. They will ask you to design ML systems on a whiteboard.

Day-to-day: model serving infrastructure, A/B testing frameworks, feature stores, retraining pipelines, latency optimization, incident debugging.

My connection: The SSV RAG chatbot is an ML engineering project — dual backends with failover, streaming responses, three-tier caching, 5.5x latency optimization, token budgets, monitoring. It is not about model novelty; it is about making the model reliable and fast in production.

### AI Engineer

The job: build applications powered by LLMs. Agents, RAG pipelines, prompt engineering, tool orchestration, structured output. You are building the system around the model, not the model itself.

What they evaluate: ability to ship working products fast, understanding of LLM capabilities and limitations, prompt design, system architecture for agent workflows, evaluation methodology for non-deterministic systems.

Day-to-day: designing agent architectures, building RAG pipelines, prompt engineering, evaluating output quality, integrating with enterprise systems, handling edge cases and failures.

My connection: iFAST (14-tool autonomous agent with Bayesian evidence fusion) and SPOT CHECK (dual-agent compliance assessment) are AI engineering projects. They use LLMs as components in carefully designed systems.

### Data Scientist

The job: extract insights from data, design experiments, communicate results to stakeholders who are not technical.

What they evaluate: statistical reasoning, experimental design (A/B tests, causal inference), communication skills, business sense. SQL proficiency is table stakes. They will ask you to design an experiment, interpret ambiguous results, or explain a tradeoff to a non-technical audience.

Day-to-day: SQL queries, dashboards, stakeholder presentations, experiment design, statistical analysis, "can you look at this data and tell me what's happening?"

My connection: Less directly — but the experimental design work in Chapter 12 (paired t-tests, ablations, statistical significance) overlaps heavily with what data scientists do.

### MLOps / Platform Engineer

The job: build the infrastructure that ML engineers and data scientists use. Model registries, experiment tracking, CI/CD for models, deployment automation, monitoring systems.

What they evaluate: infrastructure skills (Kubernetes, Docker, cloud services), CI/CD, monitoring, automation, reliability engineering. ML knowledge is secondary to DevOps skills.

Day-to-day: Kubernetes clusters, Airflow DAGs, model registries (MLflow), monitoring dashboards, GPU allocation, cost optimization.

---

## 1.2 What Companies Look For

### FAANG / Big Tech (Google, Meta, Amazon, Microsoft, Apple)

**Hiring philosophy:** Generalist with depth. They want someone who can solve novel problems across domains.

**Evaluation:** Algorithms (LeetCode medium-hard), system design (both general and ML-specific), breadth of ML knowledge, behavioral (leadership principles at Amazon, Googleyness at Google). Interviews are standardized — every candidate gets the same process regardless of team.

**What matters:** Strong CS fundamentals. A candidate who knows algorithms well and has "some" ML experience often beats a candidate who has deep ML knowledge but cannot write efficient code.

**Resume signal:** Brand names, scale numbers ("served 10M users"), publications at top venues, competitive programming.

### AI Startups (Series A-C)

**Hiring philosophy:** Can this person ship? Will they figure things out without hand-holding?

**Evaluation:** Take-home projects (build a RAG pipeline in 48 hours), pair programming, architecture discussions ("how would you build X?"), culture fit. Less standardized — the CTO might decide on the spot.

**What matters:** Speed and breadth. If you can build an end-to-end agent system in a week — frontend, backend, LLM integration, evaluation, deployment — you are valuable. Depth is secondary to velocity.

**Resume signal:** Shipped products, GitHub repos with actual users, blog posts showing how you solved real problems, side projects that demonstrate curiosity.

### Research Labs (DeepMind, FAIR, OpenAI Research, Anthropic Research)

**Hiring philosophy:** Can this person push the frontier of knowledge?

**Evaluation:** Paper presentations (present your own work, critique others'), mathematical derivations, research taste ("what problems do you think are important and why?"), experimental design.

**What matters:** Publications at top venues (NeurIPS, ICML, ICLR, AAAI). Strong mathematical foundation. Original ideas. A single strong first-author paper beats ten mediocre co-authored ones.

**Resume signal:** Publication record, research statement, advisor's reputation, novelty of approach.

### Enterprise / Corporate AI (Banks, Automotive, Healthcare)

**Hiring philosophy:** Will this person build systems we can trust?

**Evaluation:** System reliability ("how do you handle model failures?"), security awareness, integration with existing systems, compliance knowledge, stakeholder management.

**What matters:** Production experience. Systems that run 24/7 without human intervention. Understanding of regulatory constraints. Ability to work within existing corporate infrastructure.

**Resume signal:** Production deployments with SLAs, cross-functional collaboration, security clearances, domain expertise.

---

## 1.3 How ATS & AI Screening Works

Before a human reads your resume, automated systems decide whether you pass the first gate. Understanding these systems is not gaming — it is clear communication.

### Keyword Matching

ATS systems (Workday, Greenhouse, Lever) perform both exact and semantic keyword matching against the job description. If a job requires "RAG" and your resume says "retrieval-augmented generation" but never uses the acronym, some systems will miss it. Use both.

**Practical advice:** Read the job description. Identify every technical skill mentioned. Ensure each skill appears at least once in your resume, in the exact phrasing used by the JD. This is not deception — it is speaking the same language.

### Impact Quantification

Screeners — human and automated — scan for numbers. Every bullet on your resume should have at least one quantified impact:

- Bad: "Improved system performance"
- Better: "Reduced latency from 12s to 2.2s (5.5x improvement)"
- Bad: "Built a diagnostic tool"
- Better: "Built autonomous diagnostic agent achieving 97% accuracy across 67,000 signals"

Numbers are proof. Unquantified claims are just claims.

### Action Verb Strength

Start every bullet with a strong action verb that conveys the level of ownership:

- **Led / Architected / Designed:** You owned the system end-to-end
- **Built / Implemented / Developed:** You wrote the code
- **Improved / Optimized / Reduced:** You made something better (implies a before/after)
- **Assisted / Supported / Participated:** You were a helper, not a driver (avoid these)

### Relevance Density

Pack more relevant content per line. If you are applying for an AI Engineer role, lead with agent systems and RAG — not your coursework in database normalization.

### Education Signal

For early-career candidates (0-3 years), education acts as a filter. M.Tech from IIT carries weight in India. For experienced candidates, education matters less — your shipped work speaks louder.

---

## 1.4 The Skill Taxonomy

Your skills are not locked to one industry. Every technique you have mastered transfers across domains. Understanding this map gives you flexibility in where you apply.

### Autonomous Agents

If you can build agents that orchestrate tools, maintain state, and make decisions: you are relevant to every technology company building AI products. This is the hottest skill in 2025-2026.

- **Companies:** Anthropic, OpenAI, Google DeepMind, every AI startup, enterprise software companies (Salesforce, ServiceNow)
- **From my work:** iFAST (14-tool autonomous agent), SPOT CHECK (dual-agent compliance system)

### RAG (Retrieval-Augmented Generation)

If you can build production retrieval systems that ground LLM responses in real data: legal tech, healthcare, finance, enterprise software, and every company with internal knowledge bases wants you.

- **Companies:** Legal (Harvey, Casetext), healthcare (Hippocratic AI), finance (Bloomberg), enterprise (Microsoft Copilot, Glean, Notion)
- **From my work:** SSV RAG Chatbot (Jira + Confluence retrieval with hybrid search)

### Bayesian Reasoning & Evidence Fusion

If you can build systems that update beliefs from multiple evidence streams: finance (risk modeling), cybersecurity (threat detection), healthcare (diagnostic support), and quality/reliability engineering.

- **Companies:** Insurance, fintech, cybersecurity firms, autonomous vehicles (sensor fusion is Bayesian)
- **From my work:** iFAST (Log-Likelihood Ratio engine fusing 67,000 signals)

### Graph Algorithms

If you understand graph traversal, pathfinding, and network analysis: cybersecurity (attack path analysis), supply chain (network optimization), social networks, recommendation systems.

- **From my work:** MTP thesis (A* search over reasoning graphs), iFAST (Jira issue-link graph traversal)

### Search & Routing

If you can design systems that efficiently explore solution spaces: AI labs (MCTS for reasoning), search companies (query routing), logistics (combinatorial optimization).

- **From my work:** MTP thesis (A* with formal admissibility guarantees, topology-based routing)

---

## 1.5 Building Your Interview Narrative

Technical skill gets you through the questions. Narrative gets you the offer. Every interviewer is implicitly asking: "Does this person have a coherent story? Do they know what they want and why?"

### Pick 2-3 Projects

Choose projects that cover different skills. For me:
1. **iFAST** — shows autonomous agent engineering, Bayesian reasoning, production deployment at scale
2. **MTP Thesis** — shows research rigor, formal methods, algorithmic novelty
3. **SSV RAG** — shows production ML engineering, latency optimization, system design

Each project covers different ground. Together, they tell a complete story.

### The STAR-T Framework

For each project, prepare both a 2-minute version and a 10-minute version:

- **Situation:** What was the problem? Why did it matter? (1-2 sentences)
- **Task:** What specifically were you responsible for? (1 sentence)
- **Approach:** What did you build? What were the key technical decisions? (2-3 sentences for short, 5-8 for long)
- **Result:** What happened? Quantify. (1-2 sentences)
- **Tradeoff / What I'd Improve:** What would you do differently now? (1 sentence)

The "Tradeoff" is what separates senior candidates from junior ones. Junior candidates describe what they built. Senior candidates describe what they chose NOT to build and why.

### The 2-Minute Version (iFAST Example)

"At Mercedes-Benz R&D, we had a problem: diagnosing root causes of software failures across ECU test systems took engineers hours of manual investigation across multiple data sources. I built iFAST — an autonomous AI agent powered by Claude that orchestrates 14 tools to investigate failures end-to-end. The key technical contribution was a Bayesian Log-Likelihood Ratio engine that fuses evidence from Jira, Confluence, test logs, and historical patterns to produce ranked root cause hypotheses with calibrated confidence scores. The system achieved 97% diagnostic accuracy across 67,000 signals and reduced investigation time from hours to minutes. If I were to rebuild it, I would add an explicit planning layer — currently the agent decides tool calls reactively, but a plan-then-execute architecture would reduce unnecessary tool calls by 30-40%."

### Practice Both Versions

Record yourself. The 2-minute version should feel effortless — you should be able to deliver it without thinking. The 10-minute version is for deep dives where the interviewer is genuinely interested. If you cannot tell the 2-minute version smoothly, you are not ready.

---

## 1.6 Summary

Before you study a single algorithm, know:
1. **What role** you are targeting (this determines what to emphasize)
2. **What type of company** you want (this determines the evaluation style)
3. **How to present yourself** (this determines whether you get the interview)
4. **Where your skills transfer** (this expands your search space)
5. **What narrative ties it together** (this gets you the offer)

The rest of this book covers the technical knowledge. This chapter covers the strategy. Both are necessary; neither is sufficient alone.
