# Chapter 8: Graph Algorithms & Dependency Analysis

> "Show me your dependencies and I'll show you your failure modes."

---

## 8.1 Graph Fundamentals

A **graph** is a mathematical structure G = (V, E) consisting of a set of **vertices** (nodes) and a set of **edges** (connections between nodes). Graphs model relationships: modules that depend on each other, cities connected by roads, users connected by friendships.

### Directed vs. Undirected

In a **directed graph** (digraph), edges have direction — an edge (u, v) means "u points to v" but not necessarily the reverse. Dependency relationships are inherently directed: if Module A consumes a signal produced by Module B, the edge is B → A. In an **undirected graph**, edges are symmetric: if u connects to v, then v connects to u.

### Weighted vs. Unweighted

A **weighted graph** assigns a numerical value to each edge — representing cost, distance, latency, or capacity. An **unweighted graph** treats all edges equally. Dependency graphs are typically unweighted (a dependency either exists or it doesn't), but weighted variants encode dependency strength or data volume.

### Key Terminology

| Term | Definition |
|------|-----------|
| **Vertex (node)** | A fundamental unit in the graph |
| **Edge (arc)** | A connection between two vertices |
| **In-degree** | Number of edges pointing INTO a node |
| **Out-degree** | Number of edges pointing OUT of a node |
| **Path** | A sequence of edges connecting two vertices |
| **Cycle** | A path that starts and ends at the same vertex |
| **DAG** | Directed Acyclic Graph — a directed graph with no cycles |

### Graph Representations

**Adjacency List** — each node stores a list of its neighbors. Space: O(V + E). Best for sparse graphs (most real-world graphs). Iterating over a node's neighbors is O(degree). Checking if edge (u,v) exists is O(degree(u)).

```python
# Adjacency list (dictionary of lists)
graph = {
    "ModuleA": ["ModuleB", "ModuleC"],
    "ModuleB": ["ModuleD"],
    "ModuleC": ["ModuleD"],
    "ModuleD": []
}
```

**Adjacency Matrix** — a V×V matrix where entry [i][j] = 1 if edge (i,j) exists. Space: O(V²). Best for dense graphs. Edge lookup is O(1). Iterating neighbors is O(V).

```python
# Adjacency matrix
#           A  B  C  D
matrix = [[0, 1, 1, 0],  # A → B, A → C
          [0, 0, 0, 1],  # B → D
          [0, 0, 0, 1],  # C → D
          [0, 0, 0, 0]]  # D → nothing
```

**For dependency graphs with 1,128 modules and 1.6M edges:** adjacency list is the clear winner. The matrix would waste memory on V² = 1.27M cells, while the adjacency list stores only the edges that actually exist.

### Directed Acyclic Graphs (DAGs)

A DAG is a directed graph with no cycles. This is the fundamental structure for dependency resolution — if A depends on B and B depends on A, you have a circular dependency (a bug). DAGs guarantee that a valid build order exists, which is exactly what topological sort computes.

---

## 8.2 Breadth-First Search (BFS)

BFS explores a graph level by level, visiting all neighbors of the current node before moving deeper. It uses a **queue** (FIFO) to maintain the exploration frontier.

### Algorithm

```python
from collections import deque

def bfs(graph, start):
    """Breadth-first search from start node."""
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []
    
    while queue:
        node = queue.popleft()
        order.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return order
```

### Complexity

- **Time:** O(V + E) — every vertex and edge is examined exactly once.
- **Space:** O(V) — the queue can hold at most V nodes (in a star graph, all leaves are enqueued simultaneously).

### Use Cases

| Use Case | Why BFS? |
|----------|----------|
| Shortest path (unweighted) | BFS finds minimum-hop paths by construction |
| Level-order traversal | Natural level-by-level expansion |
| Connected components | BFS from each unvisited node discovers a component |
| **Upstream dependency resolution** | Find all ancestors within N hops |

BFS is the algorithm of choice when you need to answer "what are all the things within N steps of this node?" — exactly the question dependency analysis asks.

---

## 8.3 Depth-First Search (DFS)

DFS explores as deep as possible along each branch before backtracking. It uses a **stack** (LIFO) — either the call stack (recursion) or an explicit stack.

### Algorithm (Iterative)

```python
def dfs(graph, start):
    """Depth-first search using explicit stack."""
    visited = set()
    stack = [start]
    order = []
    
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        
        # Add neighbors in reverse for consistent ordering
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append(neighbor)
    
    return order
```

### Recursive vs. Iterative

The recursive version is elegant but dangerous on deep graphs — Python's default recursion limit is 1,000 frames. A dependency graph with a chain of 1,500 modules would crash. The iterative version handles arbitrary depth.

```python
def dfs_recursive(graph, node, visited=None):
    """Recursive DFS — concise but limited by stack depth."""
    if visited is None:
        visited = set()
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
    return visited
```

### Complexity

- **Time:** O(V + E) — same as BFS.
- **Space:** O(V) — the stack depth in the worst case (linear graph).

### Use Cases

| Use Case | Why DFS? |
|----------|----------|
| Cycle detection | Track "currently in stack" nodes — back edge = cycle |
| Topological sort | Post-order DFS gives reverse topological order |
| Path existence | "Does ANY path exist from A to B?" |
| Strongly connected components | Tarjan's or Kosaraju's algorithm |

### BFS vs. DFS: When to Use Which

**BFS** when you need shortest paths or level information. **DFS** when you need to detect cycles, compute topological order, or explore all reachable nodes without caring about distance. In dependency analysis, both have roles: BFS for "what are the immediate N-hop ancestors?" and DFS for "is there a circular dependency anywhere?"

---

## 8.4 Topological Sort

A **topological ordering** of a DAG is a linear sequence of all vertices such that for every directed edge (u, v), vertex u appears before v. It represents a valid execution order — build B before A if A depends on B.

### Kahn's Algorithm (BFS-Based)

```python
from collections import deque

def topological_sort_kahn(graph, all_nodes):
    """Kahn's algorithm: iteratively remove zero in-degree nodes."""
    # Compute in-degrees
    in_degree = {node: 0 for node in all_nodes}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    # Start with all zero in-degree nodes
    queue = deque([n for n in all_nodes if in_degree[n] == 0])
    order = []
    
    while queue:
        node = queue.popleft()
        order.append(node)
        
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # If order doesn't contain all nodes, a cycle exists
    if len(order) != len(all_nodes):
        raise ValueError("Graph has a cycle — topological sort impossible")
    
    return order
```

### Application: Build Systems & Dependency Resolution

Consider a software build: Module D depends on B and C, B depends on A, C depends on A. A valid build order is: A → B → C → D (or A → C → B → D). Topological sort computes this automatically.

**Cycle detection as a side effect:** If Kahn's algorithm terminates without processing all nodes, the remaining nodes form a cycle. This is how build systems detect circular dependencies — if `npm install` says "circular dependency detected," it's running a topological sort that failed.

---

## 8.5 Shortest Paths

### Dijkstra's Algorithm

For graphs with non-negative edge weights, Dijkstra's algorithm finds the shortest path from a source to all other vertices.

```python
import heapq

def dijkstra(graph, start):
    """Dijkstra's shortest path using min-heap priority queue."""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    previous = {}
    
    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)
        
        # Skip if we already found a shorter path
        if current_dist > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node]:
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances, previous
```

### Complexity

- **Time:** O((V + E) log V) with a binary heap. The log V factor comes from heap operations.
- **Space:** O(V) for the distance and previous arrays.

### Greedy Property

Dijkstra is greedy — it always expands the closest unexpanded node. This works because edge weights are non-negative, so no later path can improve on an already-settled node. If negative weights exist, use Bellman-Ford (O(VE)) instead.

### A* Search (Preview — Covered in Chapter 10)

A* extends Dijkstra with a heuristic function h(n) that estimates the remaining cost to the goal. It expands nodes in order of f(n) = g(n) + h(n), where g(n) is the known cost from start to n. When h(n) is admissible (never overestimates), A* is optimal and typically explores far fewer nodes than Dijkstra.

---

## 8.6 Dependency Graphs in Software Systems

### What Is a Dependency Graph?

A dependency graph is a directed graph where nodes represent software components (modules, services, packages, signals) and edges represent data flow or dependency relationships. An edge from B to A means "A depends on B" or equivalently "B feeds A."

### Why They Matter

- **Impact analysis:** If Module X changes, which downstream modules are affected?
- **Root-cause analysis:** Signal Z failed — which upstream modules could be responsible?
- **Build ordering:** In what sequence should modules be compiled?
- **Dead code detection:** Modules with zero out-degree and zero in-degree are orphans.

### Static vs. Dynamic Analysis

**Static analysis** extracts dependencies from source code without executing it. You parse imports, function calls, data flow declarations. It's complete (finds all *possible* dependencies) but imprecise (includes paths that never execute at runtime).

**Dynamic analysis** observes dependencies at runtime through tracing, profiling, or instrumentation. It's precise (only shows *actual* dependencies) but incomplete (misses paths not exercised in the current run).

For automotive software at Mercedes, static analysis was the right choice: the auto-generated code's dependency structure is deterministic and fully expressed in the source. Runtime tracing would have required executing the embedded software on hardware-in-the-loop simulators — impractical for 1,128 modules.

### Scale Considerations

Real-world dependency graphs are massive. The Linux kernel has 30,000+ files with millions of include relationships. A microservices architecture at a large company may have 5,000+ services with complex inter-dependencies. The iFAST system handles 1,128 modules with 1.6M+ edges — large enough to require efficient algorithms but small enough for a single machine.

---

## 8.7 NetworkX in Python

NetworkX is the standard Python library for graph construction and analysis. It's pure Python (slow for massive graphs but excellent for prototyping and moderate scale).

### Creating and Querying Graphs

```python
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add edges (implicitly creates nodes)
G.add_edge("BrakeModule", "StabilityControl", signal="SIG_BRK_PRESSURE")
G.add_edge("WheelSpeed", "StabilityControl", signal="SIG_WHEEL_RPM")
G.add_edge("StabilityControl", "Dashboard", signal="SIG_ESC_ACTIVE")

# Query: who are the ancestors of Dashboard?
ancestors = nx.ancestors(G, "Dashboard")
# {'BrakeModule', 'WheelSpeed', 'StabilityControl'}

# Shortest path
path = nx.shortest_path(G, "BrakeModule", "Dashboard")
# ['BrakeModule', 'StabilityControl', 'Dashboard']

# Topological sort
order = list(nx.topological_sort(G))
# ['BrakeModule', 'WheelSpeed', 'StabilityControl', 'Dashboard']

# Cycle detection
try:
    cycle = nx.find_cycle(G)
except nx.NetworkXNoCycle:
    print("No cycles — valid DAG")
```

### Performance Notes

NetworkX is pure Python and uses dictionaries of dictionaries internally. For graphs up to ~100K edges, it performs well. Beyond that:

| Library | Language | Speed vs. NetworkX | Best For |
|---------|----------|-------------------|----------|
| NetworkX | Python | 1× (baseline) | Prototyping, algorithms |
| graph-tool | C++ (Python bindings) | 10-100× | Large static graphs |
| rustworkx | Rust (Python bindings) | 10-50× | Drop-in NetworkX replacement |
| igraph | C (Python bindings) | 10-50× | Community detection, analysis |

For the iFAST engine with 1.6M edges, NetworkX was adequate because graph construction happens once and queries are simple BFS traversals. If the engine required repeated full-graph algorithms (e.g., betweenness centrality on every query), a compiled backend would be necessary.

---

## 8.8 Case Study: iFAST Dependency Engine

### The Problem

Mercedes-Benz's autonomous driving platform runs on three Electronic Control Units (ECUs), each executing hundreds of software modules. These modules communicate via **data signals** — one module publishes a signal, others subscribe. The complete system comprises:

- **1,128 software modules** across 3 ECUs
- **67,655 unique data signals**
- **1,655,171 dependency relationships**

When a signal fails during testing, engineers must trace upstream: "Which modules could have caused this failure?" The existing process relied on Movetto, a GUI-based tool that exported static SVG diagrams. These diagrams were:

1. **Stale** — generated from old code versions, not the current branch
2. **Incomplete** — 35% of signals were missing from the diagrams entirely
3. **Slow** — manual export took 5-30 minutes per query
4. **Non-queryable** — an SVG image cannot answer "show me all paths of length 3"

### The Solution: Automated Graph Extraction

The key insight: Mercedes uses TargetLink (a Simulink code generator) to produce C source code. This auto-generated code follows perfectly regular naming conventions:

```c
// Output signal: Rte_IWrite_{Module}_P_{Signal}_{Signal}
Rte_IWrite_BrakeCtrl_P_BrkPressure_BrkPressure(value);

// Input signal: Rte_IRead_{Module}_R_{Signal}_{Signal}
float pressure = Rte_IRead_StabilityCtrl_R_BrkPressure_BrkPressure();
```

Because the naming is machine-generated and perfectly regular, regex is not a hack — it's the mathematically correct parser for a regular language.

### Extraction Pipeline

```python
import re
from pathlib import Path
from collections import defaultdict

# Patterns for auto-generated signal access functions
PATTERNS = {
    "output": re.compile(r"Rte_IWrite_(\w+?)_P_(\w+?)_\w+"),
    "input": re.compile(r"Rte_IRead_(\w+?)_R_(\w+?)_\w+"),
    "config": re.compile(r"Rte_CData_(\w+?)_(\w+)"),
    "internal": re.compile(r"Rte_IrvIRead_(\w+?)_(\w+)")
}

def extract_signals(source_dir: Path) -> dict:
    """Scan all .c files and extract module-signal relationships."""
    producers = defaultdict(set)  # signal → set of producing modules
    consumers = defaultdict(set)  # signal → set of consuming modules
    
    for c_file in source_dir.rglob("*.c"):
        content = c_file.read_text(errors="ignore")
        
        for match in PATTERNS["output"].finditer(content):
            module, signal = match.group(1), match.group(2)
            producers[signal].add(module)
        
        for match in PATTERNS["input"].finditer(content):
            module, signal = match.group(1), match.group(2)
            consumers[signal].add(module)
    
    return producers, consumers
```

### Graph Construction

```python
import networkx as nx

def build_dependency_graph(producers, consumers):
    """Build directed graph: producer → consumer for each shared signal."""
    G = nx.DiGraph()
    
    for signal in producers:
        if signal in consumers:
            for producer_module in producers[signal]:
                for consumer_module in consumers[signal]:
                    G.add_edge(
                        producer_module, 
                        consumer_module,
                        signal=signal,
                        type="data_flow"
                    )
    
    return G
```

### Upstream Resolution (BFS)

The core query: given a failed signal, find all upstream modules that could be the root cause.

```python
from collections import deque

def find_upstream_ancestors(graph, target_signal, max_depth=None):
    """BFS to find all upstream modules that feed this signal.
    
    Returns list of (module, depth) tuples sorted by proximity.
    Depth = 1 means direct producer; depth = 2 means the producer's
    producer, etc.
    """
    queue = deque([(target_signal, 0)])
    visited = set()
    ancestors = []
    
    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        if max_depth and depth > max_depth:
            continue
        visited.add(node)
        ancestors.append((node, depth))
        
        for predecessor in graph.predecessors(node):
            queue.append((predecessor, depth + 1))
    
    return sorted(ancestors, key=lambda x: x[1])
```

### Version-Specific Builds

Each software release is tagged in Git. The engine takes a tag as input, checks out that version, parses the source, and builds the graph. This means:

- Every query is answered against the **exact code version** under test
- Graphs are deterministic (same tag always produces same graph) and cached permanently
- **Tag comparison:** diff two versions to see which signals were added, removed, or rewired

```python
def compare_versions(graph_v1, graph_v2):
    """Identify dependency changes between two software versions."""
    edges_v1 = set(graph_v1.edges(data=True))
    edges_v2 = set(graph_v2.edges(data=True))
    
    added = edges_v2 - edges_v1
    removed = edges_v1 - edges_v2
    
    return {"added": added, "removed": removed}
```

### Results

| Metric | Before (Movetto) | After (iFAST Engine) |
|--------|-------------------|---------------------|
| Signal coverage | 65% | 100% |
| Query time | 5-30 minutes | 86 seconds (full build) |
| Accuracy | Unknown (no validation) | 100% (10/10 benchmark) |
| Staleness | Days to weeks | Zero (built from current code) |
| Queryable | No (static SVG) | Yes (programmatic API) |

The benchmark validated 10 signals across 3 ECU modules against manually verified ground truth — the engine matched perfectly on all upstream resolution queries.

### Key Design Decision: Why Regex, Not AST Parsing?

The C code is **auto-generated** by TargetLink. It follows a perfectly regular naming pattern with zero variation. In formal language theory, the signal declarations form a **regular language** — parseable by a finite automaton (regex). An AST parser (which handles context-free grammars) would be:

1. **Over-engineered** — using a more powerful tool than the problem requires
2. **Slower** — parsing full C syntax to extract only function names
3. **More fragile** — C parsers must handle preprocessor directives, includes, and platform-specific extensions

The regex approach processes all 1,128 modules across 3 repositories in 86 seconds. An AST-based approach would conservatively take 10-20x longer with no accuracy benefit.

---

## 8.9 Summary & Interview Tips

### Core Algorithms to Know Cold

| Algorithm | Time | Space | Key Application |
|-----------|------|-------|-----------------|
| BFS | O(V+E) | O(V) | Shortest unweighted path, level traversal |
| DFS | O(V+E) | O(V) | Cycle detection, topological sort |
| Topological Sort | O(V+E) | O(V) | Build order, dependency resolution |
| Dijkstra | O((V+E)logV) | O(V) | Shortest weighted path (non-negative) |

### Interview Patterns

**"Tell me about a time you worked with large-scale data."**
"I built a dependency graph engine at Mercedes that processes 1,128 modules and 1.6M+ dependency edges. The system extracts the complete dependency graph from auto-generated C source code using regex-based static analysis, then resolves upstream dependencies via BFS. It achieved 100% accuracy on benchmark validation and replaced a manual process that was 35% incomplete."

**"How would you detect circular dependencies?"**
"Run topological sort using Kahn's algorithm. If the algorithm terminates without processing all nodes, the remaining nodes participate in a cycle. Alternatively, during DFS, maintain a 'currently in stack' set — if you encounter a node already in the stack, you've found a back edge, which proves a cycle exists."

**"How would you handle a graph with millions of edges?"**
"For construction and simple traversals (BFS/DFS), even Python handles millions of edges in seconds — these are O(V+E) algorithms. For expensive algorithms (betweenness centrality, PageRank), I'd use graph-tool or rustworkx which are 10-100x faster than NetworkX. For graphs that don't fit in memory, I'd use a distributed framework like Apache Giraph or store the graph in a database like Neo4j."

**"Why BFS over DFS for dependency resolution?"**
"BFS naturally gives you results ordered by distance — direct dependencies first, transitive dependencies later. This matches what engineers want: 'show me the most likely root causes, starting with the closest upstream modules.' DFS might explore a deep chain of 50 modules before showing you the direct neighbor that's the actual culprit."

### Key Takeaway

Graph algorithms are not just theoretical — they solve real engineering problems at scale. The iFAST dependency engine demonstrates that combining simple algorithms (BFS, regex parsing) with domain insight (auto-generated code has regular structure) can outperform complex commercial tools. The system processes 1.6M+ edges in 86 seconds, achieves 100% accuracy, and eliminated hours of manual work per week for the engineering team.
