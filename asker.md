You are a senior software engineer tasked with generating a code retrieval dataset from a given repository.

## Your task

Given a repository, you will:
1. Explore the repository structure and identify its key architectural layers and most important files
2. Generate exactly 30 meaningful questions about the codebase
3. For each question, provide the exact code snippet from the repository that answers it

## Step 1 — Explore the repository

Before generating questions, thoroughly read the repository:
- Look at the top-level directory structure
- Read the README
- Identify the main entry points, core modules, and key abstractions
- Note the architectural layers (e.g. API layer, data models, business logic, utilities, configuration, tests)
- Identify the 10–15 most important files that define how the system works

## Step 2 — Generate questions

Generate 30 questions that a developer unfamiliar with the codebase would type into a search bar. Imagine someone who knows what they want to achieve but does not yet know where to look.

Good questions:
- Are phrased in terms of **intent, behavior, or outcome** — not in terms of the code that answers them
- **Never name the specific function, class, method, or variable** that is the answer (that would give it away)
- Are specific to this repository's domain and purpose (not generic Python questions)
- Spread across different files and layers — do not cluster in one file
- Vary in scope: some target a single mechanism, others ask where two components connect
- Have a single, unambiguous code snippet as the answer

To check a question before including it: if reading the question alone tells you which function or class to look at, rewrite it.

**Bad** (names the answer, describes the implementation):
- "How does `apply_rules` translate a JAX tree path into a PartitionSpec?"
- "What is the three-phase structure of `Agent.step`?"
- "How does `call_llm` handle models that do not support JSON mode?"

**Good** (expresses intent, answer location is not obvious):
- "How does the model decide which device each parameter shard lives on?"
- "What happens between receiving a user message and calling the LLM?"
- "How does the system fall back when the primary language model is unavailable?"

Avoid questions that:
- Name any identifier from the codebase (function, class, variable, file)
- Are too trivial (e.g. "what does the import statement do")
- Require multiple disjoint snippets to answer
- Are answerable only by prose, not by code

## Step 3 — Output format

Output a JSON array of 30 objects. Each object must have the following fields:

```json
[
  {
    "question": "How does the application authenticate incoming API requests?",
    "file": "src/auth/middleware.py",
    "line_start": 42,
    "line_end": 61,
    "snippet": "<exact verbatim code from the file, preserving indentation>"
  },
  ...
]
```

Rules for snippets:
- Copy the code verbatim from the file — do not paraphrase or simplify
- Include enough context to be self-contained (full function or class body, not just one line)
- `line_start` and `line_end` must be accurate line numbers in the file
- Each snippet must come from a different location — do not reuse the same lines for two questions
