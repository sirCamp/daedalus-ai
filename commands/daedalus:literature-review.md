---
name: daedalus:literature-review
description: Generate a structured literature review for the current research
user_invocable: true
---

# Literature Review

Generate a structured literature review, find relevant papers, and build the library.

## IMPORTANT: Use Daedalus MCP tools ONLY

Do NOT read or search Daedalus source code. Do NOT use Read/Grep/Glob to look at library.py or paper.py.
Instead, use ONLY the following MCP tools to interact with the paper library:

- `daedalus_literature_review` — generate the review (auto-searches if library is empty)
- `daedalus_add_paper` — add a paper to the project's library by arxiv_id
- `daedalus_get_paper_details` — get full details of a paper
- `daedalus_search_papers` — search Semantic Scholar
- `daedalus_list_papers` — list papers in the library
- `daedalus_save_note` — save literature insights to memory

## Steps

1. Call `daedalus_literature_review` with `include_search: true`
   - If user specified a focus topic, pass it as `focus`
   - If the library is empty, the tool auto-searches based on experiment hypotheses
2. Present the review sections to the user:
   - Papers in library with key findings
   - Papers vs experiments — which papers support which outcomes
   - **Gaps** — experiments without paper backing, rejected hypotheses needing literature
   - **New papers** — recently found papers not yet in the library
3. **For each relevant new paper found**, call `daedalus_add_paper` with:
   - `arxiv_id` — the paper's arXiv ID
   - `relevance_note` — why it matters for this research
   - `key_findings` — bullet points from the abstract
   - `tags` — matching the research focus
4. If the user wants details on a specific paper, use `daedalus_get_paper_details`
5. Save key literature insights with `daedalus_save_note` (category: "insight")

## Gap Analysis Output

The `literature_review` tool produces a structured gap analysis:

- **Unsupported experiments**: experiments whose hypotheses have no paper backing — consider searching for supporting or contradicting work
- **Rejected hypotheses**: failed experiments that may need literature review to understand why
- **Missing baselines**: standard approaches from papers that haven't been tried yet
- **Technique gaps**: methods mentioned in papers that could apply to your task but haven't been explored

Use gap analysis to inform the research plan: add new plan steps for unexplored techniques, or search for papers that explain failed experiments.

## Important

- ALWAYS use `include_search: true` — the whole point is to find AND save papers
- After finding papers, actually ADD the relevant ones — don't just list them
- If the library was empty, say so and explain you're bootstrapping it
- Save literature insights to memory — they inform future experiments

## Usage

```
/daedalus:literature-review                    # Broad review + auto-search
/daedalus:literature-review learning rate      # Focus on learning rate papers
/daedalus:literature-review reward design      # Focus on reward design
```
