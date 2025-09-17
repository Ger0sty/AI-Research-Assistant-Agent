from typing import TypedDict

from ai2i.chain import define_prompt_llm_call
from pydantic import BaseModel

# ------------ #
# Broad Search #
# ------------ #

_broad_search_prompt_tmpl = """
Given a user-provided natural language description of desired scientific papers,
reformulate it into a concise keyword-style search query suitable for our internal paper
search (SQL/FTS). Focus on terms likely to appear in the title, abstract, or body text.
Avoid metadata terms (e.g., venue names, generic phrases like “recent work”) unless the
user explicitly asks for them.

Guidelines:
- Prefer content-bearing keywords and short key phrases.
- You may include quoted phrases if they are important (e.g., "contrastive learning").
- Avoid special boolean operators or engine-specific syntax (AND/OR/NOT, field prefixes).
- Keep it plain text and concise; no explanations.

Input description: ```{paper_description}```
"""

class FormulateBroadSearchQueryInput(TypedDict):
    paper_description: str

class BroadSearchSuggestedQueries(BaseModel):
    keyword_query: str

broad_search = (
    define_prompt_llm_call(
        _broad_search_prompt_tmpl,
        input_type=FormulateBroadSearchQueryInput,
        output_type=BroadSearchSuggestedQueries,
        custom_format_instructions=(
            'Return a JSON dict with the key "keyword_query" and the value the query reformulated as a keyword query.'
        ),
    )
    .map(lambda o: o.keyword_query)
    .contra_map(lambda s: {"paper_description": s}, input_type=str)
)