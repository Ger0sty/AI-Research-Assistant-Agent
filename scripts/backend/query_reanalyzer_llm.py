from scripts.backend.llm_utils import call_llm_json
import json

REANALYZE_PROMPT = """
You are an intelligent query re-analyzer for a scientific paper retrieval system.

Your goal is to *refine or expand* the user's search query based on feedback.

Input:
- Previous query: "{old_query}"
- Previous analysis (structured as JSON): {old_analysis}
- User feedback: "{feedback}"

Instructions:
1. Understand what the user wants changed — e.g., focus area, author, recency, venue, etc.
2. Rewrite the query to reflect the new intent while preserving relevant context.
3. Be concise — no long sentences or explanations.
4. Return valid JSON only, in this format:

{
  "updated_query": "new query text ready for search",
  "changes": "one-sentence summary of what was updated"
}
"""

def reanalyze_query_llm(old_query: str, feedback: str, old_analysis: dict) -> dict:
    try:
        # Ensure analysis is valid JSON string for insertion
        analysis_str = json.dumps(old_analysis, indent=2)
        prompt = REANALYZE_PROMPT.format(
            old_query=old_query,
            old_analysis=analysis_str,
            feedback=feedback
        )
        return call_llm_json(prompt)
    except Exception as e:
        return {"error": f"Reanalysis failed: {str(e)}"}
