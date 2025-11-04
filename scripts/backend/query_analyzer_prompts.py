# ------------------ #
# Content Extraction #
# ------------------ #
_content_extraction_prompt_tmpl = """
# Task Definition

Given a query for finding papers about a specific topic, extract only the *content* of the query, ignoring all *metadata*. Metadata is defined to be either of the following:

* Author/Coauthor name(s)
* Year(s), or words that describe time, such as "recent", "latest"
* Words that describe the impact of the paper, such as "central", "seminal", "influential"
* Venues, such as ACL, EMNLP, AAAI
* Words that describe how the search should be carried out, such as "run an exhaustive search on..."

## Rules

* If the query contains phrases such as "papers using", "papers proposing", "survey on", keep them as part of the content. However, "papers about" and "papers on" can be ignored.
* If the query is in the form of a question, extract a coherent representation of the question that focuses on the content of the question.
* Queries that only contain metadata as described above should return `null`.
* If you're unsure what the content is, return the original query as-is. If there is no content, return `null`.

The return format should be a JSON object that looks like {"content": ...}

# Examples

{"query": "Graph-based Neural Multi-Document Summarization Yasunaga et al., 2017"}
{"content": "Graph-based Neural Multi-Document Summarization"}
Reason: "Yasunaga et al." is author metadata, and 2017 is time metadata

{"query": "classic or early papers on pretrained transformer models"}
{"content": "pretrained transformer models"}
Reason: "classic" is impact metadata, and "early" is time metadata

{"query": "good paper about CRISPR gene editing"}
{"content": "CRISPR gene editing"}
Reason: The word "good" doesn't modify the content and is inconsequential for the query

{"query": "papers about LLM chains"}
{"content": "LLM chains"}
Reason: Since the query is already for finding papers, the prefix "papers about" is redundant

{"query": "multi document summarization methods"}
{"content": "multi document summarization methods"}
Reason: Every word is essential to understand the query, and there's no metadata at all

{"query": "latest research on using annotation disagreements in classification models"}
{"content": "using annotation disagreements in classification models"}
Reason: "latest research" is time metadata

{"query": "papers from ICLR 2024"}
{"content": ""}
Reason: The query consists of metadata only"""


_author_extraction_prompt_tmpl = """
# Task Definition

Given a query for finding papers, identify the authors whose papers are being requested.
Extract only the *author(s)* names from the query, ignoring all other information.

The return format should be a JSON object that looks like {"authors": [...]}.
If there are no authors' names requested in the query, return {"authors": []}.

# Examples

{"query": "Graph-based Neural Multi-Document Summarization Yasunaga et al., 2017"}
{"authors": ["Yasunaga"]}
Reason: The "et al." suffix does not provide information on the authors

{"query": "papers on planning by Dan Weld"}
{"authors": ["Dan Weld"]}

{"query": "papers about transformer models by Google"}
{"authors": []}
Reason: "Google" is an organization, not an author name. The query does not require papers from a specific author.

{"query": "papers by author with scopus ID 123456789"}
{"authors": []}
Reason: The query does not require papers from a specific author's name, but rather from a specific author ID.

{"query": "papers on LLM chains"}
{"authors": []}
Reason: The query does not require papers from a specific author.

{"query": "papers discussing Henry David Thoreau's ideas about nature"}
{"authors": []}
Reason: The author mentioned is not requested to be the author of the paper, but rather a person whose ideas should be discussed in the paper.
"""


_venue_extraction_prompt_tmpl = """
# Task Definition

Given a query for finding papers, what venue(s) does it require the papers to be from?
Extract only the *venue(s)* required in the query, ignoring all other information.

The return format should be a JSON object that looks like {"venues": [...]}. If there are no required venues in the query, return {"venues": []}.

# Examples

{"query": "Large Language Models can Strategically Deceive their Users when Put Under Pressure, ICLR 2024"}
{"venues": ["ICLR"]}

{"query": "papers presented at either ICLR or AAAI"}
{"venues": ["ICLR", "AAAI"]}

{"query": "papers that evaluate on the CoNLL-2003 benchmark"}
{"venues": []}
Reason: The query does not require papers from a specific venue. CoNLL-2003 in this case is NOT a required venue.

{"query": "ACL papers on transformers"}
{"venues": ["ACL", "EMNLP", "NAACL", "COLING", "EACL", "TACL", "CL", "LREC", "AACL", "CoNLL", "*SEM"]}
Reason: ACL may refer to a collection of venues, try to provide a comprehensive list of specific venues if possible.
"""



_time_range_prompt_tmpl = """
# Task Definition

The current year is 2025. Given a query for finding papers about a specific topic, extract the time range mentioned in the query, if it exists. Only extract explicit mentions of time ranges.

Return a JSON object in the format: {"start": ..., "end": ...}. If neither field exists, return {"start": null, "end": null}.

# Examples

{"query": "recent papers using Earth Mover's Distance (EMD) as an evaluation metric"}
{"start": null, "end": null}

{"query": "synthesizing answers to scientific questions from search or ranker result snippets or documents, multi-document answers synthesis, last 3 years"}
{"start": 2023, "end": 2025}
Reason: the last 3 years are 2025, 2024, and 2023

{"query": "research on persona-assigned Large Language Models published in 2024"}
{"start": 2024, "end": 2024}"""
