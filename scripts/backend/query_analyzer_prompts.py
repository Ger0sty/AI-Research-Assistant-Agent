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


_recency_extraction_prompt_tmpl = """
# Task Definition

Given a query for finding papers about a specific topic, decide whether the query explicitly asks for recent papers or early papers, ignoring all other information.

Do not assume recency based on absolute years.

If the query asks for most recent papers, return the JSON object {"prefer": "recent"}, if the query asks for early papers, return the JSON object {"prefer": "early"}, and otherwise return {"prefer": null}.

# Examples

{"query": "papers that have referenced 'Attention is All You Need' recently"}
{"prefer": "recent"}

{"query": "latest papers about claim verification"}
{"prefer": "recent"}

{"query": "earlier papers on seq2seq"}
{"prefer": "early"}

{"query": "survey on multi-agent collaboration in AI and HCI"}
{"prefer": null}
"""  # noqa: E501

_centrality_extraction_prompt_tmpl = """
# Task Definition

Given a query for finding papers about a specific topic, decide whether the query asks for central papers or less cited papers, ignoring all other information.

A query asks for a central paper if it uses words like "central", "seminal", "impactful", "highly influential", "highly cited", etc.

A query asks for a less cited paper if it uses words like "less cited", "lesser known", etc.

If the query asks for central papers, return the JSON object {"centrality": "first"}, if the query asks for less cited papers, return the JSON object {"centrality": "last"}, otherwise return {"centrality": null}.

# Examples

{"query": "most important references on counterfactual data augmentation (CDA)"}
{"centrality": "first"}

{"query": "top papers in AI for Earth (environmental AI)"}
{"centrality": "first"}

{"query": "least cited papers on transformers"}
{"centrality": "last"}

{"query": "papers on LSTMs that are the least cited"}
{"centrality": "last"}

{"query": "paper on weather"}
{"centrality": null}"""  # noqa: E501


_broad_or_specific_query_type_prompt_tmpl = """
# Task Definition

You are given a user's query aimed at finding academic papers. Your goal is to determine the nature of the query based on the following criteria:

- ** unique-identifier **: The user knows the exact paper they are looking for and provides a unique identifier (the paper title or another unique name that uniquely identifies the paper).

- ** descriptions-or-keywords **: The user is searching for papers using a description, keywords, or topics. The user does not provide a unique identifier for the paper. The query may contain named entities, but they do not uniquely identify a specific paper the user is looking for.

If the query is searching by unique-identifier, return the JSON object {"type": "unique-identifier"}, otherwise return {"type": "descriptions-or-keywords"}.

# Examples

{"query": "llm hallucinations"}
{"type": "descriptions-or-keywords"}

{"query": "the snli paper"}
{"type": "unique-identifier"}

{"query": "Attention is All You Need"}
{"type": "unique-identifier"}

{"query": "GLUE paper about the evaluation of natural language understanding systems"}
{"type": "unique-identifier"}
Reason: The query is looking for a unique-identifier paper by name, and provides a description for extra context

{"query": "pretrained large language models"}
{"type": "descriptions-or-keywords"}

{"query": "paper showing that transformers are better than LSTMs"}
{"type": "descriptions-or-keywords"}

{"query": "the first paper that evaluated the performance of transformers on the GLUE benchmark"}
{"type": "descriptions-or-keywords"}
Reason: The query is looking for a specific paper, but the user does not provide a unique identifier. None of the names provided ("GLUE", "transformer") uniquely identify the paper the user is looking for.
"""



_by_title_or_name_query_type_prompt_tmpl = """
# Task Definition

Given a query for finding a specific paper, decide whether the query is looking for a paper by its title, or by some key features.

If the query is looking for a paper by name, return the JSON object {"type": "title"}, otherwise return {"type": "name"}. If unsure, return {"type": "name"}.

# Examples

{"query": "Attention is All You Need"}
{"type": "title"}

{"query": "BioBERT: a pre-trained biomedical language representation model for biomedical text mining"}
{"type": "title"}

{"query": "the snli paper"}
{"type": "name"}

{"query": "LEGOBench dataset"}
{"type": "name"}"""  # noqa: E501



