import itertools
import logging
import math
import unicodedata
from typing import Callable

from ai2i.dcollection import CitationContext, Document, Snippet

logger = logging.getLogger(__name__)


def find_relevant_snippet(doc: Document, relevant_snippet: str | None) -> list[Snippet | CitationContext] | None:
    if not relevant_snippet:
        return None
    try:
        return _find_relevant_snippet_in_doc(doc, relevant_snippet)
    except Exception as e:
        logger.exception(f"Failed to find relevant snippet: {e}")
        return [Snippet(text=relevant_snippet)]


def _find_relevant_snippet_in_doc(doc: Document, relevant_snippet: str) -> list[Snippet | CitationContext] | None:
    relevant_snippet_parts = [_render_relevant_snippet_text(part) for part in relevant_snippet.split(" ... ")]
    part_matches_aggregate: list[list[Snippet | CitationContext]] = [[] for _ in relevant_snippet_parts]

    # title
    part_matches_aggregate = _accumulate_snippet_matches(
        part_matches_aggregate,
        doc.title,
        relevant_snippet_parts,
        lambda p, s, e: Snippet(
            text=p, section_kind="title", section_title="title", char_start_offset=s, char_end_offset=e
        ),
    )

    # abstract
    part_matches_aggregate = _accumulate_snippet_matches(
        part_matches_aggregate,
        doc.abstract,
        relevant_snippet_parts,
        lambda p, s, e: Snippet(
            text=p, section_kind="abstract", section_title="abstract", char_start_offset=s, char_end_offset=e
        ),
    )

    # NEW: body text (SQL path often lacks precomputed snippets, so search the full text)
    part_matches_aggregate = _accumulate_snippet_matches(
        part_matches_aggregate,
        getattr(doc, "text", None),
        relevant_snippet_parts,
        lambda p, s, e: Snippet(
            text=p, section_kind="body", section_title="body", char_start_offset=s, char_end_offset=e
        ),
    )

    # existing precomputed snippets
    for snippet in doc.snippets or []:
        part_matches_aggregate = _accumulate_snippet_matches(
            part_matches_aggregate,
            snippet.text,
            relevant_snippet_parts,
            lambda p, s, e: Snippet(
                text=p,
                section_kind=snippet.section_kind,
                section_title=snippet.section_title,
                char_start_offset=(snippet.char_start_offset or 0) + s,
                char_end_offset=(snippet.char_end_offset or 0) + e,
            ) if isinstance(snippet, Snippet) and snippet.char_start_offset is not None else Snippet(text=p),
        )

    # citation contexts (keep, but give them finite spans for ranking)
    for context in doc.citation_contexts or []:
        part_matches_aggregate = _accumulate_snippet_matches(
            part_matches_aggregate,
            context.text,
            relevant_snippet_parts,
            # we don't have offsets on CitationContext; wrap as CitationContext but span optimizer
            # below will treat missing offsets as 0..len(text) instead of infinities.
            lambda p, s, e: CitationContext(text=p, source_corpus_id=context.source_corpus_id),
        )

    best_spans = _choose_min_combined_span(part_matches_aggregate, fallback_text_len=len(getattr(doc, "text", "") or ""))
    snippets: list[Snippet | CitationContext] = []
    for part, snippet in zip(relevant_snippet_parts, best_spans):
        snippets.append(snippet or Snippet(text=part))
    return snippets


def _render_relevant_snippet_text(part: str) -> str:
    return (
        part.replace("&quot;", '"')
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&amp;", "&")
        .replace("<<<", "")
        .replace(">>>", "")
    )


def _accumulate_snippet_matches(
    part_matches_aggregate: list[list[Snippet | CitationContext]],
    text: str | None,
    relevant_snippet_parts: list[str],
    factory: Callable[[str, int, int], Snippet | CitationContext],
) -> list[list[Snippet | CitationContext]]:
    """
    Reusable helper that finds snippet parts in a given text and accumulates them.
    The `factory` callback should produce a Snippet or CitationContext (or list thereof).
    """
    if not text:
        return part_matches_aggregate
    part_matches = _find_snippet_parts(text, relevant_snippet_parts)
    aggregated = []
    for existing_parts, (snippet_part, spans) in zip(part_matches_aggregate, zip(relevant_snippet_parts, part_matches)):
        new_parts = [factory(snippet_part, start, end) for start, end in spans]
        aggregated.append(existing_parts + new_parts)
    return aggregated


def _find_snippet_parts(text: str, relevant_snippet_parts: list[str]) -> list[list[tuple[int, int]]]:
    return [_fuzzy_find_snippet_text(text, part) for part in relevant_snippet_parts]


def _fuzzy_find_snippet_text(text: str, snippet_to_find: str) -> list[tuple[int, int]]:
    try:

        def remove_unicode_diacritics(s: str) -> str:
            return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

        if not text or not snippet_to_find:
            return []
        if not any(ch.isalnum() for ch in remove_unicode_diacritics(text)):
            return []
        if not any(ch.isalnum() for ch in remove_unicode_diacritics(snippet_to_find)):
            return []

        # Build a mapping from a 'cleaned' view to original text indices
        normalized_chars = []
        index_map = []
        for i, ch in enumerate(remove_unicode_diacritics(text)):
            if ch.isalnum():
                normalized_chars.append(ch.lower())
                index_map.append(i)

        snippet_cleaned = "".join(ch.lower() for ch in remove_unicode_diacritics(snippet_to_find) if ch.isalnum())
        normalized_text = "".join(normalized_chars)

        # Find all occurrences of snippet_cleaned in normalized_text
        matches = []
        search_start = 0
        while True:
            found_at = normalized_text.find(snippet_cleaned, search_start)
            if found_at == -1:
                break
            start_original = index_map[found_at]
            end_original = index_map[found_at + len(snippet_cleaned) - 1]
            matches.append((start_original, end_original + 1))
            search_start = found_at + 1
        return matches
    except Exception as e:
        logger.exception(f"Failed to find relevant snippet (text: {text}, snippet: {snippet_to_find}): {e}")
        return []


def _choose_min_combined_span(
    part_matches_aggregate: list[list[Snippet | CitationContext]],
    fallback_text_len: int = 0,
) -> list[Snippet | CitationContext | None]:
    parts_found = [bool(s) for s in part_matches_aggregate]
    if not any(parts_found):
        return [None] * len(part_matches_aggregate)

    combo_count = 1
    for s in part_matches_aggregate:
        combo_count *= max(1, len(s))
        if combo_count > 100_000:
            return [segments[0] if segments else None for segments in part_matches_aggregate]
    part_matches_aggregate = [s for s in part_matches_aggregate if s]

    best_combination = None
    best_span_size = math.inf

    for combo in itertools.product(*part_matches_aggregate):
        def _start(c):
            if isinstance(c, Snippet) and c.char_start_offset is not None:
                return c.char_start_offset
            return 0
        def _end(c):
            if isinstance(c, Snippet) and c.char_end_offset is not None:
                return c.char_end_offset
            # If we lack offsets (e.g., CitationContext), assume a small finite span
            return min(fallback_text_len, _start(c) + 200) if fallback_text_len else _start(c) + 200

        min_start = min(_start(c) for c in combo)
        max_end = max(_end(c) for c in combo)
        span_size = max_end - min_start

        if span_size < best_span_size:
            best_span_size = span_size
            best_combination = list(combo)

    if not best_combination:
        best_combination = [s[0] for s in part_matches_aggregate if s]

    results = []
    for part in parts_found:
        results.append(best_combination.pop(0) if part else None)
    return results
