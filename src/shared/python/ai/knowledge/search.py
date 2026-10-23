"""Private BM25 search implementation for knowledge packs."""

from __future__ import annotations

import re
from contextlib import closing
from typing import TYPE_CHECKING

from .pack import Passage, _hidden_clause, _passage

if TYPE_CHECKING:
    from .pack import KnowledgePack

_TITLE_WEIGHT = 5.0
_TEXT_WEIGHT = 1.0
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_MAX_QUERY_TOKENS = 32


def search_pack(
    pack: KnowledgePack,
    query: str,
    k: int = 8,
    include_superseded: bool = False,
) -> list[Passage]:
    """Search by weighted BM25, with authority and passage id breaking ties."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    tokens = _TOKEN_RE.findall(query)[:_MAX_QUERY_TOKENS]
    if not tokens:
        return []
    match = " OR ".join('"' + token.replace('"', "") + '"' for token in tokens)
    hidden = "" if include_superseded else _hidden_clause()
    sql = (
        "SELECT p.repo, p.source, p.anchor, p.title, p.text, p.commit_sha,"
        " p.content_hash, p.status, p.authority,"
        f" bm25(passages_fts, {_TITLE_WEIGHT}, {_TEXT_WEIGHT}) AS score"
        " FROM passages_fts JOIN passages p ON p.id = passages_fts.rowid"
        f" WHERE passages_fts MATCH ?{hidden}"
        " ORDER BY score, p.authority_rank, p.id LIMIT ?"
    )
    with closing(pack._connect()) as conn:
        rows = conn.execute(sql, (match, k)).fetchall()
    return [_passage(row) for row in rows]
