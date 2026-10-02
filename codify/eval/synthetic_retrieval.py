"""Score the lexical arm over the synthetic corpora, on a checkout with no key.

Needs `POSTGRES_URL` pointing at a disposable, migrated database.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from codify.eval.metrics import evaluate, mean_scores

# Relative to this file, so the corpora travel with the package that reads them.
_FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
GOLD = _FIXTURES / "eval" / "synthetic-retrieval-gold.json"
CORPORA = (_FIXTURES / "synthetic",)
K = 50
_ROWS = 500  # Provisions fetched, so K articles survive the collapse.

_LEXICAL = text(
    """
    SELECT id FROM provisions
    WHERE version_id = ANY(:version_ids)
      AND search_tsv @@ to_tsquery('simple', :query_match)
    ORDER BY search_tokens <@> to_bm25query(:query_tokens, 'provisions_bm25_idx')
    LIMIT :k
    """
)


async def _lexical(session: AsyncSession, query: str, vids: list[uuid.UUID]) -> list[str]:
    """The product's arm: the search tokeniser, then BM25 within one law."""
    from codify.storage.retrieval import query_tokens_for

    tokens = (await query_tokens_for(session, query, vids)).split()
    if not tokens:
        return []
    rows = await session.execute(
        _LEXICAL,
        {
            "version_ids": vids,
            "query_tokens": " ".join(tokens),
            "query_match": " | ".join(tokens),
            "k": _ROWS,
        },
    )
    return [str(r[0]) for r in rows.all()]


def _manifests() -> list[tuple[Path, dict[str, Any]]]:
    if not CORPORA[0].exists():
        # The wheel carries this module and not the corpora beside the tests.
        raise SystemExit(
            f"no corpora at {CORPORA[0]}: this eval needs a source checkout, not an installed wheel"
        )
    out = []
    for root in CORPORA:
        for manifest in sorted(root.glob("*/manifest.json")):
            out.append((manifest.parent, json.loads(manifest.read_text(encoding="utf-8"))))
    return out


async def _work_exists(session: AsyncSession, work_uri: str) -> bool:
    return bool(
        (
            await session.execute(
                text("SELECT 1 FROM laws WHERE frbr_work_uri = :u"), {"u": work_uri}
            )
        ).scalar()
    )


async def _ingest(session: AsyncSession) -> dict[str, uuid.UUID]:
    """Parse every shipped corpus into the database. Returns work URI to version id."""
    from codify.akn.io import parse_akn
    from codify.jurisdictions import load_config
    from codify.pipeline.enrich.bluebell import parse_to_akn
    from codify.storage.repository import save_document

    versions: dict[str, uuid.UUID] = {}
    reused: list[str] = []
    for directory, manifest in _manifests():
        code = manifest["jurisdiction"]
        config = load_config(code)
        if not config.synthetic:
            # Writes only where the jurisdiction declares its corpus invented.
            raise SystemExit(f"{code} is not a synthetic jurisdiction; refusing to ingest")
        for law in manifest["laws"]:
            akn = parse_to_akn(
                (directory / law["file"]).read_text(encoding="utf-8"),
                code,
                doctype=law.get("doctype", "act"),
                date=law["date"][:4],
                number=law["number"],
                language=law["language"],
            )
            doc = parse_akn(akn)
            if await _work_exists(session, doc.frbr_work_uri):
                # `stored` says the expression was inserted, not that the law
                # was. Cleaning up an existing law takes its other expressions.
                reused.append(doc.frbr_work_uri)
                continue
            version_id = await save_document(
                session,
                doc,
                jurisdiction_code=code,
                law_title=law.get("short_title") or law["file"],
                akn_xml=akn,
                year=int(law["date"][:4]),
                number=law["number"],
            )
            if doc.frbr_work_uri in versions:
                raise SystemExit(f"two manifests claim {doc.frbr_work_uri}")
            versions[doc.frbr_work_uri] = version_id
    if reused:
        raise SystemExit(
            f"{len(reused)} of these works are already in this database, so a run "
            "would score rows it did not write. Delete them and run again."
        )
    await session.flush()
    return versions


async def _eids(session: AsyncSession, ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    rows = (
        await session.execute(
            text("SELECT id, akn_eid FROM provisions WHERE id = ANY(:ids)"),
            {"ids": [uuid.UUID(i) for i in ids]},
        )
    ).all()
    return {str(r[0]): str(r[1]) for r in rows}


def _article_of(eid: str) -> str:
    """The eId trimmed to its article or section, which is what a query wants."""
    parts = eid.split("__")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith(("art_", "sec_")):
            return "__".join(parts[: i + 1])
    return eid


def _collapse(ranked: list[str], eids: dict[str, str]) -> list[str]:
    """Rank articles, not paragraphs.

    Paragraphs would divide recall, and let one article hold several positions.
    """
    seen: set[str] = set()
    out: list[str] = []
    for pid in ranked:
        key = _article_of(eids.get(pid, pid))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out[:K]


async def _resolve(
    session: AsyncSession, versions: dict[str, uuid.UUID], queries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """eIds to provision ids, dropping a missing target loudly rather than
    scoring it as a miss and blaming the arm for a stale fixture."""
    resolved, missing = [], []
    for q in queries:
        version_id = versions.get(q["work_uri"])
        if version_id is None:
            missing.append(f"{q['id']}: no version for {q['work_uri']}")
            continue
        # A target names an article; its rows are the leaf paragraphs beneath.
        found: dict[str, list[str]] = {}
        for eid in q["relevant"]:
            rows = (
                await session.execute(
                    text(
                        "SELECT id FROM provisions WHERE version_id = :v "
                        "AND (akn_eid = :eid OR starts_with(akn_eid, :eid || '__'))"
                    ),
                    {"v": version_id, "eid": eid},
                )
            ).all()
            if rows:
                found[eid] = [str(r[0]) for r in rows]
        absent = [e for e in q["relevant"] if e not in found]
        if absent:
            missing.append(f"{q['id']}: {q['work_uri']} has no {', '.join(absent)}")
            continue
        resolved.append({**q, "version_id": version_id, "targets": sorted(found)})
    for line in missing:
        print(f"  dropped {line}")
    return resolved


async def run(keep: bool) -> int:
    manifests = _manifests()  # Checks the corpora are here before anything else reads.
    spec = json.loads(GOLD.read_text(encoding="utf-8"))
    url = os.environ["POSTGRES_URL"].replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    codes = {m["jurisdiction"] for _, m in manifests}

    try:
        async with factory() as session:
            versions = await _ingest(session)
            print(f"ingested {len(versions)} laws across {len(codes)} synthetic jurisdictions")
            gold = await _resolve(session, versions, spec["queries"])
            if not gold:
                print("no scorable queries")
                return 1

            per_language: dict[str, list[dict[str, float]]] = defaultdict(list)
            for q in gold:
                ranked = await _lexical(session, q["query"], [q["version_id"]])
                eids = await _eids(session, ranked)
                scores = evaluate(_collapse(ranked, eids), set(q["targets"]))
                per_language[q["language"]].append(scores)
                print(
                    f"  {q['id']:<26} {q['language']}  "
                    + "  ".join(f"{k}={v:.2f}" for k, v in scores.items())
                )

            print(f"\nlexical arm, k={K}, {len(gold)} queries")
            for language in sorted(per_language):
                rows = per_language[language]
                means = mean_scores(rows)
                print(
                    f"  {language}  n={len(rows):<3} "
                    + "  ".join(f"{k}={v:.3f}" for k, v in means.items())
                )
            # Mean of the per-language means, matching the private runner: 18
            # English queries would otherwise outweigh three in another language.
            overall = mean_scores([mean_scores(rows) for rows in per_language.values()])
            print("  " + "  ".join(f"{k}={v:.3f}" for k, v in overall.items()) + "  (all)")
            if keep:
                await session.commit()
            else:
                # Nothing is committed: no cleanup to get wrong, and no rows
                # left in `search_terms`, which has no cascade from `laws`.
                await session.rollback()
    finally:
        await engine.dispose()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="commit the ingested corpora")
    raise SystemExit(asyncio.run(run(ap.parse_args().keep)))


if __name__ == "__main__":
    main()
