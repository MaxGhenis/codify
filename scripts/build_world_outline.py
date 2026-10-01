#!/usr/bin/env python3
"""Build a GeoJSON FeatureCollection with jurisdiction codes and profile tiers.

Usage:
    uv run python scripts/build_world_outline.py --output world-countries.geojson

Use --source to read a local Natural Earth GeoJSON instead of downloading it.
Output is written only to the explicitly supplied path. Small geometries are
preserved; large geometries are simplified when Shapely is installed. The
source's Crimea polygon is assigned to Ukraine before simplification.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from codify.jurisdictions import JURISDICTIONS_DIR, load_registry

SOURCE_URL = (
    "https://raw.githubusercontent.com/martynafford/natural-earth-geojson/"
    "master/10m/cultural/ne_10m_admin_0_countries.json"
)

# Natural Earth 10m source has ~14 decimal places. 3 decimals ≈ 111m precision,
# sub-pixel at any world-map zoom and roughly halves the payload vs 5.
COORD_PRECISION = 3

# Adaptive simplification budget: features at or below SMALL_VERT_THRESHOLD pass
# through untouched (small countries keep full 10m fidelity, so
# island states render properly). Larger features simplify with Douglas-Peucker
# (via shapely) toward LARGE_VERT_TARGET_RATIO × original vertex count, found by
# tolerance bisection. Empirically: Russia 32k → ~5k, US 14k → ~2k, payload
# ~3.2MB → ~1.1MB while preserving every small-state shape.
SMALL_VERT_THRESHOLD = 200
LARGE_VERT_TARGET_RATIO = 0.15

# UK is one country in the source map but four constituents in the Codify
# registry. Pick gb-eng as the canonical click-through target.
UK_NUMERIC = "826"
UK_CONSTITUENT_CODES = ("gb-eng", "gb-sct", "gb-wls", "gb-nir")

# Natural Earth uses "-99" as a sentinel for features where the author chose
# not to assign ISO codes (political disputes, unrecognised states). Norway is
# a notable case, its ISO_N3 is "-99" despite being the most mainstream ISO
# country imaginable. For these we fall back on the 3-letter ADM0_A3 code and
# match against our geo.json iso_alpha3 field.
UNASSIGNED = {"-99", "-1", ""}


def _build_stele_lookups() -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    """Return (iso_numeric → code, iso_alpha3 → code, code → tier)."""
    geo_path = JURISDICTIONS_DIR / "geo.json"
    geo = json.loads(geo_path.read_text())

    num_to_code: dict[str, str] = {}
    alpha3_to_code: dict[str, str] = {}
    for code, g in geo.items():
        iso_numeric = (g or {}).get("iso_numeric") or ""
        iso_alpha3 = (g or {}).get("iso_alpha3") or ""
        if iso_numeric and iso_numeric not in num_to_code:
            num_to_code[iso_numeric] = code
        if iso_alpha3 and iso_alpha3 not in alpha3_to_code:
            alpha3_to_code[iso_alpha3] = code

    # The ladder runs 1 to 5. Subsumed and no_corpus jurisdictions have a null
    # tier and render as 0, uncovered grey.
    registry = load_registry()
    tiers = {
        j["code"]: (int(j["tier"]) if j.get("tier") in (1, 2, 3, 4, 5) else 0) for j in registry
    }

    # UK-constituent fallback, one map feature, four codify profiles.
    if UK_NUMERIC not in num_to_code:
        for code in UK_CONSTITUENT_CODES:
            if code in tiers:
                num_to_code[UK_NUMERIC] = code
                break

    return num_to_code, alpha3_to_code, tiers


# Crimea + Sevastopol are internationally recognised as Ukraine, but Natural
# Earth's admin_0 layer renders the peninsula as a Russian polygon. This
# bbox bounds the peninsula (west of the Russian Taman mainland, ~36.9°E);
# the lone Russian sub-polygon contained in it is reassigned to Ukraine.
_CRIMEA_BBOX = (32.2, 44.2, 36.8, 46.4)  # (min_lng, min_lat, max_lng, max_lat)


def _ring_bbox(ring: list) -> tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _within_crimea(ring: list) -> bool:
    b = _ring_bbox(ring)
    return (
        b[0] >= _CRIMEA_BBOX[0]
        and b[1] >= _CRIMEA_BBOX[1]
        and b[2] <= _CRIMEA_BBOX[2]
        and b[3] <= _CRIMEA_BBOX[3]
    )


def _reassign_crimea(features: list[dict]) -> None:
    """Move the Crimea polygon from Russia's geometry into Ukraine's, in place.

    Specialist logic for Ukraine: NE attributes the peninsula to Russia.
    Ukraine's mainland already reaches the Isthmus of Perekop, so adding
    Crimea as another polygon leaves no gap and Russia keeps its mainland.
    """
    by_code = {f["properties"].get("stele_code"): f for f in features}
    ru, ua = by_code.get("ru"), by_code.get("ua")
    if not ru or not ua or ru["geometry"]["type"] != "MultiPolygon":
        return
    keep, moved = [], []
    for poly in ru["geometry"]["coordinates"]:
        (moved if _within_crimea(poly[0]) else keep).append(poly)
    if not moved:
        return
    ru["geometry"]["coordinates"] = keep
    if ua["geometry"]["type"] == "Polygon":
        ua["geometry"] = {
            "type": "MultiPolygon",
            "coordinates": [ua["geometry"]["coordinates"]],
        }
    ua["geometry"]["coordinates"].extend(moved)
    # Dissolve the isthmus seam so the outline layer doesn't double-stroke
    # where Crimea meets the mainland. Needs shapely (GEOS); without it the
    # peninsula still renders as Ukraine, just with a visible internal edge.
    try:
        from shapely.geometry import mapping, shape
        from shapely.ops import unary_union

        ua["geometry"] = mapping(unary_union(shape(ua["geometry"])))
        ua["geometry"]["coordinates"] = _round_coords(ua["geometry"]["coordinates"])
    except ImportError:
        print("  (shapely not installed — Crimea reassigned but seam not dissolved)")
    print(f"  reassigned {len(moved)} Crimea polygon(s) from Russia to Ukraine")


def _round_coords(obj):
    """Recursively round all numeric coordinates to COORD_PRECISION decimals.

    Accepts lists or tuples (shapely's ``mapping()`` yields nested tuples).
    """
    if isinstance(obj, (list, tuple)):
        # Leaf pair [lng, lat] vs nested list, check by element type.
        if obj and isinstance(obj[0], (int, float)):
            return [round(float(v), COORD_PRECISION) for v in obj]
        return [_round_coords(v) for v in obj]
    return obj


def _vertex_count(geometry: dict) -> int:
    """Total coordinate pairs across all rings of a (Multi)Polygon."""

    def _count(node) -> int:
        if isinstance(node, (list, tuple)):
            if node and isinstance(node[0], (int, float)):
                return 1
            return sum(_count(child) for child in node)
        return 0

    return _count(geometry.get("coordinates", []))


def _adaptive_simplify(geometry: dict, *, target_ratio: float = LARGE_VERT_TARGET_RATIO) -> dict:
    """Simplify a single feature's geometry toward a target vertex count.

    Small features pass through untouched so island states keep full fidelity.
    Larger features Douglas-Peucker-simplify (via shapely) with a tolerance
    chosen by bisection to land near ``original_verts * target_ratio``, small
    enough to dent the payload, generous enough that the silhouette survives.

    No-ops if shapely isn't installed; the caller logs a warning once.
    """
    verts = _vertex_count(geometry)
    if verts <= SMALL_VERT_THRESHOLD:
        return geometry
    try:
        from shapely.geometry import mapping, shape
    except ImportError:
        return geometry

    target = max(SMALL_VERT_THRESHOLD, int(verts * target_ratio))
    geom = shape(geometry)
    # Bisect tolerance in degrees over a wide range. 15 iterations is plenty
    # to hit ±10% of target, beyond that we trade size for runtime with no
    # visible benefit on a world-zoom map.
    lo, hi = 1e-4, 5.0
    best = geom
    best_verts = verts
    for _ in range(15):
        mid = (lo + hi) / 2
        simp = geom.simplify(mid, preserve_topology=True)
        if simp.is_empty:
            hi = mid
            continue
        sv = _vertex_count(mapping(simp))
        if sv > target:
            lo = mid  # need more aggressive simplification
        else:
            hi = mid
            if abs(sv - target) < abs(best_verts - target):
                best, best_verts = simp, sv
    return mapping(best)


def _feature_stele_code(
    props: dict,
    num_to_code: dict[str, str],
    alpha3_to_code: dict[str, str],
) -> str:
    """Look up a Codify jurisdiction code from a Natural Earth feature's properties."""
    iso_n3 = str(props.get("ISO_N3") or "").strip()
    if iso_n3 and iso_n3 not in UNASSIGNED and iso_n3 in num_to_code:
        return num_to_code[iso_n3]
    # Fallback, ADM0_A3 is Natural Earth's "always assigned" 3-letter code,
    # whereas ISO_A3 is "-99" for unresolved cases.
    adm0_a3 = str(props.get("ADM0_A3") or "").strip().upper()
    if adm0_a3 and adm0_a3 in alpha3_to_code:
        return alpha3_to_code[adm0_a3]
    return ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination GeoJSON path")
    parser.add_argument("--source", type=Path, help="Local source GeoJSON; otherwise download")
    args = parser.parse_args(argv)
    output: Path = args.output
    if args.source:
        source = json.loads(args.source.read_text())
    else:
        print(f"Fetching {SOURCE_URL}...")
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as resp:
            source = json.load(resp)

    num_to_code, alpha3_to_code, tiers = _build_stele_lookups()

    features: list[dict] = []
    matched = 0
    unmatched: list[str] = []
    for src_feature in source.get("features", []):
        props = src_feature.get("properties") or {}
        geometry = src_feature.get("geometry")
        if geometry is None:
            continue

        name = props.get("NAME") or props.get("ADMIN") or ""
        stele_code = _feature_stele_code(props, num_to_code, alpha3_to_code)
        tier = tiers.get(stele_code, 0) if stele_code else 0

        if stele_code:
            matched += 1
        else:
            unmatched.append(f"{props.get('ISO_N3', '?'):>5}  {name}")

        # Round coordinates down to keep the file slim (see COORD_PRECISION).
        geometry = {
            **geometry,
            "coordinates": _round_coords(geometry.get("coordinates", [])),
        }

        features.append(
            {
                "type": "Feature",
                "id": str(props.get("ISO_N3") or props.get("ADM0_A3") or name),
                "properties": {
                    "name": name,
                    "stele_code": stele_code,
                    "tier": tier,
                },
                "geometry": geometry,
            }
        )

    _reassign_crimea(features)

    # Variable-resolution simplification, small countries pass through at
    # 10m fidelity; large ones shrink. Crimea reassignment ran first so the
    # peninsula's polygon ends up in Ukraine before we simplify either side.
    raw_total = sum(_vertex_count(f["geometry"]) for f in features)
    biggest_before = 0
    biggest_after = 0
    biggest_name = ""
    for f in features:
        v_before = _vertex_count(f["geometry"])
        f["geometry"] = _adaptive_simplify(f["geometry"])
        # Re-round after shapely, its mapping() yields raw floats.
        f["geometry"] = {
            **f["geometry"],
            "coordinates": _round_coords(f["geometry"].get("coordinates", [])),
        }
        v_after = _vertex_count(f["geometry"])
        if v_before > biggest_before:
            biggest_before, biggest_after = v_before, v_after
            biggest_name = f["properties"].get("name") or "?"
    new_total = sum(_vertex_count(f["geometry"]) for f in features)
    print(
        f"  simplified: {raw_total:,} → {new_total:,} verts "
        f"({100 * new_total / max(1, raw_total):.0f}%); "
        f"largest country '{biggest_name}' {biggest_before:,} → {biggest_after:,}"
    )

    collection = {"type": "FeatureCollection", "features": features}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(collection, separators=(",", ":")))
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")
    print(f"  {matched}/{len(features)} countries matched a Codify jurisdiction")
    if unmatched:
        print(f"  unmatched ({len(unmatched)}):")
        for row in unmatched:
            print(f"    {row}")


if __name__ == "__main__":
    main()
