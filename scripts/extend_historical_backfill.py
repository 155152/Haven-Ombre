#!/usr/bin/env python3
"""Incrementally extend a completed historical backfill by one frozen day.

The extension preserves all prior manual review/promotion decisions and reviews
only the newly added day plus similarity edges that touch a new candidate.
It writes staging artifacts only. It never writes Ombre Bucket Markdown or the
Memory Moment index.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.review_historical_bridge import (  # noqa: E402
    _all_edge_components,
    _candidate_record,
    _first_pass_batch,
    build_bridge_merge_staging,
)
from scripts.review_historical_llm_memory import (  # noqa: E402
    build_merge_staging,
    build_review_batches,
    collect_staged_summaries,
    normalize_manual_review_batch,
    score_related_pairs,
)
from scripts.review_historical_promotion import (  # noqa: E402
    _candidate_row,
    build_promotion_staging,
    normalize_promotion_batch,
)

EXTENSION_VERSION = 1
DEFAULT_REVIEW_BATCH_SIZE = 28
DEFAULT_PROMOTION_BATCH_SIZE = 50
PROMOTION_TARGETS = ("bucket_core", "bucket_moment", "raw_only")


@dataclass(frozen=True)
class ExtensionPaths:
    root: Path
    source_manifest: Path
    day_review_root: Path
    bridge_root: Path
    promotion_root: Path


def extension_paths(root: Path) -> ExtensionPaths:
    resolved = root.resolve()
    return ExtensionPaths(
        root=resolved,
        source_manifest=resolved / "source_manifest.json",
        day_review_root=resolved / "day_review",
        bridge_root=resolved / "bridge_review",
        promotion_root=resolved / "promotion_review",
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _candidate_id(row: dict[str, Any]) -> str:
    value = str(row.get("merge_id") or "").strip()
    if not value:
        raise ValueError("retained candidate has empty merge_id")
    return value


def _promotion_candidate_id(row: dict[str, Any]) -> str:
    value = str(row.get("candidate_id") or "").strip()
    if not value:
        raise ValueError("promotion row has empty candidate_id")
    return value


def _promotion_rows(staging: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for target in PROMOTION_TARGETS:
        for raw in staging.get(f"{target}_candidates") or []:
            row = dict(raw)
            candidate_id = _promotion_candidate_id(row)
            if candidate_id in rows:
                raise ValueError(f"duplicate promotion candidate_id: {candidate_id}")
            if str(row.get("promotion_target") or target) != target:
                raise ValueError(f"promotion target mismatch: {candidate_id}")
            row["promotion_target"] = target
            rows[candidate_id] = row
    return rows


def _load_reviewed_batches(
    batches: list[dict[str, Any]],
    manual_dir: Path,
    *,
    promotion: bool = False,
) -> list[dict[str, Any]]:
    manual_dir = manual_dir.resolve()
    reviewed: list[dict[str, Any]] = []
    for batch in batches:
        batch_id = str(batch.get("batch_id") or "")
        path = manual_dir / f"{batch_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing manual decision file: {path}")
        raw = _load_json(path)
        reviewed.append(
            normalize_promotion_batch(batch, raw)
            if promotion
            else normalize_manual_review_batch(batch, raw)
        )
    return reviewed


def build_new_day_review(
    *,
    stage_parent: Path,
    new_date: str,
    manual_dir: Path,
    batch_size: int = DEFAULT_REVIEW_BATCH_SIZE,
) -> dict[str, Any]:
    records = collect_staged_summaries(stage_parent.resolve(), new_date, new_date)
    pairs = score_related_pairs(records)
    batches = build_review_batches(records, pairs, batch_size=max(1, int(batch_size)))
    reviewed = _load_reviewed_batches(batches, manual_dir)
    manifest = {
        "start_date": new_date,
        "end_date": new_date,
        "records": records,
    }
    staging = build_merge_staging(manifest, reviewed)
    if not staging.get("review_complete") or int(staging.get("pending_summary_count") or 0) != 0:
        raise ValueError("new-day manual review is incomplete")
    return {
        "date": new_date,
        "source_summary_count": len(records),
        "related_pair_count": len(pairs),
        "batch_count": len(batches),
        "batches": batches,
        "reviewed_batches": reviewed,
        "merge_staging": staging,
    }


def build_incremental_bridge_groups(
    previous_candidates: list[dict[str, Any]],
    new_candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    previous_by_id = {_candidate_id(row): row for row in previous_candidates}
    new_by_id = {_candidate_id(row): row for row in new_candidates}
    if set(previous_by_id) & set(new_by_id):
        raise ValueError("previous and new candidate ids overlap")
    combined = [*previous_candidates, *new_candidates]
    by_id = {**previous_by_id, **new_by_id}
    records = [_candidate_record(row) for row in combined]
    all_pairs = score_related_pairs(records)
    new_ids = set(new_by_id)

    incremental_pairs: list[dict[str, Any]] = []
    for pair in all_pairs:
        left_id = str(pair["left_id"])
        right_id = str(pair["right_id"])
        left_new = left_id in new_ids
        right_new = right_id in new_ids
        if not left_new and not right_new:
            continue
        if left_new and right_new:
            if _first_pass_batch(by_id[left_id]) == _first_pass_batch(by_id[right_id]):
                continue
        incremental_pairs.append(pair)

    bridge_ids = {
        value
        for pair in incremental_pairs
        for value in (str(pair["left_id"]), str(pair["right_id"]))
    }
    components = _all_edge_components(bridge_ids, incremental_pairs) if bridge_ids else []
    record_by_id = {row["summary_id"]: row for row in records}
    related: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in incremental_pairs:
        related[str(pair["left_id"])].append(
            {"summary_id": str(pair["right_id"]), "score": pair["score"]}
        )
        related[str(pair["right_id"])].append(
            {"summary_id": str(pair["left_id"]), "score": pair["score"]}
        )
    for value in related:
        related[value].sort(key=lambda row: (-float(row["score"]), str(row["summary_id"])))

    groups: list[dict[str, Any]] = []
    for index, component in enumerate(
        sorted(components, key=lambda ids: (min(record_by_id[v]["date"] for v in ids), ids[0])),
        start=1,
    ):
        component_set = set(component)
        items: list[dict[str, Any]] = []
        for value in sorted(component, key=lambda v: (record_by_id[v]["date"], v)):
            item = dict(record_by_id[value])
            item["related"] = [
                row for row in related.get(value, []) if row["summary_id"] in component_set
            ]
            items.append(item)
        groups.append(
            {
                "version": EXTENSION_VERSION,
                "batch_id": f"bridge_{index:03d}",
                "item_count": len(items),
                "items": items,
            }
        )

    manifest = {
        "version": EXTENSION_VERSION,
        "previous_candidate_count": len(previous_candidates),
        "new_candidate_count": len(new_candidates),
        "incremental_related_pair_count": len(incremental_pairs),
        "bridge_candidate_count": len(bridge_ids),
        "group_count": len(groups),
        "largest_group_size": max((group["item_count"] for group in groups), default=0),
        "bridge_candidate_ids": sorted(bridge_ids),
        "incremental_related_pairs": incremental_pairs,
        "formal_memory_write": False,
    }
    return manifest, groups


def build_incremental_bridge(
    *,
    previous_candidates: list[dict[str, Any]],
    new_candidates: list[dict[str, Any]],
    manual_dir: Path,
) -> dict[str, Any]:
    manifest, groups = build_incremental_bridge_groups(previous_candidates, new_candidates)
    reviewed = _load_reviewed_batches(groups, manual_dir) if groups else []
    upstream = {
        "retained_candidates": [*previous_candidates, *new_candidates],
    }
    staging = build_bridge_merge_staging(upstream, manifest, reviewed)
    if not staging.get("bridge_review_complete") or int(staging.get("pending_bridge_candidate_count") or 0) != 0:
        raise ValueError("incremental bridge review is incomplete")
    return {
        "manifest": manifest,
        "groups": groups,
        "reviewed_groups": reviewed,
        "merge_staging": staging,
    }


def build_incremental_promotion_batches(
    previous_promotion: dict[str, Any],
    final_candidates: list[dict[str, Any]],
    *,
    batch_size: int = DEFAULT_PROMOTION_BATCH_SIZE,
) -> dict[str, Any]:
    previous_rows = _promotion_rows(previous_promotion)
    final_by_id = {_candidate_id(row): row for row in final_candidates}
    carried_ids = sorted(set(previous_rows) & set(final_by_id))
    dropped_ids = sorted(set(previous_rows) - set(final_by_id))
    changed_ids = sorted(
        set(final_by_id) - set(previous_rows),
        key=lambda value: ((final_by_id[value].get("evidence_dates") or [""])[-1], value),
    )
    changed = [final_by_id[value] for value in changed_ids]
    size = max(1, int(batch_size))
    batches: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(changed), size), start=1):
        items = [_candidate_row(row) for row in changed[start : start + size]]
        batches.append(
            {
                "version": EXTENSION_VERSION,
                "batch_id": f"promotion_{index:03d}",
                "item_count": len(items),
                "items": items,
            }
        )
    return {
        "previous_rows": previous_rows,
        "carried_ids": carried_ids,
        "dropped_ids": dropped_ids,
        "changed_ids": changed_ids,
        "batches": batches,
    }


def merge_incremental_promotion(
    *,
    previous_promotion: dict[str, Any],
    final_candidates: list[dict[str, Any]],
    bridge_candidate_ids: list[str],
    manual_dir: Path,
    batch_size: int = DEFAULT_PROMOTION_BATCH_SIZE,
) -> dict[str, Any]:
    plan = build_incremental_promotion_batches(
        previous_promotion,
        final_candidates,
        batch_size=batch_size,
    )
    bridge_ids = {str(value) for value in bridge_candidate_ids}
    unexpected_dropped = set(plan["dropped_ids"]) - bridge_ids
    if unexpected_dropped:
        raise ValueError(
            f"previous promotion candidates disappeared outside incremental bridge: {sorted(unexpected_dropped)}"
        )

    batches = list(plan["batches"])
    reviewed = _load_reviewed_batches(batches, manual_dir, promotion=True) if batches else []
    delta = build_promotion_staging(batches, reviewed) if batches else {
        "promotion_review_complete": True,
        "pending_candidate_count": 0,
        "decision_groups": [],
        "bucket_core_candidates": [],
        "bucket_moment_candidates": [],
        "raw_only_candidates": [],
    }
    if not delta.get("promotion_review_complete") or int(delta.get("pending_candidate_count") or 0) != 0:
        raise ValueError("incremental promotion review is incomplete")

    carried = [plan["previous_rows"][value] for value in plan["carried_ids"]]
    target_rows: dict[str, list[dict[str, Any]]] = {target: [] for target in PROMOTION_TARGETS}
    for row in carried:
        target = str(row.get("promotion_target") or "")
        if target not in target_rows:
            raise ValueError(f"invalid carried promotion target: {target!r}")
        target_rows[target].append(dict(row))
    for target in PROMOTION_TARGETS:
        target_rows[target].extend(dict(row) for row in delta.get(f"{target}_candidates") or [])
        target_rows[target].sort(key=lambda row: (str(row.get("date") or ""), str(row.get("candidate_id") or "")))

    final_ids = {_candidate_id(row) for row in final_candidates}
    promoted_ids = {
        _promotion_candidate_id(row)
        for target in PROMOTION_TARGETS
        for row in target_rows[target]
    }
    if promoted_ids != final_ids:
        missing = sorted(final_ids - promoted_ids)
        extra = sorted(promoted_ids - final_ids)
        raise ValueError(f"combined promotion coverage mismatch: missing={missing} extra={extra}")

    result = {
        "version": EXTENSION_VERSION,
        "source_candidate_count": len(final_candidates),
        "reviewed_candidate_count": len(final_candidates),
        "pending_candidate_count": 0,
        "promotion_review_complete": True,
        "formal_memory_write": False,
        "formal_moment_write": False,
        "previous_carried_candidate_count": len(plan["carried_ids"]),
        "previous_dropped_candidate_ids": plan["dropped_ids"],
        "incremental_candidate_count": len(plan["changed_ids"]),
        "incremental_candidate_ids": plan["changed_ids"],
        "incremental_decision_groups": list(delta.get("decision_groups") or []),
    }
    for target in PROMOTION_TARGETS:
        result[f"{target}_candidates"] = target_rows[target]
        result[f"{target}_count"] = len(target_rows[target])
    return {
        "plan": plan,
        "reviewed_batches": reviewed,
        "promotion_staging": result,
    }


def build_extension(
    *,
    stage_parent: Path,
    previous_bridge_staging_path: Path,
    previous_promotion_staging_path: Path,
    new_date: str,
    day_manual_dir: Path,
    bridge_manual_dir: Path,
    promotion_manual_dir: Path,
    review_batch_size: int = DEFAULT_REVIEW_BATCH_SIZE,
    promotion_batch_size: int = DEFAULT_PROMOTION_BATCH_SIZE,
) -> dict[str, Any]:
    previous_bridge = _load_json(previous_bridge_staging_path.resolve())
    if not previous_bridge.get("bridge_review_complete") or int(previous_bridge.get("pending_bridge_candidate_count") or 0) != 0:
        raise ValueError("previous bridge staging must be complete")
    if previous_bridge.get("formal_memory_write") is not False:
        raise ValueError("previous bridge staging must be non-writing")
    previous_candidates = list(previous_bridge.get("retained_candidates") or [])

    previous_promotion = _load_json(previous_promotion_staging_path.resolve())
    if not previous_promotion.get("promotion_review_complete") or int(previous_promotion.get("pending_candidate_count") or 0) != 0:
        raise ValueError("previous promotion staging must be complete")
    if previous_promotion.get("formal_memory_write") is not False or previous_promotion.get("formal_moment_write") is not False:
        raise ValueError("previous promotion staging must be non-writing")
    previous_promoted_ids = set(_promotion_rows(previous_promotion))
    previous_candidate_ids = {_candidate_id(row) for row in previous_candidates}
    if previous_promoted_ids != previous_candidate_ids:
        raise ValueError("previous promotion coverage does not match previous bridge candidates")

    day = build_new_day_review(
        stage_parent=stage_parent,
        new_date=new_date,
        manual_dir=day_manual_dir,
        batch_size=review_batch_size,
    )
    new_candidates = list(day["merge_staging"].get("retained_candidates") or [])
    bridge = build_incremental_bridge(
        previous_candidates=previous_candidates,
        new_candidates=new_candidates,
        manual_dir=bridge_manual_dir,
    )
    final_candidates = list(bridge["merge_staging"].get("retained_candidates") or [])
    promotion = merge_incremental_promotion(
        previous_promotion=previous_promotion,
        final_candidates=final_candidates,
        bridge_candidate_ids=list(bridge["manifest"].get("bridge_candidate_ids") or []),
        manual_dir=promotion_manual_dir,
        batch_size=promotion_batch_size,
    )
    return {
        "version": EXTENSION_VERSION,
        "new_date": new_date,
        "formal_memory_write": False,
        "formal_moment_write": False,
        "previous_candidate_count": len(previous_candidates),
        "new_day_source_summary_count": day["source_summary_count"],
        "new_day_retained_candidate_count": len(new_candidates),
        "incremental_bridge_candidate_count": bridge["manifest"]["bridge_candidate_count"],
        "incremental_bridge_group_count": bridge["manifest"]["group_count"],
        "final_retained_candidate_count": len(final_candidates),
        "incremental_promotion_candidate_count": promotion["promotion_staging"]["incremental_candidate_count"],
        "promotion_counts": {
            target: promotion["promotion_staging"][f"{target}_count"]
            for target in PROMOTION_TARGETS
        },
        "day_review": day,
        "bridge_review": bridge,
        "promotion_review": promotion,
    }


def write_extension_artifacts(result: dict[str, Any], root: Path) -> dict[str, Any]:
    paths = extension_paths(root)
    paths.root.mkdir(parents=True, exist_ok=True)
    day = result["day_review"]
    bridge = result["bridge_review"]
    promotion = result["promotion_review"]

    for batch in day["batches"]:
        _json_dump(paths.day_review_root / "review_batches" / f"{batch['batch_id']}.json", batch)
    for reviewed in day["reviewed_batches"]:
        _json_dump(paths.day_review_root / "decisions" / f"{reviewed['batch_id']}.json", reviewed)
    _json_dump(paths.day_review_root / "merge_staging.json", day["merge_staging"])

    _json_dump(paths.bridge_root / "source_manifest.json", bridge["manifest"])
    for group in bridge["groups"]:
        _json_dump(paths.bridge_root / "review_groups" / f"{group['batch_id']}.json", group)
    for reviewed in bridge["reviewed_groups"]:
        _json_dump(paths.bridge_root / "decisions" / f"{reviewed['batch_id']}.json", reviewed)
    _json_dump(paths.bridge_root / "merge_staging.json", bridge["merge_staging"])

    for batch in promotion["plan"]["batches"]:
        _json_dump(paths.promotion_root / "review_batches" / f"{batch['batch_id']}.json", batch)
    for reviewed in promotion["reviewed_batches"]:
        _json_dump(paths.promotion_root / "decisions" / f"{reviewed['batch_id']}.json", reviewed)
    _json_dump(paths.promotion_root / "promotion_staging.json", promotion["promotion_staging"])

    source_manifest = {
        key: value
        for key, value in result.items()
        if key not in {"day_review", "bridge_review", "promotion_review"}
    }
    source_manifest["extension_root"] = str(paths.root)
    source_manifest["promotion_staging"] = str(paths.promotion_root / "promotion_staging.json")
    _json_dump(paths.source_manifest, source_manifest)
    return source_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally extend a completed historical backfill by one day")
    parser.add_argument("--stage-parent", required=True)
    parser.add_argument("--previous-bridge-staging", required=True)
    parser.add_argument("--previous-promotion-staging", required=True)
    parser.add_argument("--new-date", required=True)
    parser.add_argument("--day-manual-dir", required=True)
    parser.add_argument("--bridge-manual-dir", required=True)
    parser.add_argument("--promotion-manual-dir", required=True)
    parser.add_argument("--extension-root", required=True)
    parser.add_argument("--review-batch-size", type=int, default=DEFAULT_REVIEW_BATCH_SIZE)
    parser.add_argument("--promotion-batch-size", type=int, default=DEFAULT_PROMOTION_BATCH_SIZE)
    args = parser.parse_args()

    result = build_extension(
        stage_parent=Path(args.stage_parent),
        previous_bridge_staging_path=Path(args.previous_bridge_staging),
        previous_promotion_staging_path=Path(args.previous_promotion_staging),
        new_date=str(args.new_date),
        day_manual_dir=Path(args.day_manual_dir),
        bridge_manual_dir=Path(args.bridge_manual_dir),
        promotion_manual_dir=Path(args.promotion_manual_dir),
        review_batch_size=max(1, int(args.review_batch_size)),
        promotion_batch_size=max(1, int(args.promotion_batch_size)),
    )
    manifest = write_extension_artifacts(result, Path(args.extension_root))
    print(json.dumps({"status": "historical_extension_ready", **manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
