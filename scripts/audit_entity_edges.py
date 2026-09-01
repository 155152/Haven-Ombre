#!/usr/bin/env python3
"""Audit lightweight entity-edge coverage for a bucket snapshot."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager
from entity_edges import extract_entity_edges_from_bucket
from identity import identity_names
from memory_layers import LAYER_SOURCE_RECORD, infer_bucket_layer
from self_anchor import is_self_anchor_bucket
from utils import load_config


DEFAULT_SNAPSHOT_ROOT = ROOT / "tmp" / "p0-local-snapshot-20260628-1528"
GENERIC_OBJECT_KEYS = {
    "我们",
    "我们的",
    "咱们",
    "一起",
    "共同",
    "关系",
    "记忆",
    "承诺",
    "约定",
    "项目",
    "故事",
    "暗号",
    "未来",
    "小雨",
    "haven",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read entity_edges.jsonl and dry-run the current extractor to find coverage gaps."
    )
    parser.add_argument(
        "--snapshot-root",
        default=str(DEFAULT_SNAPSHOT_ROOT),
        help="Snapshot root containing buckets/ and state/. Default: tmp/p0-local-snapshot-20260628-1528",
    )
    parser.add_argument(
        "--buckets-dir",
        default="",
        help="Override buckets directory. Defaults to <snapshot-root>/buckets.",
    )
    parser.add_argument(
        "--state-dir",
        default="",
        help="Override state directory. Defaults to <snapshot-root>/state.",
    )
    parser.add_argument(
        "--exclude-archive",
        action="store_true",
        help="Only inspect active permanent/dynamic/feel buckets.",
    )
    parser.add_argument(
        "--cases-file",
        default="",
        help="Optional JSONL cases file with expected_bucket_ids to audit expected bucket edge coverage.",
    )
    parser.add_argument("--json-output", default="", help="Optional JSON output path.")
    parser.add_argument("--md-output", default="", help="Optional Markdown report path.")
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--top-objects", type=int, default=20)
    parser.add_argument(
        "--generic-bucket-threshold",
        type=int,
        default=5,
        help="Flag object keys appearing in at least this many buckets as broad/noisy.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Append missing dry-run entity edges to state/entity_edges.jsonl. Default is dry-run.",
    )
    parser.add_argument(
        "--backup-dir",
        default="",
        help="Backup directory used with --apply when entity_edges.jsonl already exists.",
    )
    return parser.parse_args(argv)


def _compact(value: Any) -> str:
    return re.sub(r"[\s。；;，,、：:\"'“”‘’「」『』【】\[\]（）()!?！？~～._-]+", "", str(value or "").lower())


def _eligible_entity_edge_bucket(bucket: dict[str, Any]) -> bool:
    meta = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    return bool(
        bucket
        and not is_self_anchor_bucket(bucket)
        and infer_bucket_layer(bucket) != LAYER_SOURCE_RECORD
        and meta.get("type") != "feel"
        and not meta.get("protected")
    )


def _bucket_name(bucket: dict[str, Any]) -> str:
    meta = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    return str(meta.get("name") or bucket.get("name") or bucket.get("id") or "").strip()


def _bucket_domain(bucket: dict[str, Any]) -> list[str]:
    meta = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    domain = meta.get("domain") or []
    if isinstance(domain, str):
        return [domain]
    if isinstance(domain, list):
        return [str(item) for item in domain if str(item or "").strip()]
    return []


def _bucket_tags(bucket: dict[str, Any]) -> list[str]:
    meta = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        return [tags]
    if isinstance(tags, list):
        return [str(item) for item in tags if str(item or "").strip()]
    return []


def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(edge.get("subject") or ""),
        str(edge.get("relation") or ""),
        str(edge.get("object_key") or _compact(edge.get("object_text") or "")),
        str(edge.get("bucket_id") or ""),
    )


def _normalize_extracted_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        row = dict(edge)
        row["object_key"] = str(row.get("object_key") or _compact(row.get("object_text") or ""))
        key = _edge_key(row)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(row)
    return normalized


def _missing_edges(existing_edges: list[dict[str, Any]], dry_run_edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_keys = {_edge_key(edge) for edge in existing_edges}
    return [edge for edge in dry_run_edges if _edge_key(edge) not in existing_keys]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _backup_existing_edges(path: Path, backup_dir_arg: str) -> str:
    if not path.exists():
        return ""
    backup_dir = Path(backup_dir_arg) if backup_dir_arg else path.parent / "backups" / f"entity_edges_backfill_{_utc_stamp()}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / path.name
    shutil.copy2(path, backup_path)
    return str(backup_path)


def _file_needs_leading_newline(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as fh:
        fh.seek(-1, 2)
        return fh.read(1) not in {b"\n", b"\r"}


def _append_edges_jsonl(path: Path, edges: list[dict[str, Any]]) -> int:
    if not edges:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = _file_needs_leading_newline(path)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        if needs_newline:
            fh.write("\n")
        for edge in edges:
            fh.write(json.dumps(edge, ensure_ascii=False, sort_keys=True) + "\n")
    return len(edges)


def _load_cases(path: str) -> list[dict[str, Any]]:
    if not path:
        return []
    cases_path = Path(path)
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(cases_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{cases_path}:{line_no}: invalid JSON: {exc}") from exc
        expected = row.get("expected_bucket_ids") or row.get("expected_ids") or []
        if isinstance(expected, str):
            expected_ids = [expected]
        else:
            expected_ids = [str(item) for item in expected if str(item or "").strip()]
        cases.append(
            {
                "query": str(row.get("query") or "").strip(),
                "expected_bucket_ids": expected_ids,
            }
        )
    return cases


def _read_existing_edges(state_dir: Path) -> list[dict[str, Any]]:
    path = state_dir / "entity_edges.jsonl"
    if not path.exists():
        return []
    edges: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            edges.append(raw)
    return _normalize_extracted_edges(edges)


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_root = Path(args.snapshot_root).resolve()
    buckets_dir = Path(args.buckets_dir).resolve() if args.buckets_dir else snapshot_root / "buckets"
    state_dir = Path(args.state_dir).resolve() if args.state_dir else snapshot_root / "state"
    config = deepcopy(load_config())
    config["buckets_dir"] = str(buckets_dir)
    config["state_dir"] = str(state_dir)
    return config


def _summarize_edges(edges: list[dict[str, Any]], bucket_by_id: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    relation_counts = Counter(str(edge.get("relation") or "") for edge in edges)
    subject_counts = Counter(str(edge.get("subject") or "") for edge in edges)
    object_buckets: dict[str, set[str]] = defaultdict(set)
    object_text: dict[str, str] = {}
    object_relations: dict[str, Counter[str]] = defaultdict(Counter)
    relation_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for edge in edges:
        relation = str(edge.get("relation") or "")
        bucket_id = str(edge.get("bucket_id") or "")
        object_key = str(edge.get("object_key") or _compact(edge.get("object_text") or ""))
        if object_key:
            object_buckets[object_key].add(bucket_id)
            object_text.setdefault(object_key, str(edge.get("object_text") or ""))
            object_relations[object_key][relation] += 1
        if relation and len(relation_samples[relation]) < args.sample_limit:
            relation_samples[relation].append(_edge_preview(edge, bucket_by_id))

    object_rows = [
        {
            "object_key": key,
            "object_text": object_text.get(key, ""),
            "bucket_count": len(bucket_ids),
            "edge_count": sum(object_relations[key].values()),
            "relations": dict(object_relations[key].most_common()),
        }
        for key, bucket_ids in sorted(
            object_buckets.items(),
            key=lambda item: (-len(item[1]), object_text.get(item[0], ""), item[0]),
        )
    ]
    top_objects = object_rows[: args.top_objects]

    broad_objects = [
        row
        for row in object_rows
        if row["bucket_count"] >= args.generic_bucket_threshold
        or row["object_key"] in GENERIC_OBJECT_KEYS
        or len(row["object_key"]) <= 1
    ][: args.top_objects]
    return {
        "edge_count": len(edges),
        "edge_bucket_count": len({str(edge.get("bucket_id") or "") for edge in edges if edge.get("bucket_id")}),
        "relation_counts": dict(relation_counts.most_common()),
        "subject_counts": dict(subject_counts.most_common()),
        "top_objects": top_objects,
        "broad_or_noisy_objects": broad_objects,
        "samples_by_relation": dict(relation_samples),
    }


def _edge_preview(edge: dict[str, Any], bucket_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bucket_id = str(edge.get("bucket_id") or "")
    bucket = bucket_by_id.get(bucket_id) or {}
    return {
        "bucket_id": bucket_id,
        "bucket_name": _bucket_name(bucket),
        "subject": edge.get("subject"),
        "relation": edge.get("relation"),
        "object_text": edge.get("object_text"),
        "confidence": edge.get("confidence"),
        "evidence": edge.get("evidence"),
    }


def _bucket_preview(bucket: dict[str, Any]) -> dict[str, Any]:
    return {
        "bucket_id": str(bucket.get("id") or ""),
        "bucket_name": _bucket_name(bucket),
        "domain": _bucket_domain(bucket),
        "tags": _bucket_tags(bucket),
    }


def _edges_by_bucket(edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        bucket_id = str(edge.get("bucket_id") or "")
        if bucket_id:
            rows[bucket_id].append(edge)
    return dict(rows)


def _bucket_edge_rows(
    bucket_ids: list[str] | set[str],
    bucket_by_id: dict[str, dict[str, Any]],
    edge_map: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edge_map = edge_map or {}
    for bucket_id in sorted(str(item) for item in bucket_ids if str(item or "").strip()):
        bucket = bucket_by_id.get(bucket_id) or {"id": bucket_id}
        edges = edge_map.get(bucket_id) or []
        relation_counts = Counter(str(edge.get("relation") or "") for edge in edges)
        objects = []
        seen_objects: set[str] = set()
        for edge in edges:
            obj = str(edge.get("object_text") or "")
            if obj and obj not in seen_objects:
                seen_objects.add(obj)
                objects.append(obj)
            if len(objects) >= 5:
                break
        row = _bucket_preview(bucket)
        row.update(
            {
                "edge_count": len(edges),
                "relation_counts": dict(relation_counts.most_common()),
                "objects": objects,
            }
        )
        rows.append(row)
    return rows


def _case_coverage(
    cases: list[dict[str, Any]],
    existing_bucket_ids: set[str],
    dry_run_bucket_ids: set[str],
    bucket_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        expected_ids = [str(item) for item in case.get("expected_bucket_ids") or [] if str(item or "").strip()]
        rows.append(
            {
                "query": case.get("query") or "",
                "expected_bucket_ids": expected_ids,
                "expected_bucket_names": {
                    bucket_id: _bucket_name(bucket_by_id.get(bucket_id) or {})
                    for bucket_id in expected_ids
                },
                "expected_with_existing_edges": [bucket_id for bucket_id in expected_ids if bucket_id in existing_bucket_ids],
                "expected_with_dry_run_edges": [bucket_id for bucket_id in expected_ids if bucket_id in dry_run_bucket_ids],
                "expected_missing_dry_run_edges": [bucket_id for bucket_id in expected_ids if bucket_id not in dry_run_bucket_ids],
            }
        )
    return rows


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Entity Edges Coverage Audit",
        "",
        f"- snapshot_root: `{report['snapshot_root']}`",
        f"- buckets_dir: `{report['buckets_dir']}`",
        f"- state_dir: `{report['state_dir']}`",
        f"- existing_edges_path: `{report['existing_edges_path']}`",
        f"- existing_edges_file_exists: `{report['existing_edges_file_exists']}`",
        f"- include_archive: `{report['include_archive']}`",
        f"- total_buckets: `{summary['total_buckets']}`",
        f"- existing_edges: `{summary['existing_edges']}` across `{summary['existing_edge_buckets']}` buckets",
        f"- dry_run_edges: `{summary['dry_run_edges']}` across `{summary['dry_run_edge_buckets']}` buckets",
        f"- backfill_edges: `{summary['backfill_edges']}` across `{summary['backfill_edge_buckets']}` buckets",
        f"- apply: `{report['backfill']['apply']}`",
        f"- applied_backfill_edges: `{summary['applied_backfill_edges']}`",
        f"- backup_path: `{report['backfill']['backup_path']}`",
        f"- missing_backfill_buckets: `{summary['missing_backfill_bucket_count']}`",
        f"- dry_run_no_edge_buckets: `{summary['dry_run_no_edge_bucket_count']}`",
        "",
        "## Dry-run Relation Counts",
        "",
    ]
    for relation, count in report["dry_run"]["relation_counts"].items():
        lines.append(f"- `{relation}`: {count}")
    lines.extend(["", "## Broad Or Noisy Objects", ""])
    broad = report["dry_run"]["broad_or_noisy_objects"]
    if not broad:
        lines.append("- None flagged.")
    else:
        for row in broad:
            lines.append(
                f"- `{row['object_text']}`: {row['bucket_count']} buckets, {row['edge_count']} edges, relations={row['relations']}"
            )
    lines.extend(["", "## Missing Backfill Samples", ""])
    for row in report["missing_backfill_samples"]:
        lines.append(f"- `{row['bucket_id']}` {row['bucket_name']}")
    if not report["missing_backfill_samples"]:
        lines.append("- None.")
    lines.extend(["", "## Backfill Edge Samples", ""])
    for edge in report["backfill"]["edges"][: summary["sample_limit"]]:
        lines.append(
            f"- `{edge.get('bucket_id')}` {edge.get('subject')} / {edge.get('relation')} / {edge.get('object_text')}"
        )
    if not report["backfill"]["edges"]:
        lines.append("- None.")
    lines.extend(["", "## Dry-run No-edge Samples", ""])
    for row in report["dry_run_no_edge_samples"]:
        lines.append(f"- `{row['bucket_id']}` {row['bucket_name']}")
    if not report["dry_run_no_edge_samples"]:
        lines.append("- None.")
    if report.get("case_coverage"):
        lines.extend(["", "## Case Expected Bucket Coverage", ""])
        for row in report["case_coverage"]:
            missing = row["expected_missing_dry_run_edges"]
            lines.append(
                f"- `{row['query']}`: dry_run {len(row['expected_with_dry_run_edges'])}/{len(row['expected_bucket_ids'])}, "
                f"existing {len(row['expected_with_existing_edges'])}/{len(row['expected_bucket_ids'])}, "
                f"missing={missing}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def audit(args: argparse.Namespace) -> dict[str, Any]:
    config = _build_config(args)
    snapshot_root = Path(args.snapshot_root).resolve()
    buckets_dir = Path(config["buckets_dir"]).resolve()
    state_dir = Path(config["state_dir"]).resolve()
    if not buckets_dir.exists():
        raise FileNotFoundError(f"Buckets dir does not exist: {buckets_dir}")

    bucket_mgr = BucketManager(config)
    all_buckets = await bucket_mgr.list_all(include_archive=not args.exclude_archive)
    buckets = [bucket for bucket in all_buckets if _eligible_entity_edge_bucket(bucket)]
    bucket_by_id = {str(bucket.get("id") or ""): bucket for bucket in buckets if bucket.get("id")}
    identity = identity_names(config)

    existing_edges_path = state_dir / "entity_edges.jsonl"
    existing_edges = _read_existing_edges(state_dir)
    dry_run_edges: list[dict[str, Any]] = []
    for bucket in buckets:
        dry_run_edges.extend(extract_entity_edges_from_bucket(bucket, identity))
    dry_run_edges = _normalize_extracted_edges(dry_run_edges)
    backfill_edges = _missing_edges(existing_edges, dry_run_edges)
    backup_path = ""
    applied_count = 0
    if args.apply and backfill_edges:
        backup_path = _backup_existing_edges(existing_edges_path, args.backup_dir)
        applied_count = _append_edges_jsonl(existing_edges_path, backfill_edges)

    existing_bucket_ids = {str(edge.get("bucket_id") or "") for edge in existing_edges if edge.get("bucket_id")}
    dry_run_bucket_ids = {str(edge.get("bucket_id") or "") for edge in dry_run_edges if edge.get("bucket_id")}
    backfill_bucket_ids = {str(edge.get("bucket_id") or "") for edge in backfill_edges if edge.get("bucket_id")}
    existing_edges_by_bucket = _edges_by_bucket(existing_edges)
    dry_run_edges_by_bucket = _edges_by_bucket(dry_run_edges)
    all_bucket_ids = set(bucket_by_id)
    missing_backfill_ids = sorted(dry_run_bucket_ids - existing_bucket_ids)
    no_edge_ids = sorted(all_bucket_ids - dry_run_bucket_ids)

    cases = _load_cases(args.cases_file)
    report = {
        "snapshot_root": str(snapshot_root),
        "buckets_dir": str(buckets_dir),
        "state_dir": str(state_dir),
        "existing_edges_path": str(existing_edges_path),
        "existing_edges_file_exists": existing_edges_path.exists(),
        "include_archive": not args.exclude_archive,
        "summary": {
            "total_buckets": len(buckets),
            "sample_limit": args.sample_limit,
            "existing_edges": len(existing_edges),
            "existing_edge_buckets": len(existing_bucket_ids),
            "dry_run_edges": len(dry_run_edges),
            "dry_run_edge_buckets": len(dry_run_bucket_ids),
            "backfill_edges": len(backfill_edges),
            "backfill_edge_buckets": len(backfill_bucket_ids),
            "applied_backfill_edges": applied_count,
            "missing_backfill_bucket_count": len(missing_backfill_ids),
            "dry_run_no_edge_bucket_count": len(no_edge_ids),
        },
        "backfill": {
            "apply": bool(args.apply),
            "mode": "apply" if args.apply else "dry_run",
            "path": str(existing_edges_path),
            "edge_count": len(backfill_edges),
            "edge_bucket_count": len(backfill_bucket_ids),
            "applied_edge_count": applied_count,
            "backup_path": backup_path,
            "edges": backfill_edges,
        },
        "existing": _summarize_edges(existing_edges, bucket_by_id, args),
        "dry_run": _summarize_edges(dry_run_edges, bucket_by_id, args),
        "existing_edge_buckets": _bucket_edge_rows(existing_bucket_ids, bucket_by_id, existing_edges_by_bucket),
        "dry_run_edge_buckets": _bucket_edge_rows(dry_run_bucket_ids, bucket_by_id, dry_run_edges_by_bucket),
        "missing_backfill_buckets": _bucket_edge_rows(missing_backfill_ids, bucket_by_id, dry_run_edges_by_bucket),
        "dry_run_no_edge_buckets": _bucket_edge_rows(no_edge_ids, bucket_by_id),
        "missing_backfill_samples": [_bucket_preview(bucket_by_id[bucket_id]) for bucket_id in missing_backfill_ids[: args.sample_limit] if bucket_id in bucket_by_id],
        "dry_run_no_edge_samples": [_bucket_preview(bucket_by_id[bucket_id]) for bucket_id in no_edge_ids[: args.sample_limit] if bucket_id in bucket_by_id],
        "case_coverage": _case_coverage(cases, existing_bucket_ids, dry_run_bucket_ids, bucket_by_id),
    }
    return report


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = await audit(args)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.md_output:
        _write_markdown(report, Path(args.md_output))

    summary = report["summary"]
    print(
        "entity_edges audit: "
        f"buckets={summary['total_buckets']} "
        f"existing_edges={summary['existing_edges']} "
        f"existing_edge_buckets={summary['existing_edge_buckets']} "
        f"dry_run_edges={summary['dry_run_edges']} "
        f"dry_run_edge_buckets={summary['dry_run_edge_buckets']} "
        f"backfill_edges={summary['backfill_edges']} "
        f"backfill_edge_buckets={summary['backfill_edge_buckets']} "
        f"applied_backfill_edges={summary['applied_backfill_edges']} "
        f"missing_backfill_buckets={summary['missing_backfill_bucket_count']} "
        f"dry_run_no_edge_buckets={summary['dry_run_no_edge_bucket_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
