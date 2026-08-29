#!/usr/bin/env python3
"""Repair Cyberboss backfill source-evidence buckets after decay/enrichment drift.

Dry-run by default. Only buckets carrying both `source_record` and
`cyberboss_backfill` tags are eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import frontmatter
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("config must be a mapping")
    return value


def body_sha256(content: str) -> str:
    return hashlib.sha256(str(content or "").encode("utf-8")).hexdigest()


def expected_tags(meta: dict[str, Any]) -> list[str]:
    source_type = str(meta.get("source_type") or "").strip()
    tags = ["source_record", "cyberboss_backfill"]
    if source_type == "stone_feeling":
        tags.append("stone_feeling")
    elif source_type == "stone_feature":
        tags.append("stone_feature")
        category = str(meta.get("stone_category") or "").strip()
        if category:
            tags.append(category)
    elif source_type == "stone_notebook":
        tags.append("stone_notebook")
    elif source_type == "cyberboss_memory_file":
        tags.append("memory_file")
    elif source_type == "cyberboss_memory_index":
        tags.append("legacy_memory")
        category = str(meta.get("legacy_category") or "").strip()
        if category:
            tags.append(category)
    elif source_type == "cyberboss_promise_candidate":
        tags.append("promise")
        status = str(meta.get("promise_status") or meta.get("status") or "").strip()
        if status:
            tags.append(status)
    else:
        existing = [str(tag) for tag in (meta.get("tags") or []) if str(tag).strip()]
        tags.extend(tag for tag in existing if tag not in tags)
    return list(dict.fromkeys(tags))


def eligible(post: frontmatter.Post) -> bool:
    tags = {str(tag) for tag in (post.get("tags") or [])}
    return "source_record" in tags and "cyberboss_backfill" in tags and post.get("source") == "cyberboss_backfill"


def canonicalize(post: frontmatter.Post) -> tuple[frontmatter.Post, list[str]]:
    changes: list[str] = []
    source_sha = str(post.get("source_sha256") or "").strip()
    actual_sha = body_sha256(str(post.content or ""))
    if not source_sha or actual_sha != source_sha:
        raise ValueError(f"source body hash mismatch id={post.get('id')} expected={source_sha} actual={actual_sha}")

    desired_tags = expected_tags(dict(post.metadata))
    if list(post.get("tags") or []) != desired_tags:
        post["tags"] = desired_tags
        changes.append("tags")
    if post.get("type") != "source":
        post["type"] = "source"
        changes.append("type")
    if int(post.get("importance", 5) or 5) != 5:
        post["importance"] = 5
        changes.append("importance")
    if "confidence" in post.metadata:
        del post.metadata["confidence"]
        changes.append("confidence")
    for key in ("active", "deprecated", "resolved"):
        if key in post.metadata:
            del post.metadata[key]
            changes.append(key)
    if int(post.get("activation_count", 0) or 0) != 0:
        post["activation_count"] = 0
        changes.append("activation_count")
    created = str(post.get("created") or "").strip()
    if created:
        if str(post.get("last_active") or "") != created:
            post["last_active"] = created
            changes.append("last_active")
        if str(post.get("updated_at") or "") != created:
            post["updated_at"] = created
            changes.append("updated_at")
    return post, changes


def atomic_write(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config).resolve())
    buckets_dir = Path(str(config.get("buckets_dir") or ROOT / "buckets")).resolve()
    state_dir = Path(str(config.get("state_dir") or ROOT / "state")).resolve()
    backup_root = state_dir / "source_evidence_repair_backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    by_id: dict[str, list[tuple[Path, frontmatter.Post]]] = defaultdict(list)
    for path in buckets_dir.rglob("*.md"):
        try:
            post = frontmatter.load(path)
        except Exception:
            continue
        if eligible(post):
            by_id[str(post.get("id") or path.stem)].append((path, post))

    stats = {
        "mode": "apply" if args.apply else "dry-run",
        "unique_source_records": len(by_id),
        "files": sum(len(items) for items in by_id.values()),
        "duplicates": 0,
        "would_backup_duplicates": 0,
        "would_repair": 0,
        "repaired": 0,
        "hash_mismatches": [],
        "change_fields": defaultdict(int),
    }

    for bucket_id, entries in sorted(by_id.items()):
        if len(entries) > 1:
            stats["duplicates"] += len(entries) - 1
        # Prefer a non-archive copy as the canonical file.
        entries.sort(key=lambda pair: ("/archive/" in pair[0].as_posix().lower(), pair[0].as_posix()))
        canonical_path, canonical_post = entries[0]
        for duplicate_path, duplicate_post in entries[1:]:
            if body_sha256(str(duplicate_post.content or "")) != body_sha256(str(canonical_post.content or "")):
                stats["hash_mismatches"].append({"id": bucket_id, "reason": "duplicate_body_conflict"})
                continue
            stats["would_backup_duplicates"] += 1
            if args.apply:
                relative = duplicate_path.relative_to(buckets_dir)
                target = backup_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(duplicate_path), str(target))

        try:
            canonical_post, changes = canonicalize(canonical_post)
        except ValueError as exc:
            stats["hash_mismatches"].append({"id": bucket_id, "reason": str(exc)})
            continue
        if changes:
            stats["would_repair"] += 1
            for field in changes:
                stats["change_fields"][field] += 1
            if args.apply:
                atomic_write(canonical_path, frontmatter.dumps(canonical_post))
                stats["repaired"] += 1

    stats["change_fields"] = dict(sorted(stats["change_fields"].items()))
    if stats["hash_mismatches"]:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
