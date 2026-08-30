#!/usr/bin/env python3
"""Apply a validated historical materialization preview to live Ombre Buckets.

The materialization preview is the immutable write source for this stage.
This script never promotes raw_only rows and never constructs Memory Moments
independently. Each written Bucket is reloaded through BucketManager and then
indexed with MemoryMomentStore.upsert_bucket(), which derives Moments from the
Bucket Markdown using Ombre's normal parser.

The apply is idempotent and resumable:
- a missing preview Bucket is created with its deterministic historical bucket id;
- an already-live matching historical Bucket is verified and reused;
- a conflicting live Bucket with the same id fails closed;
- derived Moment indexing is safe to repeat;
- a checkpoint is atomically rewritten after every completed candidate.

No service restart is performed by this script.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager  # noqa: E402
from embedding_engine import EmbeddingEngine  # noqa: E402
from memory_moments import MemoryMomentStore, parse_bucket_moments  # noqa: E402

APPLY_VERSION = 1


@dataclass(frozen=True)
class ApplyPaths:
    staging: Path
    previews_dir: Path
    checkpoint: Path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _atomic_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("config must be a YAML object")
    if not str(value.get("buckets_dir") or "").strip():
        raise ValueError("config.buckets_dir is required")
    return value


def apply_paths(materialization_staging: Path, checkpoint: Path | None = None) -> ApplyPaths:
    staging = materialization_staging.resolve()
    manifest = _load_json(staging)
    previews_dir = Path(str(manifest.get("bucket_previews_dir") or "")).resolve()
    if not str(manifest.get("bucket_previews_dir") or "").strip():
        raise ValueError("materialization staging is missing bucket_previews_dir")
    checkpoint_path = (
        checkpoint.resolve()
        if checkpoint is not None
        else (staging.parent / "formal_apply_checkpoint.json").resolve()
    )
    return ApplyPaths(staging=staging, previews_dir=previews_dir, checkpoint=checkpoint_path)


def _validate_materialization_manifest(manifest: dict[str, Any], paths: ApplyPaths) -> list[dict[str, Any]]:
    if manifest.get("materialization_ready") is not True:
        raise ValueError("materialization staging is not ready")
    if int(manifest.get("blocking_duplicate_warning_count") or 0) != 0:
        raise ValueError("materialization staging has blocking duplicate warnings")
    if manifest.get("formal_memory_write") is not False or manifest.get("formal_moment_write") is not False:
        raise ValueError("materialization preview must remain non-writing")

    previews = list(manifest.get("bucket_previews") or [])
    expected = int(manifest.get("materialized_preview_count") or 0)
    if len(previews) != expected:
        raise ValueError("materialized preview count mismatch")
    if int(manifest.get("derived_moment_preview_count") or 0) != expected:
        raise ValueError("derived Moment preview count mismatch")
    promotion_counts = manifest.get("promotion_counts") or {}
    if int(manifest.get("raw_only_provenance_count") or 0) != int(promotion_counts.get("raw_only") or 0):
        raise ValueError("raw_only provenance count mismatch")

    bucket_ids = [str(row.get("bucket_id") or "").strip() for row in previews]
    candidate_ids = [str(row.get("candidate_id") or "").strip() for row in previews]
    if "" in bucket_ids or len(bucket_ids) != len(set(bucket_ids)):
        raise ValueError("invalid or duplicate preview bucket_id")
    if "" in candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("invalid or duplicate preview candidate_id")

    if not paths.previews_dir.exists():
        raise FileNotFoundError(f"bucket preview directory missing: {paths.previews_dir}")
    preview_files = sorted(paths.previews_dir.glob("*.md"))
    if len(preview_files) != expected:
        raise ValueError(f"bucket preview file count mismatch: {len(preview_files)} != {expected}")
    file_ids = {path.stem for path in preview_files}
    if file_ids != set(bucket_ids):
        missing = sorted(set(bucket_ids) - file_ids)
        extra = sorted(file_ids - set(bucket_ids))
        raise ValueError(f"bucket preview files mismatch: missing={missing} extra={extra}")
    return previews


def _load_preview_bucket(paths: ApplyPaths, row: dict[str, Any]) -> dict[str, Any]:
    bucket_id = str(row["bucket_id"])
    preview_path = paths.previews_dir / f"{bucket_id}.md"
    post = frontmatter.load(preview_path)
    metadata = dict(post.metadata)
    if str(metadata.get("id") or "") != bucket_id:
        raise ValueError(f"preview bucket id mismatch: {bucket_id}")
    if str(metadata.get("historical_candidate_id") or "") != str(row.get("candidate_id") or ""):
        raise ValueError(f"preview candidate id mismatch: {bucket_id}")
    if str(post.content or "") != str(row.get("content") or ""):
        raise ValueError(f"preview content mismatch: {bucket_id}")
    expected_meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in (
        "name",
        "type",
        "source",
        "date",
        "historical_candidate_id",
        "historical_promotion_target",
        "historical_memory_scope",
        "source_summary_ids",
        "source_raw_event_ids",
    ):
        if metadata.get(key) != expected_meta.get(key):
            raise ValueError(f"preview metadata mismatch for {bucket_id}: {key}")
    moments = parse_bucket_moments({"id": bucket_id, "metadata": metadata, "content": str(post.content or "")})
    if len(moments) != 1:
        raise ValueError(f"preview no longer derives exactly one Moment: {bucket_id}")
    expected_moment = row.get("derived_moment_preview") or {}
    if str(moments[0].get("section") or "") != str(expected_moment.get("section") or ""):
        raise ValueError(f"preview derived Moment section changed: {bucket_id}")
    if str(moments[0].get("text") or "") != str(expected_moment.get("text") or ""):
        raise ValueError(f"preview derived Moment text changed: {bucket_id}")
    return {"id": bucket_id, "metadata": metadata, "content": str(post.content or ""), "path": str(preview_path)}


def _live_bucket_matches(expected: dict[str, Any], live: dict[str, Any]) -> bool:
    expected_meta = expected.get("metadata") or {}
    live_meta = live.get("metadata") or {}
    if str(live.get("content") or "") != str(expected.get("content") or ""):
        return False
    for key in (
        "id",
        "name",
        "type",
        "source",
        "date",
        "historical_candidate_id",
        "historical_promotion_target",
        "historical_memory_scope",
        "historical_evidence_dates",
        "source_summary_ids",
        "source_raw_event_ids",
        "protected_historical_event",
    ):
        if live_meta.get(key) != expected_meta.get(key):
            return False
    return True


def _create_kwargs(expected: dict[str, Any]) -> dict[str, Any]:
    meta = dict(expected["metadata"])
    reserved = {
        "id",
        "name",
        "tags",
        "domain",
        "valence",
        "arousal",
        "importance",
        "type",
        "created",
        "last_active",
        "updated_at",
        "activation_count",
        "confidence",
        "period",
        "date",
        "pinned",
        "protected",
        "anchor",
        "resolved",
        "digested",
        "source",
    }
    extra_metadata = {key: value for key, value in meta.items() if key not in reserved}
    return {
        "content": expected["content"],
        "tags": list(meta.get("tags") or []),
        "importance": int(meta.get("importance") or 5),
        "domain": list(meta.get("domain") or []),
        "valence": float(meta.get("valence") if meta.get("valence") is not None else 0.5),
        "arousal": float(meta.get("arousal") if meta.get("arousal") is not None else 0.3),
        "bucket_type": str(meta.get("type") or "dynamic"),
        "name": str(meta.get("name") or "") or None,
        "pinned": bool(meta.get("pinned")),
        "protected": bool(meta.get("protected")),
        "bucket_id": str(meta["id"]),
        "source": str(meta.get("source") or "") or None,
        "created": str(meta.get("created") or "") or None,
        "last_active": str(meta.get("last_active") or "") or None,
        "updated_at": str(meta.get("updated_at") or "") or None,
        "anchor": bool(meta.get("anchor")),
        "resolved": bool(meta.get("resolved")),
        "digested": bool(meta.get("digested")),
        "confidence": meta.get("confidence"),
        "period": str(meta.get("period") or "") or None,
        "date": str(meta.get("date") or "") or None,
        "extra_metadata": extra_metadata,
    }


def _new_checkpoint(manifest: dict[str, Any], paths: ApplyPaths, embedding_enabled: bool) -> dict[str, Any]:
    existing_duplicates = list(manifest.get("existing_duplicates") or [])
    return {
        "version": APPLY_VERSION,
        "materialization_staging": str(paths.staging),
        "bucket_previews_dir": str(paths.previews_dir),
        "formal_memory_write": False,
        "formal_moment_write": False,
        "derived_moment_index_write": False,
        "apply_complete": False,
        "embedding_enabled": bool(embedding_enabled),
        "expected_bucket_write_count": int(manifest.get("materialized_preview_count") or 0),
        "raw_only_provenance_count": int(manifest.get("raw_only_provenance_count") or 0),
        "existing_duplicate_skip_count": len(existing_duplicates),
        "existing_duplicates": existing_duplicates,
        "created_count": 0,
        "already_present_count": 0,
        "moment_indexed_count": 0,
        "embedding_refreshed_count": 0,
        "embedding_skipped_count": 0,
        "failed_count": 0,
        "items": [],
    }


def _checkpoint_item_map(checkpoint: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("bucket_id") or ""): row
        for row in checkpoint.get("items") or []
        if str(row.get("bucket_id") or "")
    }


def _refresh_checkpoint_counts(checkpoint: dict[str, Any]) -> None:
    items = list(checkpoint.get("items") or [])
    checkpoint["created_count"] = sum(1 for row in items if row.get("bucket_status") == "created")
    checkpoint["already_present_count"] = sum(1 for row in items if row.get("bucket_status") == "already_present")
    checkpoint["moment_indexed_count"] = sum(1 for row in items if row.get("moment_indexed") is True)
    checkpoint["embedding_refreshed_count"] = sum(1 for row in items if row.get("embedding_refreshed") is True)
    checkpoint["embedding_skipped_count"] = sum(1 for row in items if row.get("embedding_skipped"))
    checkpoint["failed_count"] = sum(1 for row in items if row.get("error"))


async def apply_materialization(
    *,
    config: dict[str, Any],
    materialization_staging: Path,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    paths = apply_paths(materialization_staging, checkpoint_path)
    manifest = _load_json(paths.staging)
    rows = _validate_materialization_manifest(manifest, paths)

    manager = BucketManager(config)
    moment_store = MemoryMomentStore(config)
    embedding = EmbeddingEngine(config)

    checkpoint = (
        _load_json(paths.checkpoint)
        if paths.checkpoint.exists()
        else _new_checkpoint(manifest, paths, embedding.enabled)
    )
    if str(checkpoint.get("materialization_staging") or "") != str(paths.staging):
        raise ValueError("existing apply checkpoint belongs to a different materialization staging")
    completed = _checkpoint_item_map(checkpoint)

    ordered_rows = sorted(rows, key=lambda row: (str(row.get("date") or ""), str(row.get("candidate_id") or "")))
    for row in ordered_rows:
        bucket_id = str(row["bucket_id"])
        expected = _load_preview_bucket(paths, row)
        prior = completed.get(bucket_id)
        if prior and not prior.get("error") and prior.get("moment_indexed") is True:
            live = await manager.get(bucket_id)
            if not live or not _live_bucket_matches(expected, live):
                raise ValueError(f"completed checkpoint bucket no longer matches live data: {bucket_id}")
            continue

        item = {
            "candidate_id": str(row["candidate_id"]),
            "bucket_id": bucket_id,
            "promotion_target": str(row.get("promotion_target") or ""),
            "date": str(row.get("date") or ""),
            "bucket_status": "",
            "moment_indexed": False,
            "moment_count": 0,
            "embedding_refreshed": False,
            "embedding_skipped": "",
            "error": "",
        }
        created_now = False
        target_path: Path | None = None
        try:
            live = await manager.get(bucket_id)
            if live:
                if not _live_bucket_matches(expected, live):
                    raise ValueError("live bucket id exists with conflicting historical content")
                item["bucket_status"] = "already_present"
            else:
                created_id = await manager.create(**_create_kwargs(expected))
                if created_id != bucket_id:
                    raise ValueError(f"BucketManager returned unexpected id: {created_id}")
                created_now = True
                item["bucket_status"] = "created"
                target_relpath = str(row.get("target_relpath") or "")
                target_path = (Path(str(config["buckets_dir"])) / Path(target_relpath)).resolve()

            live = await manager.get(bucket_id)
            if not live or not _live_bucket_matches(expected, live):
                raise ValueError("live bucket verification failed after create/reuse")

            moments = moment_store.upsert_bucket(live)
            if len(moments) != 1:
                raise ValueError(f"derived Moment count changed after live write: {len(moments)}")
            expected_moment = row.get("derived_moment_preview") or {}
            if str(moments[0].get("section") or "") != str(expected_moment.get("section") or ""):
                raise ValueError("derived Moment section mismatch after live write")
            if str(moments[0].get("text") or "") != str(expected_moment.get("text") or ""):
                raise ValueError("derived Moment text mismatch after live write")
            item["moment_indexed"] = True
            item["moment_count"] = 1
            item["moment_id"] = str(moments[0].get("moment_id") or "")

            if embedding.enabled:
                from utils import bucket_text_for_embedding

                item["embedding_refreshed"] = bool(
                    await embedding.generate_and_store(bucket_id, bucket_text_for_embedding(live))
                )
                if not item["embedding_refreshed"]:
                    item["embedding_skipped"] = "generation_failed"
            else:
                item["embedding_skipped"] = "disabled"
        except Exception as exc:
            item["error"] = str(exc)
            if created_now and not item["moment_indexed"]:
                # Fail closed for a newly-created item that could not complete its
                # derived Moment index. This path is deterministic and did not
                # exist before this apply run.
                if target_path and target_path.exists():
                    target_path.unlink()
                try:
                    moment_store.delete_bucket(bucket_id)
                except Exception:
                    pass
                try:
                    embedding.delete_embedding(bucket_id)
                except Exception:
                    pass
                item["bucket_status"] = "rolled_back"
            existing_items = [row for row in checkpoint.get("items") or [] if str(row.get("bucket_id") or "") != bucket_id]
            existing_items.append(item)
            checkpoint["items"] = existing_items
            _refresh_checkpoint_counts(checkpoint)
            checkpoint["formal_memory_write"] = False
            checkpoint["derived_moment_index_write"] = False
            checkpoint["apply_complete"] = False
            _atomic_json_dump(paths.checkpoint, checkpoint)
            raise RuntimeError(f"historical materialization apply failed at {bucket_id}: {exc}") from exc

        existing_items = [row for row in checkpoint.get("items") or [] if str(row.get("bucket_id") or "") != bucket_id]
        existing_items.append(item)
        checkpoint["items"] = existing_items
        _refresh_checkpoint_counts(checkpoint)
        _atomic_json_dump(paths.checkpoint, checkpoint)

    _refresh_checkpoint_counts(checkpoint)
    expected_count = int(checkpoint["expected_bucket_write_count"])
    successful_bucket_count = int(checkpoint["created_count"]) + int(checkpoint["already_present_count"])
    if successful_bucket_count != expected_count:
        raise ValueError(f"formal Bucket apply coverage mismatch: {successful_bucket_count} != {expected_count}")
    if int(checkpoint["moment_indexed_count"]) != expected_count:
        raise ValueError("derived Moment index coverage mismatch")
    if int(checkpoint["failed_count"]) != 0:
        raise ValueError("apply checkpoint still contains failed items")

    checkpoint["formal_memory_write"] = True
    checkpoint["formal_moment_write"] = False
    checkpoint["derived_moment_index_write"] = True
    checkpoint["apply_complete"] = True
    _atomic_json_dump(paths.checkpoint, checkpoint)
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply validated historical materialization previews to live Ombre")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--materialization-staging", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        parser.error("formal write requires explicit --apply")

    config = _load_config(Path(args.config).resolve())
    checkpoint = Path(args.checkpoint).resolve() if str(args.checkpoint).strip() else None
    result = asyncio.run(
        apply_materialization(
            config=config,
            materialization_staging=Path(args.materialization_staging),
            checkpoint_path=checkpoint,
        )
    )
    print(
        json.dumps(
            {
                "status": "historical_materialization_applied",
                "apply_complete": result["apply_complete"],
                "formal_memory_write": result["formal_memory_write"],
                "formal_moment_write": result["formal_moment_write"],
                "derived_moment_index_write": result["derived_moment_index_write"],
                "expected_bucket_write_count": result["expected_bucket_write_count"],
                "created_count": result["created_count"],
                "already_present_count": result["already_present_count"],
                "moment_indexed_count": result["moment_indexed_count"],
                "embedding_enabled": result["embedding_enabled"],
                "embedding_refreshed_count": result["embedding_refreshed_count"],
                "embedding_skipped_count": result["embedding_skipped_count"],
                "existing_duplicate_skip_count": result["existing_duplicate_skip_count"],
                "raw_only_provenance_count": result["raw_only_provenance_count"],
                "failed_count": result["failed_count"],
                "checkpoint": str(apply_paths(Path(args.materialization_staging), checkpoint).checkpoint),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
