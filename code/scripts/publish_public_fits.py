from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
import re

import boto3
from dotenv import load_dotenv


HASH_RE = re.compile(r"^[0-9a-f]{8}$")
ALLOWED_SUFFIXES = (".json", ".parquet", ".npz")
KEEP_NAMES = ("config.json",)
SKIP_DIRS = {
    ("MCDR", "tau_sweep", "glmhmm_K2"),
}


def load_env() -> None:
    code_root = Path(__file__).resolve().parents[1]
    repo_root = code_root.parent
    for candidate in (code_root / ".env", repo_root / ".env", repo_root.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def require_env(name: str) -> str:
    value = env(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def fits_root() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "fits"


def iter_public_alias_dirs(root: Path):
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            for alias_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                triplet = (task_dir.name, model_dir.name, alias_dir.name)
                if triplet in SKIP_DIRS:
                    continue
                if HASH_RE.fullmatch(alias_dir.name):
                    continue
                yield triplet, alias_dir


def should_publish_file(path: Path) -> bool:
    if path.name.endswith("_cv_repeats.parquet"):
        return False
    if path.name == "tau_sweep_summary.parquet":
        return False
    if path.name in KEEP_NAMES:
        return True
    return path.suffix in ALLOWED_SUFFIXES and (
        path.name.endswith("_metrics.parquet") or path.name.endswith("_arrays.npz")
    )


def collect_manifest(root: Path) -> dict:
    manifest: dict[str, object] = {
        "version": 1,
        "tasks": {},
    }
    tasks = manifest["tasks"]
    assert isinstance(tasks, dict)

    for (task_name, model_kind, alias), alias_dir in iter_public_alias_dirs(root):
        files = []
        total_size = 0
        for file_path in sorted(p for p in alias_dir.iterdir() if p.is_file() and should_publish_file(p)):
            rel_path = f"{task_name}/{model_kind}/{alias}/{file_path.name}"
            files.append(rel_path)
            total_size += file_path.stat().st_size

        if not files:
            continue

        task_node = tasks.setdefault(task_name, {})
        model_node = task_node.setdefault(model_kind, {})
        model_node[alias] = {
            "files": files,
            "size_bytes": total_size,
        }

    return manifest


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=require_env("GLMHMMT_R2_ENDPOINT_URL"),
        aws_access_key_id=require_env("GLMHMMT_R2_ACCESS_KEY_ID"),
        aws_secret_access_key=require_env("GLMHMMT_R2_SECRET_ACCESS_KEY"),
        region_name=env("GLMHMMT_R2_REGION", "auto"),
    )


def upload_bytes(*, client, bucket: str, key: str, payload: bytes, content_type: str) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType=content_type,
    )


def upload_file(*, client, bucket: str, key: str, path: Path) -> None:
    content_type, _ = mimetypes.guess_type(path.name)
    extra = {"ContentType": content_type or "application/octet-stream"}
    client.upload_file(str(path), bucket, key, ExtraArgs=extra)


def main() -> None:
    load_env()
    root = fits_root()
    prefix = env("GLMHMMT_R2_PREFIX", "public-fits").strip("/")
    bucket = require_env("GLMHMMT_R2_BUCKET")
    client = s3_client()

    manifest = collect_manifest(root)
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    upload_bytes(
        client=client,
        bucket=bucket,
        key=f"{prefix}/manifest.json",
        payload=manifest_bytes,
        content_type="application/json",
    )

    published = 0
    for task_name, models in manifest["tasks"].items():
        for model_kind, aliases in models.items():
            for alias, meta in aliases.items():
                for rel_path in meta["files"]:
                    source = root / rel_path
                    upload_file(
                        client=client,
                        bucket=bucket,
                        key=f"{prefix}/{rel_path}",
                        path=source,
                    )
                    published += 1

    print(
        json.dumps(
            {
                "bucket": bucket,
                "prefix": prefix,
                "published_files": published + 1,
                "public_base_url": env("GLMHMMT_PUBLIC_FITS_BASE_URL"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
