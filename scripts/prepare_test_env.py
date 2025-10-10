#!/usr/bin/env python3
"""
Prepare deterministic data plane state for contract/integration tests.

This script is executed inside the Docker Compose test-runner container before
pytest is launched. It resets PostgreSQL (vector store metadata),
flushes Redis, and clears the MinIO artifact bucket so every run starts from a
known-good baseline. That guarantees tests can be re-run locally/CI without
leftover state causing flaky behaviour.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse, urlunparse

import redis
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from minio import Minio
from minio.error import S3Error


def _log(message: str) -> None:
    """Emit a simple timestamped log line."""
    ts = time.strftime("%H:%M:%S")
    print(f"[prepare-test-env {ts}] {message}", flush=True)


def _wait_for(fn: Callable[[], bool], name: str, timeout: int = 60, interval: int = 2) -> None:
    """Poll a readiness/check function until it returns True or timeout."""
    start = time.monotonic()
    last_error: Optional[Exception] = None

    while time.monotonic() - start < timeout:
        try:
            if fn():
                _log(f"{name} ready")
                return
        except Exception as exc:  # noqa: BLE001 - we just log and retry
            last_error = exc
        time.sleep(interval)

    if last_error:
        raise RuntimeError(f"Timed out waiting for {name}: {last_error}") from last_error
    raise RuntimeError(f"Timed out waiting for {name}")


# --- PostgreSQL helpers ----------------------------------------------------

def _postgres_admin_dsn(worker_dsn: str) -> str:
    """Derive an admin DSN (connect to 'postgres' DB) from service DSN."""
    parsed = urlparse(worker_dsn)
    admin_path = "/postgres"
    return urlunparse(parsed._replace(path=admin_path))


def _postgres_db_name(worker_dsn: str) -> str:
    """Extract target database name from DSN path."""
    parsed = urlparse(worker_dsn)
    db_name = parsed.path.lstrip("/")
    if not db_name:
        raise ValueError("PostgreSQL DSN is missing database name")
    return db_name


def _postgres_user(worker_dsn: str) -> str:
    """Extract database user from DSN."""
    parsed = urlparse(worker_dsn)
    if not parsed.username:
        raise ValueError("PostgreSQL DSN is missing username")
    return parsed.username


def reset_postgres() -> None:
    """Drop and recreate the ML database, then run initialization SQL."""
    worker_dsn = os.environ["ML_VECTOR_DB_DSN"]
    admin_dsn = os.environ.get("ML_VECTOR_DB_ADMIN_DSN", _postgres_admin_dsn(worker_dsn))
    db_name = _postgres_db_name(worker_dsn)
    db_owner = _postgres_user(worker_dsn)

    _log(f"Resetting PostgreSQL database '{db_name}' as owner '{db_owner}'")

    # Drop and recreate the database
    admin_conn = psycopg2.connect(admin_dsn)
    admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        with admin_conn.cursor() as cur:
            cur.execute("SELECT pid FROM pg_stat_activity WHERE datname = %s;", (db_name,))
            for (pid,) in cur.fetchall():
                cur.execute("SELECT pg_terminate_backend(%s);", (pid,))

            cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name)))
            cur.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(db_name),
                    sql.Identifier(db_owner),
                )
            )
    finally:
        admin_conn.close()

    # Run database bootstrap SQL inside the fresh database
    init_sql_path = Path(__file__).resolve().parent / "init_db.sql"
    init_sql = init_sql_path.read_text()

    worker_conn = psycopg2.connect(worker_dsn)
    worker_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        with worker_conn.cursor() as cur:
            cur.execute(init_sql)
    finally:
        worker_conn.close()

    _log("PostgreSQL reset complete")


# --- Redis helpers ---------------------------------------------------------

def reset_redis() -> None:
    """Flush Redis to remove any cached embeddings/events."""
    redis_url = os.environ["ML_REDIS_URL"]
    client = redis.from_url(redis_url)
    client.flushall()
    _log("Redis flushed")


# --- MinIO helpers ---------------------------------------------------------

def _minio_client() -> tuple[Minio, str]:
    """Create MinIO client and resolve artifact bucket name."""
    endpoint = os.environ.get("ML_MINIO_ENDPOINT", "http://minio:9000")
    access_key = os.environ.get("ML_MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("ML_MINIO_SECRET_KEY", "minioadmin123")
    artifact_uri = os.environ.get("ML_ARTIFACT_URI", "s3://ml-artifacts")

    parsed = urlparse(endpoint)
    secure = parsed.scheme == "https"
    host = parsed.netloc or parsed.path  # allow bare host:port

    artifact = urlparse(artifact_uri)
    bucket = artifact.netloc or artifact.path.lstrip("/")
    if not bucket:
        bucket = "ml-artifacts"

    client = Minio(
        host,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )
    return client, bucket


def reset_minio() -> None:
    """Ensure artifact bucket exists and remove existing objects."""
    client, bucket = _minio_client()

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        _log(f"Created MinIO bucket '{bucket}'")
        return

    _log(f"Clearing MinIO bucket '{bucket}'")
    objects = client.list_objects(bucket, recursive=True)
    to_delete = [obj.object_name for obj in objects]
    if not to_delete:
        _log("MinIO bucket already empty")
        return

    for object_name in to_delete:
        try:
            client.remove_object(bucket, object_name)
        except S3Error as exc:  # noqa: PERF203 - explicit logging
            _log(f"Warning: failed to delete object '{object_name}': {exc}")

    _log("MinIO bucket cleared")


# --- Wait-check wrappers ---------------------------------------------------

def check_postgres_ready() -> bool:
    worker_dsn = os.environ["ML_VECTOR_DB_DSN"]
    conn = psycopg2.connect(worker_dsn)
    conn.close()
    return True


def check_redis_ready() -> bool:
    redis_url = os.environ["ML_REDIS_URL"]
    client = redis.from_url(redis_url)
    client.ping()
    return True


def check_minio_ready() -> bool:
    client, _ = _minio_client()
    client.list_buckets()
    return True


def main() -> None:
    """Entry point: wait for dependencies, then reset them."""
    required_env = [
        "ML_VECTOR_DB_DSN",
        "ML_REDIS_URL",
        "ML_MINIO_ENDPOINT",
        "ML_MINIO_ACCESS_KEY",
        "ML_MINIO_SECRET_KEY",
        "ML_ARTIFACT_URI",
    ]
    missing = [var for var in required_env if not os.environ.get(var)]
    if missing:
        _log(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    _wait_for(check_postgres_ready, "PostgreSQL", timeout=90)
    _wait_for(check_redis_ready, "Redis", timeout=60)
    _wait_for(check_minio_ready, "MinIO", timeout=60)

    reset_postgres()
    reset_redis()
    reset_minio()

    _log("Environment preparation complete")


if __name__ == "__main__":
    main()
