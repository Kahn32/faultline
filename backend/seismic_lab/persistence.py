"""Optional production cache and prediction history with local fallbacks."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os


class AnalysisStore:
    def __init__(self) -> None:
        self.memory: dict[str, dict] = {}
        self.redis = None
        self.clickhouse = None
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis

                self.redis = redis.Redis.from_url(
                    redis_url, decode_responses=True,
                    socket_connect_timeout=1, socket_timeout=1,
                )
                self.redis.ping()
            except Exception:
                self.redis = None
        clickhouse_host = os.getenv("CLICKHOUSE_HOST")
        if clickhouse_host:
            try:
                import clickhouse_connect

                self.clickhouse = clickhouse_connect.get_client(
                    host=clickhouse_host,
                    port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
                    database=os.getenv("CLICKHOUSE_DB", "faultline"),
                    connect_timeout=1,
                )
                self.clickhouse.command("""
                    CREATE TABLE IF NOT EXISTS prediction_history (
                        analysis_id String,
                        created_at DateTime,
                        model_version String,
                        filename String,
                        p_probability Float32,
                        magnitude Float32,
                        distance_km Float32,
                        latency_ms Float32
                    ) ENGINE = MergeTree ORDER BY (created_at, analysis_id)
                """)
            except Exception:
                self.clickhouse = None

    @property
    def status(self) -> dict[str, str]:
        return {
            "cache": "redis" if self.redis is not None else "memory",
            "history": "clickhouse" if self.clickhouse is not None else "disabled",
        }

    def get(self, key: str) -> dict | None:
        if self.redis is not None:
            try:
                value = self.redis.get(f"faultline:analysis:{key}")
                return json.loads(value) if value else None
            except Exception:
                self.redis = None
        return self.memory.get(key)

    def save(self, key: str, response: dict, filename: str) -> None:
        self.memory[key] = response
        if self.redis is not None:
            try:
                self.redis.setex(f"faultline:analysis:{key}", 3600, json.dumps(response))
            except Exception:
                self.redis = None
        if self.clickhouse is not None:
            try:
                self.clickhouse.insert(
                    "prediction_history",
                    [[
                        key,
                        datetime.now(timezone.utc).replace(tzinfo=None),
                        response["model_version"],
                        filename,
                        response["p_wave"]["probability"],
                        response["estimates"]["magnitude"] if response["estimates"] else float("nan"),
                        response["estimates"]["distance_km"] if response["estimates"] else float("nan"),
                        response["latency_ms"],
                    ]],
                    column_names=[
                        "analysis_id", "created_at", "model_version", "filename",
                        "p_probability", "magnitude", "distance_km", "latency_ms",
                    ],
                )
            except Exception:
                self.clickhouse = None
