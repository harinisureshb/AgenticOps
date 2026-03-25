# tools.py — LangChain tools for AgenticOps agents

import json
import os

import chromadb
import requests
from chromadb.utils import embedding_functions
from langchain_core.tools import tool

# ──────────────── Paths ────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TELEMETRY_PATH = os.path.join(BASE_DIR, "logs", "telemetry_logs.json")
APP_LOGS_PATH = os.path.join(BASE_DIR, "logs", "application_logs.json")
CICD_LOGS_PATH = os.path.join(BASE_DIR, "logs", "cicd_logs.json")
FAQS_PATH = os.path.join(BASE_DIR, "FAQs", "resolution_faqs.json")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "vector_database", "chroma_db")
COLLECTION_NAME = "sre_runbooks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ──────────────── Helpers ────────────────
def _load_json(path: str) -> list:
    """Load a JSON file and return the parsed list."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _telemetry_by_metric(metric_name: str) -> list[dict]:
    """Filter telemetry logs for a specific metric."""
    data = _load_json(TELEMETRY_PATH)
    return [entry for entry in data if entry.get("metric") == metric_name]


def _stats(values: list[float]) -> dict:
    """Compute basic statistics for a list of values."""
    if not values:
        return {"count": 0, "min": 0, "max": 0, "avg": 0}
    return {
        "count": len(values),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "avg": round(sum(values) / len(values), 2),
    }


# ═══════════════ METRICS TOOLS ═══════════════

@tool
def analyze_cpu_metrics() -> str:
    """Analyze CPU utilization metrics from telemetry logs.
    Detects spikes above 80% and returns statistics with anomalous entries."""
    entries = _telemetry_by_metric("cpu_utilization_percent")
    values = [e["value"] for e in entries]
    stats = _stats(values)

    spikes = [e for e in entries if e["value"] > 80]
    spike_summary = ""
    if spikes:
        spike_summary = f"\n\n🔴 ALERT: {len(spikes)} CPU spike(s) detected (>80%):\n"
        for s in spikes[:20]:  # limit output
            spike_summary += f"  - {s['timestamp']}: {s['value']}% on {s['service']}\n"
    else:
        spike_summary = "\n\n✅ No CPU spikes detected (all readings below 80%)."

    return (
        f"CPU Utilization Analysis ({stats['count']} data points)\n"
        f"  Min: {stats['min']}%  |  Max: {stats['max']}%  |  Avg: {stats['avg']}%"
        f"{spike_summary}"
    )


@tool
def analyze_memory_metrics() -> str:
    """Analyze memory usage metrics from telemetry logs.
    Detects high usage above 1500MB and returns statistics with anomalous entries."""
    entries = _telemetry_by_metric("memory_usage_mb")
    values = [e["value"] for e in entries]
    stats = _stats(values)

    high_mem = [e for e in entries if e["value"] > 1500]
    alert = ""
    if high_mem:
        alert = f"\n\n🔴 ALERT: {len(high_mem)} high memory reading(s) detected (>1500MB):\n"
        for h in high_mem[:20]:
            alert += f"  - {h['timestamp']}: {h['value']}MB on {h['service']}\n"
    else:
        alert = "\n\n✅ Memory usage within normal range (all below 1500MB)."

    return (
        f"Memory Usage Analysis ({stats['count']} data points)\n"
        f"  Min: {stats['min']}MB  |  Max: {stats['max']}MB  |  Avg: {stats['avg']}MB"
        f"{alert}"
    )


@tool
def analyze_latency_metrics() -> str:
    """Analyze payment gateway and DB query latency from telemetry logs.
    Detects payment latency spikes (>1000ms) and DB latency spikes (>200ms)."""
    # Payment gateway latency
    pg_entries = _telemetry_by_metric("payment_gateway_latency_ms")
    pg_values = [e["value"] for e in pg_entries]
    pg_stats = _stats(pg_values)

    pg_spikes = [e for e in pg_entries if e["value"] > 1000]
    pg_alert = ""
    if pg_spikes:
        pg_alert = f"\n  🔴 {len(pg_spikes)} payment latency spike(s) (>1000ms):\n"
        for s in pg_spikes[:15]:
            pg_alert += f"    - {s['timestamp']}: {s['value']}ms\n"
    else:
        pg_alert = "\n  ✅ Payment gateway latency normal."

    # DB query latency
    db_entries = _telemetry_by_metric("db_query_latency_ms")
    db_values = [e["value"] for e in db_entries]
    db_stats = _stats(db_values)

    db_spikes = [e for e in db_entries if e["value"] > 200]
    db_alert = ""
    if db_spikes:
        db_alert = f"\n  🔴 {len(db_spikes)} DB latency spike(s) (>200ms):\n"
        for s in db_spikes[:15]:
            db_alert += f"    - {s['timestamp']}: {s['value']}ms\n"
    else:
        db_alert = "\n  ✅ DB query latency normal."

    return (
        f"Latency Analysis\n"
        f"Payment Gateway ({pg_stats['count']} points): "
        f"Min={pg_stats['min']}ms | Max={pg_stats['max']}ms | Avg={pg_stats['avg']}ms"
        f"{pg_alert}\n"
        f"DB Query ({db_stats['count']} points): "
        f"Min={db_stats['min']}ms | Max={db_stats['max']}ms | Avg={db_stats['avg']}ms"
        f"{db_alert}"
    )


@tool
def analyze_error_rates() -> str:
    """Analyze checkout error rate metrics from telemetry logs.
    Detects elevated error rates above 5%."""
    entries = _telemetry_by_metric("checkout_error_rate_percent")
    values = [e["value"] for e in entries]
    stats = _stats(values)

    elevated = [e for e in entries if e["value"] > 5]
    alert = ""
    if elevated:
        alert = f"\n\n🔴 ALERT: {len(elevated)} elevated error rate reading(s) (>5%):\n"
        for e in elevated[:20]:
            alert += f"  - {e['timestamp']}: {e['value']}% on {e['service']}\n"
    else:
        alert = "\n\n✅ Error rates within normal range (all below 5%)."

    return (
        f"Error Rate Analysis ({stats['count']} data points)\n"
        f"  Min: {stats['min']}%  |  Max: {stats['max']}%  |  Avg: {stats['avg']}%"
        f"{alert}"
    )


@tool
def analyze_active_sessions() -> str:
    """Analyze active checkout sessions from telemetry logs.
    Detects session count spikes above 1000."""
    entries = _telemetry_by_metric("active_checkout_sessions")
    values = [e["value"] for e in entries]
    stats = _stats(values)

    spikes = [e for e in entries if e["value"] > 1000]
    alert = ""
    if spikes:
        alert = f"\n\n🔴 ALERT: {len(spikes)} session spike(s) (>1000 concurrent):\n"
        for s in spikes[:20]:
            alert += f"  - {s['timestamp']}: {s['value']} sessions on {s['service']}\n"
    else:
        alert = "\n\n✅ Active sessions within normal range (all below 1000)."

    return (
        f"Active Sessions Analysis ({stats['count']} data points)\n"
        f"  Min: {stats['min']}  |  Max: {stats['max']}  |  Avg: {stats['avg']}"
        f"{alert}"
    )


# ═══════════════ LOGS TOOLS ═══════════════

@tool
def get_failed_application_logs() -> str:
    """Retrieve ERROR and WARN level application logs.
    Returns a summary of failures grouped by method and error type."""
    data = _load_json(APP_LOGS_PATH)

    errors = [e for e in data if e.get("level") == "ERROR"]
    warnings = [e for e in data if e.get("level") == "WARN"]

    # Group errors by method
    error_by_method: dict[str, list] = {}
    for e in errors:
        method = e.get("method", "unknown")
        error_by_method.setdefault(method, []).append(e)

    result = f"Application Log Analysis\n"
    result += f"  Total ERROR logs: {len(errors)}\n"
    result += f"  Total WARN logs: {len(warnings)}\n\n"

    if errors:
        result += "ERROR breakdown by method:\n"
        for method, errs in error_by_method.items():
            result += f"\n  📛 {method} ({len(errs)} errors):\n"
            # Show unique error messages
            unique_msgs = set(e["message"] for e in errs)
            for msg in unique_msgs:
                count = sum(1 for e in errs if e["message"] == msg)
                result += f"    - [{count}x] {msg}\n"
            # Show time range
            times = [e["timestamp"] for e in errs]
            result += f"    Time range: {min(times)} → {max(times)}\n"

    if warnings:
        result += f"\nWARNING logs ({len(warnings)} total):\n"
        warn_methods = set(w["method"] for w in warnings)
        for method in warn_methods:
            count = sum(1 for w in warnings if w["method"] == method)
            result += f"  ⚠️ {method}: {count} warning(s)\n"

    if not errors and not warnings:
        result += "✅ No errors or warnings found in application logs."

    return result


@tool
def get_error_log_timeline() -> str:
    """Get a timeline of ERROR logs to identify incident windows.
    Groups errors into time windows to detect bursts."""
    data = _load_json(APP_LOGS_PATH)
    errors = [e for e in data if e.get("level") == "ERROR"]

    if not errors:
        return "✅ No ERROR logs found."

    # Group by hour
    hourly: dict[str, int] = {}
    for e in errors:
        hour = e["timestamp"][:13]  # "2026-03-24T07"
        hourly[hour] = hourly.get(hour, 0) + 1

    result = "Error Timeline (hourly distribution):\n"
    for hour in sorted(hourly.keys()):
        bar = "█" * hourly[hour]
        result += f"  {hour}:00Z  | {hourly[hour]:3d} errors | {bar}\n"

    # Identify peak hour
    peak_hour = max(hourly, key=hourly.get)
    result += f"\n🔴 Peak error hour: {peak_hour}:00Z with {hourly[peak_hour]} errors"

    return result


# ═══════════════ CI/CD TOOLS ═══════════════

@tool
def get_cicd_failures() -> str:
    """Retrieve failed CI/CD pipeline runs.
    Returns details of failed builds, tests, deployments, and security scans."""
    data = _load_json(CICD_LOGS_PATH)

    failures = [e for e in data if e.get("status") == "FAILED"]
    all_runs = len(data)

    result = f"CI/CD Pipeline Analysis ({all_runs} total runs)\n"
    result += f"  Failed: {len(failures)}  |  Success rate: {round((all_runs - len(failures)) / all_runs * 100, 1)}%\n\n"

    if failures:
        result += "Failed pipelines:\n"
        for f in failures:
            result += (
                f"  ❌ Pipeline {f['pipeline_id']} | Stage: {f['stage']} | "
                f"Commit: {f['commit_hash']} | Duration: {f['duration_sec']}s | "
                f"Time: {f['timestamp']}\n"
            )
    else:
        result += "✅ All pipelines succeeded."

    # Check for rollbacks
    rollbacks = [e for e in data if e.get("stage") == "rollback_production"]
    if rollbacks:
        result += f"\n\n⚠️ Production rollbacks detected ({len(rollbacks)}):\n"
        for r in rollbacks:
            result += f"  🔄 {r['timestamp']} | Commit: {r['commit_hash']} | Status: {r['status']}\n"

    return result


@tool
def get_deployment_timeline() -> str:
    """Get a timeline of production deployments and rollbacks
    to identify which deployments might have caused incidents."""
    data = _load_json(CICD_LOGS_PATH)

    deploys = [
        e for e in data
        if e.get("stage") in ("deploy_production", "deploy_staging", "rollback_production")
    ]

    if not deploys:
        return "No deployment events found."

    result = "Deployment Timeline:\n"
    for d in sorted(deploys, key=lambda x: x["timestamp"]):
        icon = "🚀" if "deploy" in d["stage"] else "🔄"
        status_icon = "✅" if d["status"] == "SUCCESS" else "❌"
        result += (
            f"  {icon} {d['timestamp']} | {d['stage']} | "
            f"Commit: {d['commit_hash']} | {status_icon} {d['status']}\n"
        )

    return result


# ═══════════════ RESOLUTION TOOLS ═══════════════

@tool
def search_resolution_faqs(query: str, n_results: int = 3) -> str:
    """Search the FAQ knowledge base for resolution steps matching the query.
    Uses semantic similarity search over the ChromaDB vector store of SRE runbooks.
    Returns a JSON list of matching documents with metadata and similarity scores.

    Args:
        query: A description of the incident or problem to find resolutions for.
        n_results: Number of top matching FAQs to return (default 3).
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=hf_ef,
        metadata={"hnsw:space": "cosine"},
    )

    results = collection.query(query_texts=[query], n_results=n_results)

    if not results["documents"][0]:
        return json.dumps([])

    documents = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        documents.append({
            "faq_id": meta.get("faq_id"),
            "category": meta.get("category"),
            "content": doc,
            "similarity_score": round(1 - dist, 4),
        })

    return documents


# ═══════════════ EMAIL TOOL ═══════════════

POWER_PLATFORM_URL = (
    "https://825521871ec5ed2ca0a95a6119079e.48.environment.api.powerplatform.com:443"
    "/powerautomate/automations/direct/workflows/77fa7e9fde374365ac1aaf731125657f"
    "/triggers/manual/paths/invoke?api-version=1"
)


@tool
def send_email_via_power_platform(email_html: str, timestamp: str) -> str:
    """Send an incident report email by triggering a Power Automate flow.

    Args:
        email_html: The full HTML content of the email body.
        timestamp: The incident timestamp to include in the payload.
    """
    payload = {
        "timestamp": timestamp,
        "email_html": email_html,
    }
    resp = requests.post(
        POWER_PLATFORM_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if resp.ok:
        return f"Email sent successfully (HTTP {resp.status_code})."
    return f"Failed to send email: HTTP {resp.status_code} — {resp.text[:500]}"

