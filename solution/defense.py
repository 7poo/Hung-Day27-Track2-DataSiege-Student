"""Cost-aware streaming defenses for all five Data Siege event types."""
from api import Verdict


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def _verdict(alert, pillar, reasons):
    """Return a consistent verdict and keep error responses fail-safe."""
    return Verdict(
        alert=bool(alert),
        confidence=0.95 if alert else 0.85,
        reason=", ".join(reasons) if reasons else "within calibrated limits",
        pillar=pillar,
    )


def _tool_error(result, pillar):
    if not isinstance(result, dict) or "error" in result:
        reason = result.get("error", "invalid toolkit response") if isinstance(result, dict) else "invalid toolkit response"
        return _verdict(False, pillar, [reason])
    return None


def _online_outlier(ctx, key, value, warmup=6, z_limit=2.2, direction=0):
    """One-pass z-score against prior observations; update only after scoring."""
    stats = ctx.state.setdefault("online_stats", {}).setdefault(
        key, {"n": 0, "mean": 0.0, "m2": 0.0}
    )
    flagged = False
    if stats["n"] >= warmup and stats["m2"] > 0.0:
        variance = stats["m2"] / (stats["n"] - 1)
        delta_from_mean = value - stats["mean"]
        if direction > 0:
            flagged = delta_from_mean > z_limit * (variance ** 0.5)
        elif direction < 0:
            flagged = delta_from_mean < -z_limit * (variance ** 0.5)
        else:
            flagged = abs(delta_from_mean) > z_limit * (variance ** 0.5)
    # Do not let a suspected anomaly poison the learned clean distribution.
    if not flagged:
        stats["n"] += 1
        delta = value - stats["mean"]
        stats["mean"] += delta / stats["n"]
        stats["m2"] += delta * (value - stats["mean"])
    return flagged


def check_data_batch(payload, ctx):
    p = ctx.tools.batch_profile(payload["batch_id"])
    error = _tool_error(p, "checks")
    if error:
        return error

    b = ctx.baseline
    reasons = []
    row_mid = (b["row_count_min"] + b["row_count_max"]) / 2.0
    row_span = b["row_count_max"] - b["row_count_min"]
    if p["row_count"] > row_mid + 0.05 * row_span:
        reasons.append("row count outside calibrated range")
    if p["null_rate"].get("customer_id", 0.0) > b["null_rate_max"]:
        reasons.append("customer_id null-rate spike")
    amount_mid = (b["mean_amount_min"] + b["mean_amount_max"]) / 2.0
    amount_half = (b["mean_amount_max"] - b["mean_amount_min"]) * 0.25
    if not amount_mid - amount_half <= p["mean_amount"] <= amount_mid + amount_half:
        reasons.append("amount distribution shift")
    if p["staleness_min"] > b["staleness_min_max"]:
        reasons.append("stale batch")
    if _online_outlier(ctx, "amount_std", p["std_amount"], warmup=5, z_limit=2.3):
        reasons.append("amount variance shift")
    if _online_outlier(ctx, "amount_mean", p["mean_amount"], z_limit=2.4):
        reasons.append("adaptive amount-mean anomaly")
    if _online_outlier(ctx, "batch_staleness", p["staleness_min"],
                       z_limit=1.95, direction=1):
        reasons.append("adaptive staleness anomaly")
    return _verdict(reasons, "checks", reasons)


def check_contract_checkpoint(payload, ctx):
    d = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    error = _tool_error(d, "contracts")
    if error:
        return error

    reasons = list(d.get("violations", []))
    if d["freshness_delay_min"] > ctx.baseline["freshness_delay_max_min"]:
        reasons.append("contract freshness SLA exceeded")
    return _verdict(reasons, "contracts", reasons)


def check_lineage_run(payload, ctx):
    g = ctx.tools.lineage_graph_slice(payload["run_id"])
    error = _tool_error(g, "lineage")
    if error:
        return error

    reasons = []
    # Producers use slightly different names for the declared graph; accept
    # the common envelope variants instead of coupling the defense to one.
    expected_upstream = payload.get("expected_upstream")
    if expected_upstream is None:
        expected_upstream = payload.get("expected_upstreams")
    if expected_upstream is None:
        expected_upstream = payload.get("declared_upstream")
    if expected_upstream is None:
        expected_upstream = payload.get("required_upstream")
    if expected_upstream is not None and set(g["actual_upstream"]) != set(expected_upstream):
        reasons.append("upstream lineage mismatch")
    expected_downstream = payload.get("expected_downstream_count")
    if expected_downstream is not None and g["actual_downstream_count"] != expected_downstream:
        reasons.append("downstream lineage mismatch")
    # A zero-output transform is always orphaned even if an older producer did
    # not put an explicit expected count into the event envelope.
    if g["actual_downstream_count"] <= 0:
        reasons.append("orphaned output")
    if len(g["actual_upstream"]) < 2:
        reasons.append("missing required upstream edge")
    if g["duration_ms"] > 0.87 * ctx.baseline["lineage_duration_ms_max"]:
        reasons.append("lineage runtime anomaly")
    return _verdict(reasons, "lineage", reasons)


def check_feature_materialization(payload, ctx):
    entity = payload["feature_view"]
    d = ctx.tools.feature_drift(entity, payload["batch_id"])
    error = _tool_error(d, "ai_infra")
    if error:
        return error

    reasons = []
    if d["mean_shift_sigma"] > 1.00 * ctx.baseline["feature_mean_shift_sigma_max"]:
        reasons.append("training-serving feature skew")
    return _verdict(reasons, "ai_infra", reasons)


def check_embedding_batch(payload, ctx):
    entity = payload["corpus"]
    d = ctx.tools.embedding_drift(entity, payload["chunk_batch_id"])
    error = _tool_error(d, "ai_infra")
    if error:
        return error

    reasons = []
    if d["centroid_shift"] > 0.90 * ctx.baseline["embedding_centroid_shift_max"]:
        reasons.append("embedding centroid drift")
    if d["avg_doc_age_days"] > 0.63 * ctx.baseline["corpus_avg_doc_age_days_max"]:
        reasons.append("stale retrieval corpus")
    if _online_outlier(ctx, "embedding_shift", d["centroid_shift"],
                       warmup=6, z_limit=1.3, direction=1):
        reasons.append("adaptive embedding drift")
    return _verdict(reasons, "ai_infra", reasons)
