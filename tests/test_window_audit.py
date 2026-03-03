from random import Random

import pytest

from ante_tempus_ml.provenance_report import window_html
from ante_tempus_ml.window_audit import Finding, IndexedAudit, Query, ScanAudit, Transaction, example
from ante_tempus_ml.window_bench import compare


@pytest.mark.parametrize("factory", [ScanAudit, IndexedAudit])
def test_window_bounds_and_availability_boundaries(factory):
    width, rows, query = example()
    rows += (Transaction(4, 7, 60, 61, 999), Transaction(5, 8, 30, 50, 999),
             Transaction(6, 7, 55, 75, 1000), Transaction(7, 7, 59, 60, 500))
    audit = factory(rows, width)
    assert audit.query(query) == Finding("leak", (16500, 5), (6500, 3), 6)
    assert audit.query(Query(7, 0, 75)) == Finding("pass", (16500, 5), (16500, 5))
    assert audit.query(Query(7, 60, 120)) == Finding("pass", (999, 1), (999, 1))
    assert audit.query(Query(0, 0, 60)) == Finding("pass", (0, 0), (0, 0))
    assert audit.query(Query(7, 0, 59)) == Finding("unknown")
    assert audit.query(Query(7, 1, 80)) == Finding("unknown")
    assert audit.query(Query(7, 0, 80, operation="median")) == Finding("unknown")


@pytest.mark.parametrize("factory", [ScanAudit, IndexedAudit])
def test_unsupported_source_versions_are_not_silently_certified(factory):
    width, rows, _ = example()
    with pytest.raises(ValueError, match="versioned transaction"):
        factory(rows + (Transaction(1, 7, 10, 90, 8000),), width)
    with pytest.raises(ValueError, match="before it occurs"):
        factory((Transaction(9, 7, 10, 8, 100),), width)


def test_identical_average_is_not_evidence_of_no_leak():
    rows = [Transaction(1, 0, 2, 2, 500), Transaction(2, 0, 3, 11, 500)]
    assert IndexedAudit(rows, 10).query(Query(0, 0, 10)) == Finding("leak", (1000, 2), (500, 1), 2)


def test_both_algorithms_agree_with_raw_replay_across_generated_cases():
    for seed in range(16):
        rng = Random(seed)
        rows = []
        for i in range(200):
            event = rng.randrange(90)
            rows.append(Transaction(i, rng.randrange(4), event, event + rng.randrange(30), rng.randrange(-500, 501)))
        rng.shuffle(rows)
        scans, indexed = ScanAudit(rows, 10), IndexedAudit(rows, 10)
        for _ in range(80):
            start = rng.randrange(9) * 10
            q = Query(rng.randrange(5), start, start + 10 + rng.randrange(30))
            batch = [r for r in rows if r.account == q.account and q.start <= r.event_at < q.start + 10]
            visible = [r for r in batch if r.available_at <= q.cutoff]
            late = [r for r in batch if r.available_at > q.cutoff]
            witness = max(late, key=lambda r: (r.available_at, r.id)).id if late else None
            expected = Finding("leak" if late else "pass", (sum(r.cents for r in batch), len(batch)),
                               (sum(r.cents for r in visible), len(visible)), witness)
            assert scans.query(q) == indexed.query(q) == expected


def test_benchmark_computes_both_outputs_and_includes_setup():
    width, rows, query = example()
    result = compare(rows, width, [query, Query(query.account, query.start, 80)], repeats=1)
    assert result["verdicts"] == {"leak": 1, "pass": 1}
    for method in ("scan", "index"):
        metrics = result[method]
        assert metrics["total_ms"] == pytest.approx(metrics["build_ms"] + metrics["query_ms"])
        assert metrics["peak_bytes"] > 0

    html = window_html({"schema": 1, "rows": 3, "seed": 0, "repeats": 1, "plans": [result]})
    assert '<a href="#fixed-row-3">Transaction 3</a>' in html
    assert "Equivalent replay benchmark" in html and "Peak (MiB)" in html
