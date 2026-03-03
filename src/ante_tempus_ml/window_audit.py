"""Verify fixed, half-open event windows over immutable transaction rows."""

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import accumulate
from typing import Literal


@dataclass(frozen=True, slots=True)
class Transaction:
    id: int
    account: int
    event_at: int
    available_at: int
    cents: int


@dataclass(frozen=True, slots=True)
class Query:
    account: int
    start: int
    cutoff: int
    operation: str = "mean"


@dataclass(frozen=True, slots=True)
class Finding:
    verdict: Literal["pass", "leak", "unknown"]
    batch: tuple[int, int] | None = None  # Exact (sum in cents, count).
    replay: tuple[int, int] | None = None
    witness: int | None = None


def group(rows: Iterable[Transaction], width: int) -> dict[tuple[int, int], list[Transaction]]:
    if type(width) is not int or width <= 0:
        raise ValueError("Window width must be a positive integer")
    buckets = defaultdict(list)
    ids = set()
    for row in rows:
        if not all(type(v) is int for v in (row.id, row.account, row.event_at, row.available_at, row.cents)):
            raise ValueError("Transactions require integer fields")
        if row.id in ids:
            raise ValueError("Duplicate or versioned transaction ID; only immutable inserts are supported")
        if row.available_at < row.event_at:
            raise ValueError("A transaction cannot be available before it occurs")
        ids.add(row.id)
        buckets[row.account, row.event_at // width * width].append(row)
    return dict(buckets)


def supported(query: Query, width: int) -> bool:
    return (query.operation == "mean" and all(type(v) is int for v in (query.account, query.start, query.cutoff))
            and query.start % width == 0 and query.start + width <= query.cutoff)


class ScanAudit:
    """Reference: scan just the relevant event bucket for every prediction."""

    def __init__(self, rows: Iterable[Transaction], width: int):
        self.width = width
        self.buckets = group(rows, width)

    def query(self, query: Query) -> Finding:
        if not supported(query, self.width):
            return Finding("unknown")
        rows = self.buckets.get((query.account, query.start), ())
        batch = seen_sum = seen_count = 0
        late = None
        for row in rows:
            batch += row.cents
            if row.available_at <= query.cutoff:
                seen_sum += row.cents
                seen_count += 1
            elif late is None or (row.available_at, row.id) > (late.available_at, late.id):
                late = row
        return Finding("leak" if late else "pass", (batch, len(rows)), (seen_sum, seen_count), late.id if late else None)


@dataclass(frozen=True, slots=True)
class Prefix:
    available: tuple[int, ...]
    totals: tuple[int, ...]
    last_id: int


class IndexedAudit:
    """Recover the visible prefix and name the latest late contributor."""

    def __init__(self, rows: Iterable[Transaction], width: int):
        self.width = width
        self.buckets = {}
        for key, bucket in group(rows, width).items():
            bucket.sort(key=lambda row: (row.available_at, row.id))
            self.buckets[key] = Prefix(tuple(row.available_at for row in bucket),
                                       tuple(accumulate((row.cents for row in bucket), initial=0)), bucket[-1].id)

    def query(self, query: Query) -> Finding:
        if not supported(query, self.width):
            return Finding("unknown")
        prefix = self.buckets.get((query.account, query.start))
        if prefix is None:
            return Finding("pass", (0, 0), (0, 0))
        count = len(prefix.available)
        visible = bisect_right(prefix.available, query.cutoff)
        return Finding("leak" if visible < count else "pass", (prefix.totals[-1], count),
                       (prefix.totals[visible], visible), prefix.last_id if visible < count else None)


def example() -> tuple[int, tuple[Transaction, ...], Query]:
    return 60, (Transaction(1, 7, 10, 10, 2000), Transaction(2, 7, 30, 30, 4000),
                Transaction(3, 7, 50, 75, 9000)), Query(7, 0, 60)
