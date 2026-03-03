"""Point-in-time feature lookup on a separate timeline for each entity.

Timestamps are int64. An update is eligible at its recorded availability, or
its write time plus delay. TTL limits age since availability. Equal times favor
the last feature row in input order; missing matches return -1.
"""

from bisect import bisect_left, bisect_right
from collections import defaultdict

import numpy as np


_TIME_LIMITS = np.iinfo(np.int64)


def _availability(written, delay, available_at):
    if available_at is not None:
        ready = np.asarray(available_at, dtype=np.int64)
        if delay != 0 or ready.shape != written.shape or np.any(ready < written):
            raise ValueError("available_at must match and not precede feat_ts, and cannot be combined with delay")
        return ready

    offset = int(np.int64(delay))
    if written.size and offset:
        lo, hi = int(written.min()) + offset, int(written.max()) + offset
        if lo < _TIME_LIMITS.min or hi > _TIME_LIMITS.max:
            raise ValueError("feat_ts + delay falls outside int64")
        return written + offset
    return written


def asof_index(ev_entity, ev_ts, feat_entity, feat_ts, *, delay=0, ttl=None, allow_equal=True, available_at=None):
    """Find one eligible feature index per event using that entity's timeline."""
    events, cutoffs = np.asarray(ev_entity), np.asarray(ev_ts, dtype=np.int64)
    entities, written = np.asarray(feat_entity), np.asarray(feat_ts, dtype=np.int64)
    for keys, times in ((events, cutoffs), (entities, written)):
        if times.ndim != 1 or keys.shape != times.shape:
            raise ValueError("Entity and timestamp arrays must be one-dimensional with matching lengths")

    ready = _availability(written, delay, available_at)
    selected = np.full(cutoffs.size, -1, dtype=np.intp)
    if isinstance(ttl, np.generic):
        ttl = ttl.item()
    if not selected.size or not ready.size or (ttl is not None and not ttl >= 0):
        return selected

    timelines = defaultdict(list)
    for row, (entity, stamp) in enumerate(zip(entities.tolist(), ready.tolist())):
        timelines[entity].append((stamp, row))
    for timeline in timelines.values():
        timeline.sort()

    for event, (entity, cutoff) in enumerate(zip(events.tolist(), cutoffs.tolist())):
        if entity != entity:
            continue
        timeline = timelines.get(entity)
        if not timeline:
            continue
        # Row numbers break ties; the bounds include or exclude all rows at the cutoff.
        pos = (bisect_right(timeline, (cutoff, written.size)) if allow_equal
               else bisect_left(timeline, (cutoff, -1)))
        if pos:
            stamp, row = timeline[pos - 1]
            if ttl is None or cutoff - stamp <= ttl:
                selected[event] = row
    return selected


def take(values, idx, fill=np.nan):
    """Gather matched values, leaving missing rows filled."""
    indices = np.asarray(idx, dtype=np.intp)
    found = indices >= 0
    result = np.full(indices.shape, fill, dtype=float)
    result[found] = np.asarray(values, dtype=float)[indices[found]]
    return result


def leakage_audit(idx, feat_ts, ev_ts, delay=0, *, available_at=None):
    """Count matched rows and measure how often they violate the event cutoff."""
    matches = np.asarray(idx, dtype=np.intp)
    found = matches >= 0
    rows = matches[found]
    written = np.asarray(feat_ts, dtype=np.int64)
    ready = _availability(written, delay, available_at)
    cutoffs = np.asarray(ev_ts, dtype=np.int64)[found]
    count = rows.size
    return {
        "rows_joined": count,
        "future_written": np.count_nonzero(written[rows] > cutoffs) / count if count else 0.0,
        "not_yet_available": np.count_nonzero(ready[rows] > cutoffs) / count if count else 0.0,
    }


def brute_force_index(ev_entity, ev_ts, feat_entity, feat_ts, *, delay=0, ttl=None, allow_equal=True,
                      available_at=None):
    """Scan the eligible candidates to provide a test reference."""
    ready = _availability(np.asarray(feat_ts, dtype=np.int64), delay, available_at).tolist()
    features = tuple(zip(feat_entity, ready))
    result = []
    for entity, time in zip(ev_entity, ev_ts):
        cutoff = int(time)
        candidates = ((stamp, row) for row, (owner, stamp) in enumerate(features)
                      if owner == entity and (stamp < cutoff or (allow_equal and stamp == cutoff))
                      and (ttl is None or cutoff - stamp <= ttl))
        result.append(max(candidates, default=(0, -1))[1])
    return np.array(result, dtype=np.intp)
