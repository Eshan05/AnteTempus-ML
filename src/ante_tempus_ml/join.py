"""Point-in-time joins of feature updates onto events. Timestamps are int64 (e.g. seconds).

Availability is tracked per row with `available_at`, or derived as `feat_ts + delay`. An event at time `t`
uses the same-entity row with the latest eligible availability (<= t, or < t with allow_equal=False).
TTL is measured from availability. Ties favor the last input row; no match returns index -1.
"""
from __future__ import annotations
import numpy as np


_MIN_TIME = -(1 << 63)
_MAX_TIME = (1 << 63) - 1
_MAX_AGE = _MAX_TIME - _MIN_TIME
_PAIR_DTYPE = np.dtype([("entity", np.intp), ("time", np.int64)])


def _availability(feat_ts, delay, available_at):
    if available_at is None:
        offset = int(np.int64(delay))
        if not offset or not feat_ts.size:
            return feat_ts
        if (offset > 0 and int(feat_ts.max()) > _MAX_TIME - offset
                or offset < 0 and int(feat_ts.min()) < _MIN_TIME - offset):
            raise ValueError("feat_ts + delay falls outside int64")
        return feat_ts + offset
    if delay != 0:
        raise ValueError("Use either available_at or delay, not both")
    available = np.asarray(available_at, dtype=np.int64)
    if available.shape != feat_ts.shape or np.any(available < feat_ts):
        raise ValueError("available_at must match feat_ts and cannot precede it")
    return available


def _pair_keys(rank, timestamp):
    keys = np.empty(rank.size, dtype=_PAIR_DTYPE)
    keys["entity"] = rank
    keys["time"] = timestamp
    return keys


def asof_index(ev_entity, ev_ts, feat_entity, feat_ts, *, delay=0, ttl=None, allow_equal=True, available_at=None):
    """Index into the feature arrays of the value each event may use, or -1."""
    ev_entity, ev_ts = np.asarray(ev_entity), np.asarray(ev_ts, dtype=np.int64)
    feat_entity, feat_ts = np.asarray(feat_entity), np.asarray(feat_ts, dtype=np.int64)
    if ev_ts.ndim != 1 or ev_entity.shape != ev_ts.shape or feat_ts.ndim != 1 or feat_entity.shape != feat_ts.shape:
        raise ValueError("Entity and timestamp arrays must be one-dimensional with matching lengths")
    avail = _availability(feat_ts, delay, available_at)
    if not ev_ts.size or not avail.size:
        return np.full(ev_ts.shape, -1, dtype=np.intp)
    uniq = np.unique(feat_entity)
    f_rank = np.searchsorted(uniq, feat_entity)
    e_rank = np.searchsorted(uniq, ev_entity)
    e_rank_c = np.minimum(e_rank, len(uniq) - 1)
    e_known = uniq[e_rank_c] == ev_entity
    tmin = min(int(avail.min()), int(ev_ts.min()))
    tmax = max(int(avail.max()), int(ev_ts.max()))
    span = tmax - tmin + 1
    # Pack keys only when the largest rank and time offset can both fit.
    if span * len(uniq) <= _MAX_TIME:
        f_key = f_rank.astype(np.int64) * span + (avail - tmin)
        e_key = e_rank_c.astype(np.int64) * span + (ev_ts - tmin)
    else:
        f_key = _pair_keys(f_rank, avail)
        e_key = _pair_keys(e_rank_c, ev_ts)
    order = np.argsort(f_key, kind="stable")
    f_sorted = f_key[order]
    pos = np.searchsorted(f_sorted, e_key, side="right" if allow_equal else "left") - 1
    ok = e_known & (pos >= 0)
    pos_c = np.maximum(pos, 0)
    cand = order[pos_c]
    ok &= f_rank[cand] == e_rank_c
    if ttl is not None:
        if not ttl >= 0:
            ok[:] = False
        elif ttl < _MAX_AGE:
            # The exact age of a valid candidate can exceed signed int64.
            age = ev_ts.view(np.uint64) - avail[cand].view(np.uint64)
            ok &= age <= np.uint64(ttl)
    return np.where(ok, cand, -1)


def latest_index(ev_entity, feat_entity, feat_ts):
    """The common bug: the latest value ever written for the entity, whatever the event time."""
    ev_entity, feat_entity, feat_ts = np.asarray(ev_entity), np.asarray(feat_entity), np.asarray(feat_ts)
    order = np.lexsort((feat_ts, feat_entity))
    fe = feat_entity[order]
    last = np.r_[fe[1:] != fe[:-1], True]
    ents, idx = fe[last], order[last]
    pos = np.minimum(np.searchsorted(ents, ev_entity), len(ents) - 1)
    return np.where(ents[pos] == ev_entity, idx[pos], -1)


def day_granularity_index(ev_entity, ev_ts, feat_entity, feat_ts, day=86400):
    """Second common bug: joining on calendar dates, so same-day values later than the event are used."""
    return asof_index(ev_entity, (np.asarray(ev_ts) // day) * day + (day - 1), feat_entity, feat_ts)


def take(values, idx, fill=np.nan):
    values = np.asarray(values, dtype=float)
    if not values.size and np.all(idx < 0):
        return np.full(idx.shape, fill, dtype=float)
    out = values[np.maximum(idx, 0)]
    return np.where(idx >= 0, out, fill)


def leakage_audit(idx, feat_ts, ev_ts, delay=0, *, available_at=None):
    """Share of joined rows that used a value written after the event (future leak) or not yet available
    at the event (unavailable). Rows with no match are excluded."""
    m = idx >= 0
    feat_ts = np.asarray(feat_ts, dtype=np.int64)
    ft = feat_ts[idx[m]]
    avail = _availability(feat_ts, delay, available_at)[idx[m]]
    et = np.asarray(ev_ts, dtype=np.int64)[m]
    return {"rows_joined": int(m.sum()), "future_written": float((ft > et).mean()) if m.any() else 0.0,
            "not_yet_available": float((avail > et).mean()) if m.any() else 0.0}


def brute_force_index(ev_entity, ev_ts, feat_entity, feat_ts, *, delay=0, ttl=None, allow_equal=True,
                      available_at=None):
    """Slow reference implementation, used only in tests."""
    avail = _availability(np.asarray(feat_ts, dtype=np.int64), delay, available_at)
    out = np.full(len(ev_entity), -1)
    for i, (e, t) in enumerate(zip(ev_entity, ev_ts)):
        best = -1
        for j, (fe, a) in enumerate(zip(feat_entity, avail)):
            if fe == e and (a <= t if allow_equal else a < t) and (ttl is None or int(t) - int(a) <= ttl):
                if best < 0 or a >= avail[best]:
                    best = j
        out[i] = best
    return out
