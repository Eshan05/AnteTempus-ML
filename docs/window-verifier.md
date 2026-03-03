# Fixed-window verification

The verifier finds the transactions in one account's spending window and checks which were available by the prediction cutoff. It supports the arithmetic mean over fixed-width, aligned, half-open event windows. A query's cutoff must be at or after the window end. Source rows are immutable inserts with globally unique IDs, integer event and availability times in a shared clock, and exact integer-cent amounts. The supplied batch must contain the relevant rows, and availability times must reflect what a prediction could use. Source revisions, deletions, arbitrary predicates, and arbitrary SQL are out of scope. An unsupported query returns `unknown`; invalid or duplicate source records stop construction. Empty windows have the exact aggregate `(0, 0)`, not a numeric mean.

## Guarantee

For an account and window, let `B` be the source rows belonging to that window. Let `V` be the subset of `B` whose availability is at or before the prediction cutoff. Independent replay computes `(sum, count)` over `V`. The index sorts `B` by `(availability, ID)`, so `V` is exactly a prefix of that order. The index uses the prefix length and its exact cumulative sum to recover the replay value. If `V` is smaller than `B`, the final row in that order is a deterministic witness: its availability is after the cutoff. Otherwise every row in `B` is in `V`, so a passing batch agrees with replay. Equal numerical means alone do not establish safety: different input sets can have the same mean.

This is an argument about one operator under the stated assumptions, not a proof about a full pipeline or the truthfulness of timestamp metadata. Tests check both implementations against raw replay on generated data and cover boundaries, ties, invalid inputs, and coincidentally equal means.

## Benchmark

The reference also partitions by account and event window, so it never scans unrelated accounts or windows. It scans the relevant bucket at query time and independently computes the batch aggregate, as-of aggregate, and witness. The indexed method sorts each bucket once and stores availability and running sums. Each query then uses binary search. Both are timed with the same input validation and checked for identical complete outputs on every repeat.

Default workloads use 81,920 synthetic transactions and 100, 8,000, or 40,000 queries from a seeded sequence. Build, queries, and total time are separately reported as medians over three repeats with alternating order. Peak additional Python allocation includes setup and the output list, measured separately with `tracemalloc`; it excludes the shared input and is not process RSS. Timings exclude generating the input and serializing the report. The different batch sizes show how setup cost changes the comparison.

### DuckDB baseline

The DuckDB comparison runs the same workloads locally on one thread with two SQL plans: grouped replay and precomputed running sums with an [`ASOF LEFT JOIN`](https://duckdb.org/docs/2.0/guides/sql_features/asof_join). Both SQL plans evaluate each distinct (account, window, cutoff) once and then expand the results to all requests. Every repeat checks all four methods for exact agreement on the verdict, batch and visible sums and counts, and witness ID.

The reported medians use eight runs with rotating method order. DuckDB's setup includes opening an in-memory connection, converting and loading the shared Python rows, checking source IDs and availability, and building any prefix tables. Query time includes loading requests and returning Python findings. Package import, input generation, teardown, and report serialization are excluded. These wall-clock results compare the full batch operation from Python inputs on this machine, not raw SQL kernel time or serving latency. The SQL baseline assumes the benchmark's integer inputs fit in signed 64-bit columns. No process-memory comparison is reported.

Median total time in milliseconds on a Ryzen 5 5600H (Windows), seed 0:

| Queries | AT-ML scan | AT-ML index | DuckDB replay | DuckDB prefix |
| -------: | ---------: | ----------: | ------------: | ------------: |
|      100 |     **77** |         101 |            83 |           151 |
|    8,000 |        150 |     **111** |           137 |           203 |
|   40,000 |        584 |     **189** |           285 |           411 |

A second seed preserved the ordering at all three batch sizes. At 40,000 queries, AT-ML index took 177 ms and DuckDB replay took 283 ms. DuckDB replay was still nearly twice as fast as AT-ML scan at that batch size.

To rerun the comparison and generate `results/raw/window-duckdb-bench.json`, use this command from the repository root:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 uv run --no-project --no-cache \
  --with duckdb==2.0.1 --with numpy==2.5.5 python -m ante_tempus_ml.window_duckdb_bench
```

This is an equivalent fixed-window audit, not an isolated join microbenchmark. [ProvSQL](https://provsql.org/docs/user/temporal.html) already tracks temporal provenance and is not measured here. These results do not establish research novelty or a general performance advantage.
