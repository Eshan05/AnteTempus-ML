"""Generate a standalone, browsable audit report."""

import argparse
import json
from html import escape
from importlib.resources import files
from pathlib import Path
from statistics import mean
from string import Template

from .churn_labels import example as churn_example
from .ci_duration import Job, study
from .ci_source import COHORTS, Cohort
from .provenance import Fit, Scenario, audit, scenarios
from .target_leakage import TRAIN_END, audit_outcome, example_sources, target_trial
from .validation import GAP, SEEDS, SERIES_DEV_END, SERIES_TRAIN, scaling_trial, series_trial
from .window_audit import IndexedAudit, example as window_example


def currency(value: float | None) -> str:
    return "No value" if value is None else f"${value:,.2f}"


def scaling_result(label: str, fit: Fit | None) -> str:
    if fit is None:
        return f"<div><dt>{label}</dt><dd>No fit</dd></div>"
    return f"""<div><dt>{label}</dt><dd>{fit.z_score:.2f}
      <span class="fit-detail">Fit mean {currency(fit.mean)} &middot; scale {currency(fit.scale)}</span></dd></div>"""


def case_html(case: Scenario, opened: bool) -> str:
    result = audit(case)
    cutoff = case.cutoff.isoformat()
    batch = set(result.batch_inputs)
    replay = set(result.replay_inputs)
    witnesses = set(result.violations)
    scaling = case.prediction_dollars is not None
    batch_label, replay_label = ("Batch fit", "Replay fit") if scaling else ("Batch", "Replay")
    records = []
    for row in sorted(case.records, key=lambda r: (r.available_day, r.row_id)):
        usage = [label for present, label in ((row in batch, batch_label), (row in replay, replay_label)) if present]
        hit = row in witnesses
        records.append(f"""
          <li id="{case.key}-{row.row_id}" class="record{' witness' if hit else ''}">
            <div><strong>{row.row_id}</strong> <span>{currency(row.dollars)}</span></div>
            <div><span>Event {row.event_day.isoformat()}</span> <span>Available {row.available_day.isoformat()}</span></div>
            <div>{' / '.join(usage) if usage else 'Not used'}{' <strong class="bad">&middot; After cutoff</strong>' if hit else ''}</div>
          </li>""")
    status = "Too late" if result.timestamp_flag else "Eligible"
    source = "Flagged" if witnesses else "Passed"
    links = ", ".join(f'<a href="#{case.key}-{row.row_id}">{row.row_id}</a>' for row in result.violations)
    if scaling:
        results = scaling_result("Batch z-score", result.batch_fit) + scaling_result("Replay z-score", result.replay_fit)
        input_note = f'<p class="cause">Prediction input: <strong>{currency(case.prediction_dollars)}</strong></p>'
    else:
        results = f"""<div><dt>Batch average</dt><dd>{currency(result.batch_value)}</dd></div>
          <div><dt>Replay average</dt><dd>{currency(result.replay_value)}</dd></div>"""
        input_note = ""
    joined = result.joined
    joined_label = "Cutoff update" if joined.is_candidate else "Previous update"
    joined_value = f"{joined.value:.2f}" if scaling and joined.value is not None else currency(joined.value)
    joined_status = "Unavailable inputs" if joined.violations else "No unavailable inputs"
    return f"""
      <details id="{case.key}"{' open' if opened else ''}>
        <summary><span class="case-title">{escape(case.title)}</span>
          <span class="checks"><span>Candidate timestamp <b class="{'bad' if result.timestamp_flag else 'good'}">{status}</b></span>
          <span>Candidate inputs <b class="{'bad' if witnesses else 'good'}">{source}</b></span></span>
        </summary>
        <dl class="values">
          <div><dt>Prediction cutoff</dt><dd>{cutoff}</dd></div>
          <div><dt>Candidate availability</dt><dd>{case.declared_available.isoformat()}</dd></div>
          {results}
        </dl>
        <div class="trail">
          <p class="cause">As-of join: <strong>{joined_label}</strong> ({joined.available_day.isoformat()}) &middot;
            {joined_value} &middot; <span class="{'bad' if joined.violations else 'good'}">{joined_status}</span></p>
          <h3>{'Scaler fit records' if scaling else 'Source records'}</h3>
          {input_note}
          <p class="cause">{f'Candidate inputs unavailable at prediction time: {links}' if links else 'All candidate inputs were available at prediction time.'}</p>
          <ul>{''.join(records)}</ul>
        </div>
      </details>"""


def auc_chart(rows: tuple[tuple[str, float], ...], label: str = "Mean ROC-AUC") -> str:
    items = "".join(f"""
      <li><span>{escape(label)}</span>
        <span class="auc-track" aria-hidden="true"><i style="width:{auc:.2%}"></i></span>
        <strong>{auc:.4f}</strong></li>""" for label, auc in rows)
    return f'<p class="axis-label">{label} &middot; 0 to 1 &middot; Higher is better</p><ul class="auc-chart">{items}</ul>'


def validation_html() -> str:
    scaling = [scaling_trial(seed) for seed in SEEDS]
    series = [series_trial(seed) for seed in SEEDS]
    scaling_chart = auc_chart((
        ("Full-data fit", mean(r.full_fit.auc for r in scaling)),
        ("Training-only fit", mean(r.train_fit.auc for r in scaling)),
    ))
    series_chart = auc_chart((
        ("Random validation", mean(r.random_auc for r in series)),
        ("Forward validation", mean(r.forward_auc for r in series)),
        ("Future holdout", mean(r.future_auc for r in series)),
    ))
    fit_rows = scaling[0].full_fit.future_fit_rows
    fold_size = SERIES_DEV_END - SERIES_TRAIN - GAP
    exposed = sum(r.random_exposed for r in series)
    forward_exposed = sum(r.forward_exposed for r in series)
    denominator = fold_size * len(series)
    scoring_rows = "".join(f"""
      <tr><th scope="row">{s.seed}</th><td>{s.full_fit.auc:.4f}</td><td>{s.train_fit.auc:.4f}</td>
        <td>{r.random_auc:.4f}</td><td>{r.forward_auc:.4f}</td><td>{r.future_auc:.4f}</td></tr>"""
        for s, r in zip(scaling, series))
    witness = next((r for r in series if r.witness is not None), None)
    example = ""
    if witness is not None and witness.witness is not None:
        observed, training = witness.witness
        example = f" Seed {witness.seed}: validation step {observed} had training step {training} in its future."
    return f"""
      <div class="study">
        <h3>Scaler fitting on a chronological holdout</h3>
        {scaling_chart}
        <p class="cause">{fit_rows} holdout rows entered each full-data scaler fit; none entered training-only fits.
        Both models used training labels only and scored the same holdout. Mean log loss:
        {mean(r.full_fit.loss for r in scaling):.4f} full-data fit;
        {mean(r.train_fit.loss for r in scaling):.4f} training-only fit.</p>
      </div>
      <div class="study">
        <h3>Validation on a dependent time series</h3>
        {series_chart}
        <p class="cause">Validation rows with a training row in the next {GAP} steps:
        {exposed}/{denominator} random, {forward_exposed}/{denominator} forward.{example}</p>
        <p class="cause">The classifier uses a noisy measurement and the known time step. Development folds have matched
        sizes but score different rows. The future holdout is a separate reference: the classifier is refit on all development
        data, then evaluated after a gap.</p>
      </div>
      <details class="study-seeds">
        <summary>ROC-AUC by seed</summary>
        <div class="score-scroll" tabindex="0" role="region" aria-label="Per-seed ROC-AUC">
        <table><thead><tr><th>Seed</th><th>Full fit</th><th>Train fit</th><th>Random</th><th>Forward</th><th>Future</th></tr></thead>
        <tbody>{scoring_rows}</tbody></table></div>
      </details>"""


def target_html() -> str:
    build = TRAIN_END
    trials = [target_trial(seed) for seed in SEEDS]
    chart = auc_chart((
        ("Retrospective badge", mean(r.retrospective_auc for r in trials)),
        ("Same model, replay", mean(r.replay_auc for r in trials)),
        ("Clean model, replay", mean(r.clean_auc for r in trials)),
    ))
    records = []
    for source in example_sources(build):
        finding = audit_outcome(build, build, source)
        hit = finding.uses_own_target
        records.append(f"""
          <li id="outcome-{source.source_build}" class="record{' witness' if hit else ''}">
            <div><strong>{escape(source.feature)}</strong> <span>Build B{source.source_build}</span></div>
            <div><span>Recorded step {source.recorded_at}</span> <span>Completed step {source.completed_at}</span></div>
            <div><span>{'Timestamp flagged' if finding.late_by_timestamp else 'Timestamp passed'}</span>
                 <strong class="{'bad' if hit else 'good'}">{'Own outcome' if hit else 'Prior outcome'}</strong></div>
          </li>""")
    return f"""
      <div class="study">
        <h3>Build outcome on a chronological holdout</h3>
        {chart}
        <p class="cause">The retrospective badge is a noisy encoding of the result. Live replay removes it from
        the same fitted model; the clean control trains without it. Both models score the same holdout rows.</p>
      </div>
      <details open>
        <summary><span class="case-title">Prediction for build B{build}</span></summary>
        <div class="trail">
          <h3>Outcome sources</h3>
          <p class="cause">Prediction at step {build}. The backdated badge points to the outcome of
          <a href="#outcome-{build}">this same build</a>.</p>
          <ul>{''.join(records)}</ul>
        </div>
      </details>"""


def churn_html() -> str:
    rows = []
    for row in churn_example():
        at = row.prediction.at
        cancellation = f"Day {(row.cancellation - at).days}" if row.cancellation else "None observed"
        coverage = (row.prediction.observed_through - at).days
        label = "Excluded" if row.label is None else "Churned" if row.label else "No churn"
        rows.append(f"""<tr><th scope="row">{row.prediction.customer}</th><td>{row.tickets}</td>
          <td>{cancellation}</td><td>{coverage} days</td><td>{label}</td></tr>""")
    return f"""
      <div class="study">
        <p class="cause">The newer ticket count (7) was written before prediction but published the next day.
          The backward join returns the older count (2). The label looks forward for cancellation
          after prediction through day 30, inclusive. It stays unknown if the outcome data ends early.</p>
      </div>
      <div class="score-scroll" role="region" aria-label="30-day churn label examples" tabindex="0">
        <table><thead><tr><th scope="col">Customer</th><th scope="col">Tickets at prediction</th>
          <th scope="col">Cancellation</th><th scope="col">Follow-up</th><th scope="col">30-day label</th>
        </tr></thead><tbody>{''.join(rows)}</tbody></table>
      </div>"""


def ci_cohort(dataset: dict) -> Cohort:
    identity = dataset.get("repo"), dataset.get("workflow"), dataset.get("created_window"), dataset.get("attempt")
    for cohort in COHORTS.values():
        if identity == (cohort.repo, cohort.workflow, cohort.created, 1):
            return cohort
    raise ValueError("Expected a pinned first-attempt cohort from ante_tempus_ml.ci_source")


def ci_html(dataset: dict) -> str:
    cohort = ci_cohort(dataset)
    report = study(dataset["jobs"])
    defined = sum(w.scores[0].auc is not None for w in report.windows)
    chart = auc_chart(tuple((s.name, s.auc) for s in report.scores if s.auc is not None),
                      "Mean window ROC-AUC") if defined else '<p class="cause">No windows with both outcomes.</p>'
    precision = "; ".join(f"{s.name}: {s.ap:.4f}" for s in report.scores if s.ap is not None)
    losses = "; ".join(f"{s.name}: {s.loss:.4f}" for s in report.scores)
    holdout = sum(w.holdout for w in report.windows)
    holdout_failures = sum(w.holdout_failures for w in report.windows)
    def auc_cell(auc):
        return f"<td>{auc:.4f}</td>" if auc is not None else "<td>n/a</td>"

    windows = "".join(f"""<tr><th scope="row">{i}</th><td>{w.cutoff:%m-%d %H:%M}</td><td>{w.train:,}</td>
      <td>{w.purged}</td><td>{w.holdout:,}</td><td>{w.holdout_failures}</td>
      {''.join(auc_cell(score.auc) for score in w.scores)}</tr>"""
        for i, w in enumerate(report.windows, 1))
    row = report.witness

    def record(job: Job, title: str, flagged: bool) -> str:
        url = f"https://github.com/{cohort.repo}/actions/runs/{job.run_id}/job/{job.id}"
        return f"""
          <li class="record{' witness' if flagged else ''}">
            <div><strong>{title}</strong><a href="{url}">{job.id}</a></div>
            <div><span>Start {job.started:%Y-%m-%d %H:%M:%S %z}</span>
            <span>End {job.completed:%Y-%m-%d %H:%M:%S %z}</span></div>
            <div>{job.seconds:,.0f} s</div>
          </li>"""

    witnesses = record(row.job, "Current job", True)
    if row.overlapping_predecessor:
        witnesses += record(row.overlapping_predecessor, "Earlier start, not yet complete", True)
    if row.previous_completed:
        witnesses += record(row.previous_completed, "Known prior duration", False)
    return f"""
      <article class="ci-cohort" aria-labelledby="ci-{cohort.key}">
        <div class="study">
          <h3 id="ci-{cohort.key}">{escape(cohort.title)} &middot; {escape(cohort.created)}</h3>
          {chart}
          <p class="cause">{holdout:,} evaluated jobs with {holdout_failures} failures across four non-overlapping windows.
          AUC is averaged over the {defined} windows with both outcomes. {report.excluded:,} source records were ineligible.</p>
          {f'<p class="cause">Mean average precision in those windows: {precision}.</p>' if precision else ''}
          <p class="cause">Log loss across all jobs: {losses}.</p>
          <p class="cause">Own final duration unavailable for {report.own_unavailable:,}/{holdout:,} evaluated jobs.
          {report.with_history:,} had a known prior duration; {report.with_overlap:,} had an earlier-started same-name job
          from another run that was not yet complete.</p>
        </div>
        <details>
          <summary><span class="case-title">Forward evaluation windows</span></summary>
          <div class="score-scroll" tabindex="0" role="region" aria-label="Forward evaluation windows">
          <table><thead><tr><th>Window</th><th>Cutoff UTC</th><th>Train</th><th>Purged</th><th>Test</th><th>Failures</th>
          <th>No duration AUC</th><th>Prior AUC</th><th>Current AUC</th></tr></thead>
          <tbody>{windows}</tbody></table></div>
        </details>
        <details open>
          <summary><span class="case-title">{escape(row.job.name)} &middot; workflow run {row.job.run_id}</span></summary>
          <div class="trail"><h3>Recorded start and completion</h3><ul>{witnesses}</ul></div>
        </details>
      </article>"""


def ci_section(datasets: list[dict]) -> str:
    if not datasets:
        return ""
    if len({ci_cohort(dataset).key for dataset in datasets}) != len(datasets):
        raise ValueError("Pass at most one snapshot per CI cohort")
    articles = "".join(ci_html(dataset) for dataset in datasets)
    return f"""<section aria-labelledby="ci-duration">
      <h2 id="ci-duration">Real CI job duration</h2>
      {articles}
    </section>"""


def window_html(benchmark: dict | None = None) -> str:
    width, source, query = window_example()
    finding = IndexedAudit(source, width).query(query)
    assert finding.batch is not None and finding.replay is not None
    batch_sum, batch_count = finding.batch
    replay_sum, replay_count = finding.replay
    records = "".join(f"""<li class="record{' witness' if row.id == finding.witness else ''}" id="fixed-row-{row.id}">
      <div><strong>Transaction {row.id}</strong> <span>{currency(row.cents / 100)}</span></div>
      <div><span>Occurred at {row.event_at} s</span> <span>Available at {row.available_at} s</span></div>
      <div>{'After cutoff' if row.available_at > query.cutoff else 'Known at cutoff'}</div></li>""" for row in source)
    cost = ""
    if benchmark is not None:
        if benchmark.get("schema") != 1:
            raise ValueError("Expected a benchmark produced by ante_tempus_ml.window_bench")
        table = []
        for plan in benchmark["plans"]:
            for key, label in (("scan", "Scan replay"), ("index", "Indexed")):
                m = plan[key]
                table.append(f"""<tr><th scope="row">{plan['queries']:,} &middot; {label}</th>
                  <td>{m['build_ms']:.1f}</td><td>{m['query_ms']:.1f}</td><td>{m['total_ms']:.1f}</td>
                  <td>{m['peak_bytes'] / 1024**2:.2f}</td></tr>""")
        cost = f"""
          <div class="study"><h3>Equivalent replay benchmark</h3>
            <p class="cause">{benchmark['rows']:,} synthetic rows; seed {benchmark['seed']}; median of {benchmark['repeats']}
            runs. Both methods returned identical findings. Peak extra allocation excludes the shared input.</p>
          </div>
          <div class="score-scroll" tabindex="0" role="region" aria-label="Window benchmark costs">
          <table><thead><tr><th>Queries and method</th><th>Build (ms)</th><th>Query (ms)</th>
          <th>Total (ms)</th><th>Peak (MiB)</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>"""
    return f"""
      <section aria-labelledby="fixed-window">
        <h2 id="fixed-window">Fixed-window verifier</h2>
        <details open>
          <summary><span class="case-title">Account {query.account}: {query.start} to {query.start + width} s, excluding the end</span></summary>
          <dl class="values">
            <div><dt>Cutoff</dt><dd>{query.cutoff} s</dd></div>
            <div><dt>Verdict</dt><dd class="bad">{finding.verdict.title()}</dd></div>
            <div><dt>Batch average</dt><dd>{currency(batch_sum / batch_count / 100)}</dd></div>
            <div><dt>Replay average</dt><dd>{currency(replay_sum / replay_count / 100)}</dd></div>
          </dl>
          <div class="trail"><h3>Automatically tracked records</h3>
            <p class="cause"><a href="#fixed-row-{finding.witness}">Transaction {finding.witness}</a> was unavailable at the cutoff.</p>
            <ul>{records}</ul>
          </div>
        </details>
        {cost}
      </section>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/provenance.html"))
    parser.add_argument("--ci-jobs", type=Path, action="append", default=[],
                        help="Snapshot collected by ante_tempus_ml.ci_source. Repeat for distinct cohorts.")
    parser.add_argument("--window-bench", type=Path, help="Optional benchmark from ante_tempus_ml.window_bench.")
    args = parser.parse_args(argv)
    if args.out.suffix.lower() != ".html":
        parser.error("--out must be an .html file.")
    template = Template(files(__package__).joinpath("provenance_report.html").read_text(encoding="utf-8"))
    cases = scenarios()
    averages = [case for case in cases if case.prediction_dollars is None]
    scaling = [case for case in cases if case.prediction_dollars is not None]
    try:
        ci = ci_section([json.loads(path.read_text(encoding="utf-8")) for path in args.ci_jobs])
        window = window_html(json.loads(args.window_bench.read_text(encoding="utf-8")) if args.window_bench else None)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    content = {
        "averages": "".join(case_html(case, opened=i == 0) for i, case in enumerate(averages)),
        "scaling": "".join(case_html(case, opened=i == 0) for i, case in enumerate(scaling)),
        "validation": validation_html(),
        "target_leakage": target_html(),
        "churn": churn_html(),
        "ci_link": '<a href="#ci-duration">Real CI</a>' if args.ci_jobs else "",
        "ci_duration": ci,
        "fixed_window": window,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(template.substitute(content), encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
