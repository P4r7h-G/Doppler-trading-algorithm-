# Doppler vs GARCH event volatility research

This repository has two independent modules:

1. **Sports exchange markets** (`doppler_research.sports`, new): forecasts how much a sports market's implied probability will move (its realized variation) over fixed horizons after public team news, and scores those forecasts against what the market actually did. Built for trading-desk reviewers who bring their own licensed quote and news data.
2. **Equities** (`doppler_research`, original): daily earnings-event variance research. Its code is unchanged; its documentation is preserved [below](#equity-module-original-daily-study).

The design and its rationale are specified in [`docs/SPORTS_MARKET_ADAPTATION_SPEC.md`](docs/SPORTS_MARKET_ADAPTATION_SPEC.md).

## Sports module

### What it predicts, and what it does not

At a forecast origin `t0` after (or just before) a news item, for one selection of a pre-match football match-odds market, every model outputs a forecast of

```text
x_t                 = logit(p_t),   p_t = exchange quote midpoint (1/back + 1/lay) / 2
realized_variation  = sum over valid successive quotes in (t0, t0+h] of (x_t - x_(t-1))^2      <- primary target
net_move_proxy      = (x_(t0+h) - x_t0)^2
signed_reprice      = x_(t0+h) - x_t0                                                          <- reported, not forecast
```

for horizons `h` of 5, 30 and 120 minutes by default. In short: **"after this news, how volatile will this market be over the next h minutes?"** The forecast is a conditional expected value of a noisy positive quantity. It is **not** a match-result probability, not a direction, not an odds point forecast and not a trading-profit claim. Spectral features are unchanged when every price increment is sign-reversed, so they cannot identify direction. A single-selection forecast is marginal; it does not describe a coherent home/draw/away surface.

### Current evidence status

There is **no real sports result in this repository.** It has no licensed quote or news archive, and the only sports data it can generate is clearly labelled synthetic data with **planted** news effects. Scores on that synthetic data verify that the pipeline runs end to end; they say nothing about real markets. The accurate description is: *an open, inspectable implementation for testing whether pre/post-news spectral changes in exchange-implied probabilities improve forecasts of subsequent sports-market volatility, evaluated against event-aware and GARCH baselines on reviewer-supplied point-in-time data.*

### Quick start

Requires Python 3.10+.

```bash
python -m pip install -e ".[test,dashboard,parquet]"
python -m pytest -q
doppler-research sports make-synthetic --output examples/sports_synthetic
doppler-research sports validate --input examples/sports_synthetic
doppler-research sports run --config sports.example.json --input examples/sports_synthetic --output outputs/sports_study
doppler-research sports explain --run outputs/sports_study --list
doppler-research sports explain --run outputs/sports_study --event-key "SYN-N-019-0|SYNEX:1.200019|01901" --horizon 30m --model event_state_doppler
streamlit run src/doppler_research/sports/dashboard.py
```

On a laptop the synthetic study (48 fixtures, about 300,000 quote rows) runs in roughly 15 seconds. The dashboard can also generate a synthetic demo itself. Docker:

```bash
docker build -t doppler-garch .
docker run --rm -p 8501:8501 doppler-garch streamlit run src/doppler_research/sports/dashboard.py --server.address=0.0.0.0
```

Nothing needs API credentials, and no upload is sent anywhere: the CLI and dashboard read local files only. Run it inside your own environment for proprietary data.

### Reviewer workflow: test it on your own data

1. **Export** your quotes, markets, news and fixtures into the four canonical tables below (CSV or Parquet; blank templates in [`data/templates/sports`](data/templates/sports)). Put them in one directory or ZIP.
2. **Declare** the data's provider, licence owner and permitted use in `data_declaration` in a copy of [`sports.example.json`](sports.example.json). These are written into every output manifest.
3. **Freeze the experiment** in that config before looking at results: mode (`post_news` or `pre_release`), horizons, latency, eligible event types and confidence, quality thresholds, comparable-news hierarchy, ridge penalty, minimum training events.
4. **Validate**: `doppler-research sports validate --input DIR [--output DIR]`. Structural problems fail loudly; bad quote rows are removed and listed with a reason and their source row number.
5. **Hand-audit the replay**: `doppler-research sports replay --input DIR --market-id VENUE:MARKET --output DIR` writes the replayed book for every selection at the clock cadence, with each value's quote source row and the reason for any invalid point. Check several fixtures across news, suspensions and settlement before trusting a study.
6. **Run**: `doppler-research sports run --config cfg.json --input DIR --output OUT`.
7. **Compare forecast with reality** in `per_event_forecasts.csv` (one row per event x horizon x model, with forecast, realized value and losses), and in `common_sample_scores.csv`, `bootstrap_comparisons.csv` and `gate_checks.csv`.
8. **Audit any forecast**: `doppler-research sports explain --run OUT --event-key KEY --horizon 30m --model MODEL`, or the dashboard's *Audit trail* tab.
9. **Forecast a live situation** from data received so far: `doppler-research sports forecast --input DIR --market-id VENUE:MARKET --selection-id ID --origin 2026-11-16T14:00:00Z --horizon 30m --event-type injury --output OUT`. Training uses only earlier, fully observed events; no realized value is computed.

### Canonical input contracts

All timestamps must be ISO 8601 with `Z` or an explicit offset; timezone-naive values are rejected. Offsets are converted to UTC and the original text is kept (`<column>_raw`). Every row keeps its source row number and a row hash so outputs can point back to it. Join keys are provider-qualified (`venue:market_id`, `selection_id`); names are never keys.

| Table | Required columns | Optional columns |
| --- | --- | --- |
| `markets` (one row per selection) | `venue, sport, competition_id, fixture_id, market_id, market_type, selection_id, selection_name, home_team_id, away_team_id, scheduled_start_utc, market_created_utc, source` and `selection_role` (`home`/`draw`/`away`) for `MATCH_ODDS` | - |
| `quotes` (one row per selection update) | `venue, market_id, selection_id, provider_published_at_utc, received_at_utc, status, best_back_odds, best_back_size, best_lay_odds, best_lay_size, source` | `sequence, in_play, reconnect, traded_volume, last_traded_price` |
| `news_events` (one row per revision) | `news_id, event_type, subject_team_id, subject_player_id, source_id, source_url, source_published_at_utc, first_seen_at_utc, revision_id, confidence, raw_hash` | `fixture_id, release_scheduled_known_at_utc, severity, confirmation_status, human_review` |
| `fixtures_results` | `fixture_id, competition_id, home_team_id, away_team_id, scheduled_start_utc, fixture_status, source` | - |

`selection_role` is an addition to the spec's required list: without it the code would have to infer home/away from names. `status` is `OPEN`, `SUSPENDED` or `CLOSED`. `confidence` is `confirmed`, `reported` or `rumour`. `fixture_status` is `scheduled`, `played`, `postponed`, `abandoned`, `void` or `cancelled`. Publication and first-seen times may be blank for archival rows, but such events are excluded from the causal study.

**Quote rows rejected** (listed in `rejected_quotes.csv`): unknown market/selection, received before published (beyond `max_clock_skew_seconds`), odds `<= 1`, non-positive size, crossed book (back odds > lay odds), duplicate stream message (same market/selection/sequence, or identical content without a sequence), missing sequence when the column exists.

### Method

**Forecast origin** (one mode per run).
- `post_news`: `origin = max(source_published_at_utc, first_seen_at_utc) + post_news_latency_seconds`. The initial jump may already have happened and is not counted as forecastable.
- `pre_release`: `origin = source_published_at_utc - pre_release_lead_minutes`, only if `release_scheduled_known_at_utc <= origin` and the news was not seen before the origin.

Each `news_id` uses its earliest-received revision with eligible confidence; earlier revisions (for example a rumour) are listed in the audit trail, and nothing is overwritten. News links to a fixture explicitly through `fixture_id`, otherwise to the subject team's next fixture after the origin. The forecast selection is the subject team's runner by default.

**Causal replay.** The state at time `t` is the last OPEN quote *received* at or before `t`. A value is invalid, and never carried forward, when there is no quote yet, the quote is older than `max_quote_age_seconds`, a market break happened after it (suspension or closure, a `reconnect` row, a stream `sequence` gap, an `in_play` row), the market lacks a runner, or `t` is at or after the scheduled start. After any break every selection needs a fresh quote.

**Exclusions** (all logged in `excluded_events.csv`): no firm timestamp, first seen before published, release not scheduled at origin, ineligible confidence or type, postponed/void fixture, no market, missing runner, market not listed at origin, origin not pre-match, invalid quote at origin, spread above `max_spread_probability` or depth below `min_depth`, insufficient feature-window coverage, two news items on one fixture within `simultaneous_news_seconds`, other news for the fixture inside the target window, target window reaching kickoff, and target window interrupted by a suspension or gap (or bridged, if `target_suspension_policy` is `bridge`).

**Doppler features** (`spectral.py`), on one-minute logit increments ending at the origin. The recent 20 minutes are compared with each of three earlier, disjoint 20-minute windows. Equal durations mean the FFT frequencies coincide, so the comparison needs no interpolation. Missing steps inside a window count as zero change and are allowed only above `min_window_coverage`.
- `js`: Jensen-Shannon divergence between normalized recent and mean-baseline Hann-tapered spectra.
- `centroid_log_ratio`: log ratio of power-weighted mean frequencies.
- `log_energy_recent`, `log_energy_ratio`, `high_freq_share`, `spectral_entropy_recent`.
- `log_scale_factor` and `scale_fit_residual_js`: the frequency-scale factor `a` minimizing `JS(recent, normalize(P_base(f/a)/a))` over a fixed grid, and the distance left at the optimum.
- `*_velocity` and `*_acceleration`: first and second clock-time differences of `js` and the centroid ratio across snapshots 5 and 10 minutes earlier.

This is a frequency-scaling *analogy*, not a physical Doppler equation.

**Baselines and models**, all scored on the same events and horizon:

| Model | Forecast |
| --- | --- |
| `trailing_rv` | mean squared one-minute increment over the last 30 minutes x horizon steps |
| `ewma` | RiskMetrics EWMA one-step variance (lambda 0.94) x horizon steps |
| `garch_horizon` | GARCH(1,1) fitted at the origin on 360 minutes of increments; the **sum of conditional variances over the horizon** (`h_1 = omega + alpha e_T^2 + beta h_T`, `h_k = omega + (alpha+beta) h_(k-1)`, plus `mu^2` per step), not a one-step variance |
| `comparable_event_mean` / `_median` | earlier fully observed events, with a predeclared hierarchical fallback: market type + league + event type + liquidity band + kickoff band, then fewer keys |
| `event_adjusted_garch` | `garch_horizon` x the mean realized/GARCH ratio of those comparable events |
| `state_only` | ridge on trailing RV, spread, depth, time to kickoff, \|logit p\| |
| `doppler_only`, `state_doppler`, `event_state`, `event_state_doppler` | ridge on the named feature groups (event = log comparable mean) |
| `event_state_doppler_minus_{js,centroid,scale,energy}` | ablations removing one Doppler group |

Ridge models fit `log(target)` with a fixed penalty, using centring and scaling computed from the training rows only.

**Walk-forward.** Rows are processed in event-time order. A row can train a later forecast, or count as comparable history for it, only if its whole target window ended **before** that forecast's origin. There is no random split and no future normalization.

**Scoring.** For each horizon, every model is scored on the common intersection of events. Metrics are QLIKE (`log f + y/f` with a positive floor), MAE, RMSE and bias, in units of squared logit change summed over the horizon. Breakdowns are reported by event type, league, kickoff band, liquidity band and venue. Paired QLIKE differences get **fixture-clustered** bootstrap intervals (or calendar-week clusters), so many quotes around one event are not treated as independent tests. `gate_checks.csv` applies the spec's acceptance gate: a Doppler benefit counts only if `event_state_doppler` beats **both** `event_state` and `event_adjusted_garch` with an interval below zero. `signed_reprice_report.csv` summarizes signed moves against a zero-drift (no-change quote) baseline.

**Controls.** Each run also builds two placebo panels:
- `placebo_timestamps`: the same events with origins 240 minutes earlier.
- `shuffled_news_links`: each news item relinked, with a seeded RNG, to a different fixture that was trading at that time.

Skill that persists in these panels is not about the news; time to kickoff or market-state artefacts are likely explanations.

### Auditability: from input row to forecast to error

Every number in an output can be traced to the function that computed it and the input rows it used.

| Output | What it shows |
| --- | --- |
| `audit/traces.jsonl` | One record per event, holding the ordered steps that ran. Each step names its fully qualified function (`events.choose_revision`, `events.forecast_origin`, `events.link_fixture`, `replay.MarketReplay.grid`, `spectral.doppler_features`, `volatility.horizon_garch`, ...), with its inputs and outputs. Each horizon adds `targets.realized_targets` (with every quote source row used) and `pipeline.walk_forward`. Each model adds its forecast, realized value, QLIKE and error, plus the IDs of every training and comparable event. |
| `forecast_contributions.csv` | For every ridge forecast: intercept, and per feature the raw value, training mean and scale, standardized value, coefficient, and **contribution to the log forecast** (`forecast = exp(intercept + sum of contributions)`, exactly), with the function that computed the feature. Baselines list their parameters, such as GARCH `omega`/`alpha`/`beta`/`mu`, horizon steps and comparable-event count. |
| `feature_grid_lineage.csv` | Every pre-origin grid point: time, probability, logit, increment, validity reason, the **quote source row** that set it, and which window (recent or baseline 1-3) its increment fell in. |
| `target_lineage.csv` | Every post-origin target point, with its quote source row. |
| `event_panel.csv` | All features, GARCH fits and targets per event x horizon. |
| `feature_registry.json` | Function, formula and units of every feature. |
| `run_manifest.json`, `input_manifest.json` | Input SHA-256s, row counts, licence declaration, full config, bootstrap seed, package version, and a SHA-256 of every source file (`code.source_sha256`). |
| `excluded_events.csv`, `forecast_skips.csv`, `rejected_quotes.csv` | Every exclusion with its reason. |

Example (`sports explain`, synthetic data):

```text
Model event_state_doppler  <- doppler_research.sports.models.fit_ridge -> RidgeFit.explain
  intercept (mean log target of 23 training events with finite features) = -3.30688
  log_scale_factor             raw=     1.10904  z=    1.3781  coef=  -0.16477  contribution=-0.22707
  log_energy_recent            raw=    -4.92161  z=   0.66881  coef=  0.224389  contribution=+0.15007
  ...
  log forecast = -3.30626
  FORECAST realized_variation = 0.036653
  REALIZED = 0.0457288   QLIKE = -2.05865   error (forecast - realized) = -0.00907576
```

Runs are deterministic: the same inputs and config give identical forecasts, scores and bootstrap intervals (tested).

### Configuration

[`sports.example.json`](sports.example.json) lists every setting with its default; unknown keys are rejected. The most consequential settings:
- `mode`, `horizons_minutes`, `post_news_latency_seconds` / `pre_release_lead_minutes`.
- `probability_basis`: `raw_midpoint`, or `normalized_midpoint`, where the three midpoints are scaled to sum to one (kept as a separately labelled transformation).
- `target` (`realized_variation` or `net_move_proxy`) and `target_sampling` (`tick` per the spec, or `grid`). The other sampling is always exported as a diagnostic column. Plain GARCH, trailing RV and EWMA are built on one-minute grid increments, so under tick sampling they understate quote-level variation. The event-adjusted and ridge models learn that scale.
- `target_suspension_policy`: `exclude` or `bridge`.
- `max_quote_age_seconds`, `max_spread_probability`, `min_depth`, `min_window_coverage`.
- `comparable_fallback`, `min_comparable_events`, `min_train_events`, `ridge_alpha`.
- `bootstrap_cluster` (`fixture` or `calendar_week`), `seed`, `placebo_shift_minutes`, `shuffled_link_test`.

### Not yet implemented

- No live or historical **venue connectors**: no Betfair Exchange/Stream API reader or Betfair historical-file parser. Convert exports into the canonical tables yourself. The spec allows optional, read-only connectors; none are included.
- The `cross_venue_map` table; multi-venue comparisons need it.
- Bookmaker overround removal. Only exchange back/lay midpoints are supported.
- A coherent multi-runner (two log-ratio coordinates) forecast. v1 is marginal per selection.
- A directional model, calibrated volatility intervals, and odds or price scenarios. These are Phase 4 in the spec, gated on the volatility study.
- Opponent and league-rival event links. Only explicit fixture and subject-team next-fixture links are implemented.
- Normalization for time of day. Time to kickoff, liquidity and spread enter as standardized state controls instead.
- An untouched holdout run on real data. That needs reviewer data and a frozen config.

### Data licensing

The repository contains synthetic data only. Betfair, Smarkets, Sportmonks, Sportradar and newswire data each carry their own terms. A commercial evaluation, such as a trading firm's, generally needs a commercial data licence. The MIT licence of this software grants no rights to third-party data. Check each provider's terms before sharing populated inputs or derived quotes. This is read-only analytics: it places no bets.

## Equity module (original daily study)

Reproducible, point-in-time **conditional variance** research for scheduled equity events. The package computes transparent daily spectral features, fits a Gaussian GARCH(1,1), and compares both against state-only and event-aware baselines on the same out-of-sample events. It accepts user-owned CSVs through a CLI or local Streamlit dashboard.

**Current evidence status:** This repository does **not** reproduce the earlier AAPL + 11-company results. Twelve user-uploaded NASDAQ-format OHLCV files and an older Apple package were recovered. Their ticker identities were matched against published 9 September 2026 OHLC values, but original ticker metadata and vendor adjustment documentation were not preserved. The Apple package contains 43 earnings rows sourced to post-event Newsroom pages; it does not establish that dates and sessions were known before each forecast. Other-company dated events and prior event-level forecasts are missing. Local dataset ZIPs with the available rows are under ignored `private_data/dataset_packages`; the three ZIPs in [`data/packages`](data/packages) are **public, zero-row schema placeholders**. The supplied historical report and aggregate result tables are archived under [`research/prior_unverified`](research/prior_unverified) for traceability, not endorsed as reproduced results. Do not cite those figures as outcomes from this code.

### Quick start

Requires Python 3.10+.

On Windows PowerShell, create and activate the environment with `py -m venv .venv` and `.\.venv\Scripts\Activate.ps1`. On macOS/Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`. Then run the commands below in that environment.

```bash
python -m pip install -e ".[test,dashboard]"
python -m pytest -q
python scripts/make_synthetic_demo.py
doppler-research run --prices examples/synthetic/prices.csv --events examples/synthetic/events.csv --output outputs/synthetic
doppler-research forecast-next --prices examples/synthetic/prices.csv --events examples/synthetic/events_with_next.csv --event-id SYNTH-NEXT --output outputs/synthetic_next
streamlit run src/doppler_research/dashboard.py
```

The synthetic run checks installation only. Its scores do not establish market performance. The dashboard runs locally; uploaded files remain in that process. A hosted dashboard operator would have access to uploads, so use local deployment for proprietary data.

Docker dashboard:

```bash
docker build -t doppler-garch .
docker run --rm -p 8501:8501 doppler-garch
```

Visit `http://localhost:8501`. The Docker image contains code and blank templates, no private prices.

### Input contracts

Use UTF-8 CSV with ISO `YYYY-MM-DD` dates and one row per ticker/trading day or event. See [`data/templates`](data/templates). Price fields: `ticker,date,open,high,low,close,volume,adjustment,source`. OHLC must all be positive and on the **same split-adjusted basis**; `adjustment` must say `split_adjusted`. That label is a data-owner attestation, not something this code can verify. Prices may be dividend-adjusted if all OHLC fields share the same basis, but record that detail in `source` and do not mix bases. Duplicate dates, impossible ranges, missing values, unknown adjustment and negative volume fail validation.

Event fields: `event_id,ticker,announcement_date,session,event_type,source_url,verified_at,confidence`. `session` is `after_close`, `before_open`, `during_session`, or `unknown`; only the first two qualify. `confidence` is `verified`, `reported`, or `unverified`; only `verified` qualifies. `verified_at` is the date that an advance notice established the scheduled date and session; it **must precede** `announcement_date`. A later press release can verify what happened historically but does not prove that the timing was known at the forecast origin. `source_url` must identify the advance notice. Keep unexpected product releases in the dataset as `unverified` or `unknown` and expect them to be excluded from this prospective test.

For an after-close event, the forecast origin is that day's regular-session close and the reaction is the next trading session. For a before-open event, the origin is the prior close and the reaction is the event day's session. Events on non-trading dates are excluded and reported. No nearest-date guess is made. The event date must be present in the price table.

Convert a user-owned NASDAQ-format download with an **explicit** ticker and adjustment attestation:

```bash
doppler-research convert-nasdaq --input my_download.csv --ticker AAPL --source "NASDAQ download; retrieved YYYY-MM-DD; adjustment reviewed by owner" --adjustment split_adjusted --output private_data/aapl_prices.csv
doppler-research validate --prices private_data/aapl_prices.csv --events private_data/aapl_events.csv
doppler-research run --prices private_data/aapl_prices.csv --events private_data/aapl_events.csv --config config.example.json --output outputs/aapl
```

For a multi-company run, concatenate canonical price rows and event rows into one CSV each. The program never infers ticker identity while processing inputs. The local recovered files are under ignored `private_data/recovered_uploads`; their checksums and external OHLC matches are listed in [`data/recovered_upload_manifest.json`](data/recovered_upload_manifest.json) and [`data/price_file_matches.csv`](data/price_file_matches.csv). A match identifies a likely ticker; it is not the original download metadata. Price adjustment remains `unknown` in the local archival ZIPs and is rejected by the modelling validator until reviewed.

Once the inputs are identified and validated, create three separate data ZIPs:

```bash
doppler-research package-data --prices private_data/all_prices.csv --events private_data/all_events.csv --output private_data/dataset_packages
```

The command makes `01_aapl_events.zip`, `02_aapl_prices.zip`, and `03_other_companies.zip`, each with CSVs and a manifest containing row counts and SHA-256 checksums. It requires both AAPL and other-company verified event rows. Product announcements belong in the event CSV with their actual type and pre-event timing evidence. Missing events remain missing, not filled from a model. The recovered local ZIPs have different names and remain archival because their timing/adjustment checks are incomplete. **Review your market-data and event-source redistribution rights before sharing populated ZIPs.** The software's MIT license does not grant rights to third-party data. [Nasdaq's site terms](https://www.nasdaq.com/legal) impose restrictions on site content use and redistribution; use a license that covers your intended research and recipients.

### Model and feature specification

All model forecasts target a **one-reaction-session conditional variance proxy**, never a deterministic price. Default target is squared close-to-close log return, `y = log(C_reaction/C_origin)^2`. Optional `gap_parkinson` is `log(O_reaction/C_origin)^2 + log(H_reaction/L_reaction)^2/(4 log 2)`. This range/gap proxy is an alternate noisy target, not the same object as a close-to-close squared return. The one-step GARCH forecast is compared to it as a diagnostic; its event-adjusted version is a fairer event-aware comparator. Do not compare raw QLIKE levels across different targets.

The Doppler label is an **analogy**, not a claim of physical Doppler physics or a demonstrated trading edge. For each forecast origin, use daily log returns ending at that origin:

- 60 older returns, indices `[-80:-20]`, form the baseline spectrum.
- 20 most recent returns, indices `[-20:]`, form the recent spectrum.
- Each window is demeaned, multiplied by a Hann taper, transformed with a real FFT, and its nonzero-frequency power interpolated onto a fixed 32-point grid from `1/60` to `0.5` cycles/session.
- Normalize each power vector to sum to one. With spectra `P` and `Q` and `M=(P+Q)/2`, Jensen–Shannon deformation is `JS = 0.5 KL(P||M) + 0.5 KL(Q||M)` with natural logs. It measures **magnitude**, not direction.
- The signed frequency shift is `z_f = log(centroid_recent / centroid_baseline)` where centroid is the power-weighted frequency. Positive values shift toward higher frequency.
- `js_velocity` and `centroid_velocity` are one-session first differences; `js_acceleration` and `centroid_acceleration` are second differences, computed from snapshots ending at origin, origin minus one, and origin minus two. These use 82 returns total.
- `log_energy_recent`, `log_energy_ratio` and `high_energy_share` retain power-scale and high-band information. Zero-energy windows use a uniform spectrum with a small numerical floor for log energy.

The state controls are trailing mean squared return over 20 sessions (`rv20`) and the current log-volume z-score in a 60-session window. The event-aware control is the log mean of **earlier observed reactions for the same ticker and event type**. `historical_event_mean` and `historical_event_median` predict with that same strictly prior sample.

GARCH is refitted from price returns available at every forecast origin. In percent-return units, `epsilon_t = 100*r_t - mu` and `h_t = omega + alpha*epsilon_(t-1)^2 + beta*h_(t-1)`. Gaussian quasi maximum likelihood minimizes `0.5 Σ[log(2πh_t)+epsilon_t²/h_t]`, subject to `omega>0`, `alpha,beta>=0`, `alpha+beta<0.999`; `h_0` is the sample residual variance. The next variance forecast is `(omega + alpha*epsilon_T² + beta*h_T)/10000`. Per-event `omega`, `alpha`, `beta`, sample mean, fit size and negative log likelihood are exported in `event_features.csv`. A failed fit excludes that event with a reason. This is a transparent baseline, not a claim that Gaussian innovations fit every asset.

The models are `garch`, `historical_event_mean`, `historical_event_median`, `event_adjusted_garch` (GARCH times the earlier same-ticker/event-type mean target-to-GARCH ratio), `state_only`, `doppler_only`, `doppler_state`, `event_state`, and `event_state_doppler`. Ridge models predict log variance using a fixed penalty from the configuration and ticker fixed effects (one reference ticker absorbed by the intercept). Feature centering and scaling, ticker categories, coefficients and event baselines are recomputed only from earlier **fully observed** events; no random split or future normalization is used. This is **not an exact reimplementation** of the archived study: its executable scripts and event-level inputs are missing, and its described training schedule may differ. Version 0.1.0 is a newly specified, inspectable benchmark; its outputs must be labeled as such.

### Scores and diagnostics

`run` writes `event_features.csv`, `skipped_events.csv`, `per_event_forecasts.csv`, `common_sample_scores.csv`, `bootstrap_comparisons.csv`, and `run_manifest.json` with input checksums/configuration. A row can train a later forecast only when its reaction closed **before** that forecast's as-of date. All headline models are scored on the same event-ID intersection. Scores are pooled and per ticker. QLIKE is `log(predicted_variance) + observed_proxy/predicted_variance` (lower is better); MAE and RMSE are in variance units. Each event also gets its own QLIKE and errors. Variance forecasts have a positive numerical floor. Never interpret a negative QLIKE as a negative variance.

Paired diagnostics compare Doppler-only vs GARCH, Doppler+state vs state, and event+state+Doppler vs event+state and historical event mean. Calendar quarters are resampled with replacement and the paired mean QLIKE difference is recomputed; the 95% percentile interval is exploratory, especially for small samples or many model/target comparisons. The random seed and replicate count are recorded. No p-value or trading profitability claim is made.

Optional historical price intervals require an explicit drift assumption:

```bash
doppler-research run --prices private_data/all_prices.csv --events private_data/all_events.csv --output outputs/study --drift-log-return 0.0
```

`0.0` is an **assumption of zero log-return drift**, not a learned directional forecast. For each model/event, the interval uses the 2.5% and 97.5% empirical quantiles of standardized returns from **at least 40 earlier out-of-sample events**, scaled by its variance forecast and applied to the as-of close. Rows without that calibration history are omitted. Coverage is not guaranteed; assess it on a separate untouched sample. These intervals are scenarios conditioned on drift and variance assumptions, not point price predictions.

For a newly scheduled **after-close** event at the latest available regular-session close, add its verified calendar row to the events CSV and run `forecast-next`. It produces `next_event_variance_forecasts.csv` without using an unavailable reaction price. Supply `--drift-log-return` to request `next_event_price_intervals.csv`; it remains empty unless at least 40 prior out-of-sample events are available for calibration. No point price is emitted. The synthetic example above includes a `SYNTH-NEXT` row solely to exercise this command.

### Research limitations and reproducibility

- Earlier aggregate scores cannot be reconstructed from a report and summary tables alone. The event-level predictions, full event-source map and executable prior multi-company scripts are needed. The archived report itself says the immediate Doppler gain over state/event baselines was weak or absent; no unique frequency edge should be marketed from it.
- Market-data rights, split/dividend adjustment, ticker changes, missing sessions, schedule revisions and verified announcement timing must be resolved for each input. A hash establishes file identity, not correctness or a redistribution license. The recovered WMT file has one inconsistent OHLC row on 5 June 2023, flagged in `DATA_STATUS.md` and the local package manifest; the validator rejects it until reviewed.
- Daily 20/60-session spectra are coarse. The stronger original hypothesis concerns intraday changes as time to a known catalyst approaches zero. Do not tune this daily specification repeatedly on the prior panel and then call the result confirmatory.
- Squared returns and range proxies are noisy measurements of latent variance. GARCH does not know a scheduled event is imminent; the historical event and event-adjusted baselines are therefore essential. A model can improve QLIKE yet worsen MAE/RMSE.
- Forecasts depend on event selection, adjusted price basis, data access and fixed hyperparameters. A real validation needs untouched future events or companies, predefined targets and thresholds, and an independent data audit.
