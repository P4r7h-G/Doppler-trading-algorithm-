# Sports market Doppler research: implementation specification

**Status:** design proposal, 16 November 2026. The repository's current code accepts daily equity OHLCV and dated corporate events. It does **not** yet ingest sports odds, news feeds, or intraday exchange data, and it has no verified sports forecasting results. This document is intended for review by quantitative sports traders, including teams that will supply their own licensed data.

## 1. Research question and pilot scope

**Question:** Does a change in the frequency structure of a sports market's implied-probability path improve forecasts of *subsequent market repricing or volatility* around public team news, after controlling for existing market state and comparable past news?

Start with one narrow, reproducible panel: liquid **pre-match football match-odds (home/draw/away)** markets and independently time-stamped confirmed lineup or player-availability announcements. Record which selection is being forecast and retain all three outcomes for coherent probability normalization. Separate markets for outright season winners, signings, club investment, transfer rumours, and in-play injuries; they have different horizons, liquidity, and event mechanisms. Do not pool them in the first result.

The first useful forecast is a probability distribution for the change in a market-implied probability over fixed horizons such as 5, 30, and 120 minutes after the forecast origin. A forecast of the *magnitude* of repricing is the primary target. A direction or point-odds forecast is a separate, harder experiment with separate signed features and validation. A market-implied probability is a price-like quote, **not** the actual chance that a team will win.

### Two distinct forecast origins

1. **Before a known release** (for example, an announced lineup publication time): use only quotes and facts available before the release. The event type may be used only if it was known then. This tests whether pre-release market dynamics forecast the size of the coming repricing. It cannot use the content of a lineup not yet public.
2. **After unexpected news becomes public** (for example, a newly reported injury or signing): use the first verifiable receipt of the news as the earliest possible origin, plus a recorded ingestion/processing latency. This tests the remaining move from that origin onward. The initial market jump may already have happened and must not be counted as forecastable from post-publication features.

The user must choose one mode per run. A historical article date or injury start date alone does not establish when a trader could first have known the news.

## 2. UK-compatible data acquisition

The research program must work entirely from user-owned uploads. Live connectors are optional and must be read-only in this phase. Do not route through Kalshi or Polymarket, or use location workarounds; the user has said those venues are unavailable to them.

### Primary exchange option: Betfair Exchange

- **Market discovery and metadata:** Exchange Betting API `listEventTypes`, `listEvents`, `listMarketCatalogue`, and `listMarketBook`; use provider market IDs and runner/selection IDs as the stable joins. Betfair's [getting-started guide](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687786/Getting+Started) explains the application key and session-token flow.
- **Live quote history going forward:** [Exchange Stream API](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687396/Exchange+Stream+API), subscribing to market/runner updates and persisting the initial image plus every delta, publish time, receipt time, sequence/cursor, reconnect, status, available-to-back/lay levels and sizes, and traded volume. A delayed development key conflates changes into approximately **three-minute** messages and is unsuitable for seconds-level inference; low-latency research needs the appropriate live access and permission. See [Betfair's access note](https://support.developer.betfair.com/hc/en-us/articles/115003887871-How-do-I-get-access-to-the-Stream-API).
- **Historical backfill:** Betfair's [Historical Data service](https://support.developer.betfair.com/hc/en-us/articles/360002407732-What-data-is-provided-by-the-Historical-Data-service) and [Historical Data API](https://support.developer.betfair.com/hc/en-us/articles/12859956891932-How-Can-I-Make-HTTP-Requests-to-the-Historical-Data-API) provide purchased exchange market/price/settlement archives. Choose the package by its documented fields and sampling before purchase; the mere availability of historical data does not prove it resolves every news announcement to seconds. Downloaded archives remain outside Git.
- **Access and redistribution:** Betfair says API access requires a verified account and that commercial use needs the relevant approval or data licence. A SIG evaluation is a commercial context, so neither a personal key nor this repository's software licence should be treated as permission to redistribute Betfair data. See [Betfair API licensing](https://support.developer.betfair.com/hc/en-us/articles/360002464152-Which-API-Licence-Do-I-Require) and [application-key conditions](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687105).

### Optional data and event sources

- **SIG's own feed:** preferred for independent evaluation. Accept its licensed quote, event, fixture, and outcome exports through the same canonical schemas; do not require a Betfair account or expose its data to a hosted service.
- **Smarkets:** an alternative venue *only after explicit provider approval for this research use*. Its [API policy](https://help.smarkets.com/hc/en-gb/articles/34697834941085-Smarkets-API-Access-Integration-T-Cs) describes an approval process and fee, and restricts data harvesting, third-party sharing, and benchmarking. It is therefore **not** a default substitute for a commercial comparison sent to SIG.
- **Sportmonks:** possible fixture/team/player/transfer/sidelined and bookmaker-odds metadata. Its [premium odds history](https://docs.sportmonks.com/v3/endpoints-and-entities/endpoints/premium-odds-feed/premium-pre-match-odds/get-all-historical-odds) records bookmaker updates but says history is retrievable only until seven days after fixture start: collect forward if licensed; do not assume a multi-year backfill. Its [sidelined and lineup fields](https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/lineups-and-formations) can describe news, but the injury *start date* is not the first-publication time. Its [prediction API](https://docs.sportmonks.com/v3/tutorials-and-guides/tutorials/odds-and-predictions/predictions) updates daily, so it is a possible slow context baseline, not a minute-level target series.
- **Sportradar:** possible licensed match/lineup feed. Its [lineup API documentation](https://docs.sportradar.com/soccer/rest-api/lineups) states the endpoint can be polled at 30-second intervals. Verify league coverage, rights, whether messages include the first-publication timestamp, and the distinction between predicted and confirmed lineups before use.
- **Primary club/league announcements or a licensed newswire:** retain immutable article IDs/URLs, initial publication timestamp, first observed timestamp, revisions, source and raw payload hash. A later-edited webpage is not a historical point-in-time feed. Obtain an archive or capture the feed prospectively.
- **Fixture/results source:** retain official fixture ID, scheduled start, final status, postponement/void rules and settlement result. Use match outcomes only for a separate win-probability calibration study; they are not the realized market-volatility target.

No connector should scrape a venue in breach of its terms. Store keys in environment variables or a local secret store, never in CSV, dashboard exports, logs or Git. Before sharing any populated data or derived quotes with SIG, check the relevant provider's licence. The [UK Gambling Commission register](https://www.gamblingcommission.gov.uk/public-register/businesses) is the authoritative place to check an operator's current status; this proposal is read-only analytics, not a betting service or automated wager placement.

## 3. Canonical upload contracts (UTF-8 CSV or Parquet)

All timestamps are timezone-aware UTC instants. Retain the source's original timezone and raw timestamp text. A `source_published_at` is distinct from `received_at`; a backtest uses the later of all required availability times. Unknown timing remains null and excludes the row from time-sensitive evaluation. Every imported file has source, licence owner, capture method, original filename, SHA-256, schema version and import timestamp in a manifest.

### `markets`

Required: `venue`, `sport`, `competition_id`, `fixture_id`, `market_id`, `market_type`, `selection_id`, `selection_name`, `home_team_id`, `away_team_id`, `scheduled_start_utc`, `market_created_utc`, `source`. Retain market rules, whether draw is an outcome, settlement status, suspension periods, postponements and runner remappings. IDs must be provider-qualified; names alone are not keys.

### `quotes`

Required: `venue`, `market_id`, `selection_id`, `provider_published_at_utc`, `received_at_utc`, `status`, `best_back_odds`, `best_back_size`, `best_lay_odds`, `best_lay_size`, `source`. Recommended: depth levels, last traded price, traded volume, stream sequence and latency. Preserve each raw update; derive snapshots deterministically by replay. Reject nonpositive size, decimal odds `<=1`, impossible crossed/stale books, duplicate stream messages, missing outcome sets and quotes while suspended. Do **not** fill through a suspension, kickoff, market closure or long data gap.

### `news_events`

Required: `news_id`, `event_type`, `subject_team_id`, `subject_player_id` (nullable), `source_id`, `source_url`, `source_published_at_utc`, `first_seen_at_utc`, `revision_id`, `confidence`, `raw_hash`. Recommended: confirmation status, affected team IDs, fixture IDs, severity, injury status, lineup expected/confirmed, entity-resolution evidence and human-review flag. Publication/receipt may be null in an archival upload, but such an event must not enter a millisecond/minute causal backtest. Rumour, correction and confirmation are separate revisions; do not overwrite the initial item with later information.

### `fixtures_results` and `cross_venue_map`

Fixtures/results join provider IDs to teams, league, kickoff, fixture status and settlement. A separate versioned cross-venue map supports licensed comparisons; mappings need valid-from/valid-to times and review provenance. Never join on fuzzy names alone. The event-to-market link must be explicit or auditable: direct team, opponent, next fixture, or league rival, with the graph known as of the forecast origin.

## 4. Probability and target definitions

For a liquid exchange selection, convert both sides of the book to implied probabilities, `p_back=1/best_back_odds` and `p_lay=1/best_lay_odds`. The provisional **quote midpoint** is `(p_back+p_lay)/2`; it is not an executable price and should be accompanied by spread and size. For a full home/draw/away market, keep raw midpoint probabilities and, as a separately labelled transformation, normalize the three positive midpoints to sum to one. Bookmaker odds require an explicit overround-removal method, reported separately from exchange quotes. Use a fixed, documented clipping epsilon only for `logit(p)=log(p/(1-p))` numerics. No outcome may disappear silently.

At origin `t0`, freeze the last usable, non-stale quote already received. For a predeclared horizon `h`, define `x_t=logit(p_t)` and:

```text
signed_reprice_h = x_(t0+h) - x_t0
absolute_reprice_h = |signed_reprice_h|
net_move_proxy_h = signed_reprice_h^2
realized_variation_h = sum over valid successive quotes [x_t - x_(t-1)]^2
```

`realized_variation_h` is the **primary volatility target**: it measures the market-implied probability path after the origin, including jumps. It is not the variance of the actual match result. The forecast is a conditional expected value of this positive, noisy target. Evaluate `net_move_proxy_h` separately because a large jump followed by reversal can produce high path variation but little net repricing. `signed_reprice_h` is a distinct directional target; `p_(t0+h)` is a quote-level target. Choose deterministic endpoint tolerance, no-quote rules, suspension treatment and minimum coverage *before* scoring. Never mix pre-match and in-play horizons in one headline score.

For multi-runner outputs, a later vector model should forecast two log-ratio coordinates and map simulated draws through softmax so home/draw/away probabilities remain nonnegative and sum to one. A single-runner v1 is explicitly **marginal** and should not claim a coherent three-outcome price surface.

## 5. Doppler adaptation and model controls

Keep `src/doppler_research/spectral.py` as a transparent reference, not as a drop-in implementation. Its 60-day/20-day FFT and daily first/second differences are in units of trading sessions. Rewrite the feature engine for market-implied probability increments at a chosen clock cadence. At one-minute resolution, compare equal-duration recent and baseline windows (for example, recent 20 minutes against each of three earlier disjoint 20-minute windows); report effective sample size and gap handling. Analyse five-second or tick data only after quote-quality checks. Normalize for time to kickoff, time of day, market liquidity and scheduled-release proximity using training data alone.

Candidate Doppler features: Jensen–Shannon divergence between normalized spectra; signed log centroid ratio; band energy, entropy and high-frequency share; first and second **clock-time** differences; and a fitted spectral frequency-scale factor `a` minimizing a predeclared distance between recent spectrum and the scaled baseline `P_base(f/a)/a`. Export the fitted shift, residual distance and fit quality separately. This is a **frequency-scaling analogy**, not a physical Doppler equation: market probabilities have no wave-propagation speed or source/observer velocity. Add a planted-frequency-shift synthetic test; interpolation must not be mistaken for extra frequency resolution. Spectral power is unchanged if every signed probability increment is reversed, so these features alone cannot identify the *direction* of repricing.

Preserve inspectable comparators on the **same events and horizon**:

1. No-change quote and trailing realized-variation mean/EWMA.
2. Historical mean/median for earlier comparable news by market type, league, event type, liquidity band and proximity to kickoff, with hierarchical fallback stated in advance.
3. GARCH(1,1) on one-minute `logit(p)` increments; forecast the **sum of its conditional variances over the target horizon**, with fit window, parameters, failure reason and regime assumptions exported. One-step GARCH must not be compared against 30-minute realized variation as if horizons match.
4. Event-adjusted GARCH and a state-only model using spread, depth, volume, recent variation, time to kickoff and news timing. These are necessary controls because a plain GARCH model does not know the news context.
5. Doppler-only, state+Doppler, and event+state+Doppler, with fixed regularization and strictly past training. Ablate Jensen–Shannon, centroid shift, fitted scale, and energy features separately.

A **directional** extension may use signed order flow, the factual content and severity of public news, lineup/player value, cross-team exposure and current odds. Its output must be calibrated as a return distribution or probability of an upward/downward repricing. Do not label a symmetric volatility forecast as a predicted future odds quote. Convert a modeled quote distribution back to decimal odds with `1/p` only after calibration and report bid/ask and liquidity context; it is not a guaranteed executable price.

## 6. Causal replay and evaluation

Use immutable raw streams and deterministic replay. A feature is eligible only when every source message was **received** no later than the origin. In pre-news mode, the origin must precede the public event; in post-news mode it must follow first receipt plus an explicit processing delay. News revision and entity mapping must be versioned. Stop features at origin; score only later quotes. Log all exclusions: no firm event timestamp, thin/crossed book, missing runner, suspension, simultaneous news, market closed, bad stream sequence, ambiguous fixture mapping and overlapping evaluation windows.

Walk forward by *event time*, training only on fully observed earlier horizons. Freeze feature definitions and hyperparameters before an untouched season or league holdout. Build separate event-level and pooled reports; disclose counts by event type, market type, league, time-to-kickoff band, quote coverage and venue. Do not treat thousands of quote updates around one event as thousands of independent tests. Cluster uncertainty by fixture/event and calendar period; repeated horizons are correlated. Use paired resampling of the **common event intersection**. Keep an untouched provider or league as a transportability test if data volume permits.

Primary error metrics for the positive `realized_variation_h` target: QLIKE, MAE, RMSE and bias, all with exact target units and common sample. QLIKE is `log(v_hat) + y/v_hat` with a fixed positive forecast floor. Compare volatility-interval coverage and sharpness only if intervals are calibrated on earlier events. For signed repricing, report MAE, direction accuracy and calibration against a zero-drift baseline; for an actual match-result probability model, use Brier/log loss **separately**. Add placebo timestamps, shuffled news links, and a pre-news versus post-news split to test whether the apparent signal is merely time-to-kickoff or a market-status artefact. No claim of tradable profit follows from a good volatility score without executable-quote, spread, commission, latency and limit analysis.

## 7. Software deliverables and SIG reviewer flow

Keep the existing equity pipeline untouched. Add a `sports` namespace with venue adapters, schema validation, replay, probability conversion, event linking, intraday spectral features, horizon-matched GARCH, baselines, scoring and export. The CLI and local dashboard must both run directly from canonical user-owned files with no API credentials. Optional collection commands remain read-only.

Suggested commands (interface to implement, **not currently available**):

```text
doppler-research sports validate --markets markets.csv --quotes quotes.parquet --news news_events.csv --fixtures fixtures_results.csv
doppler-research sports replay --config sports.example.json --input private_data/sports --output outputs/sports_replay
doppler-research sports run --config sports.example.json --input private_data/sports --output outputs/sports_study
doppler-research sports forecast --market-id VENUE:MARKET --origin 2026-11-16T14:00:00Z --horizon 30m --input private_data/sports
```

The dashboard should let SIG upload the four canonical files or a ZIP with manifest, choose pre/post-news mode and horizon, inspect exclusions and point-in-time evidence, display market path with event/origin/target markers, compare per-event predictions and common-sample scores, inspect every Doppler feature and GARCH parameter, and export reproducible CSV/JSON. It should run locally or in SIG's environment; the public repository should contain **synthetic fixtures only**. Results exports include input hashes, code version, config, provider/licence declaration, timestamps, forecast origin, target window, model fit counts, forecast, realized target, errors, bootstrap seed and excluded-event reasons. No upload should be sent to an external analytics endpoint.

## 8. Implementation checklist and acceptance gates

### Phase 0 — secure data and freeze the experiment

- [ ] Choose the first league, market type, eligible news type, pre/post mode, time resolution and 5/30/120-minute horizons. Record target and holdout plan before inspecting results.
- [ ] Obtain a permitted quote archive, event archive with publication **and first-seen** times, fixture mapping, and usage/redistribution permissions. Determine whether the chosen Betfair historical tier contains the required quote depth and update cadence.
- [ ] Define quality thresholds for spread, depth, staleness, stream gaps, suspensions and overlapping news; produce a data inventory and event eligibility count. **Gate:** no performance claims until a sufficient point-in-time panel exists.

### Phase 1 — canonical data and replay

- [ ] Implement the four schemas, provider-qualified IDs, timezone/DST handling, raw-payload hashes, versioned mappings and failure reports.
- [ ] Implement historical Betfair stream reconstruction and a read-only live collector behind optional credentials. Add user-owned CSV/Parquet adapter first.
- [ ] Implement quote midpoint/probability conversion, all-outcome normalization, status handling and target construction. **Gate:** hand-audit several real fixtures across news, suspensions and settlement.

### Phase 2 — transparent models

- [ ] Implement causal multi-resolution Doppler features and planted-shift/sign-reversal tests.
- [ ] Implement horizon-matched GARCH and event-aware, state-only, no-change and EWMA controls.
- [ ] Implement expanding event-time replay, common-sample scoring, fixture-cluster bootstrap and placebo tests. **Gate:** demonstrate zero future-data access in automated leakage tests.

### Phase 3 — private reviewer package

- [ ] Add local dashboard, CLI, example configuration, input templates, synthetic example, Docker path, install instructions and documented data licence boundaries.
- [ ] Add tests for stale/crossed quotes, missing runners, feed reconnects, exact news timing, revisions, simultaneous events, match postponements, in-play transitions, timezone changes and overlapping windows.
- [ ] Run on an untouched season/league/provider; publish event counts, exclusions, all baseline comparisons and uncertainty. **Gate:** any claimed Doppler benefit must beat event+state and event-adjusted GARCH on a held-out common sample, not merely plain GARCH.

### Phase 4 — optional directional research

- [ ] Only after the volatility study, specify signed information features and a separate directional target/holdout.
- [ ] Calibrate a future probability/odds distribution, test against no-change and market-based baselines, and measure executable-quote economics separately. **Gate:** no point-odds forecast in the dashboard unless its drift/direction assumptions and out-of-sample calibration are explicit.

## 9. Current blockers and claims allowed today

The current repository has no sports quote/news archive, no provider credentials or commercial data permission, and no sports backtest. It cannot yet compare a sports Doppler forecast with SIG's realized market path. A real-time API alone is insufficient to reconstruct historical first-seen news timestamps. The development key's conflated Betfair stream cannot support seconds-level testing. Sportmonks' injury dates and lineup fields are useful context but are not, by themselves, point-in-time publication evidence. The present daily equity spectra and scores cannot be carried over to a sports market.

The accurate external description is: **“An open, inspectable proposal to test whether pre/post-news spectral changes in exchange-implied probabilities improve forecasts of subsequent sports-market repricing and volatility, evaluated against event-aware and GARCH baselines on reviewer-supplied point-in-time data.”**
