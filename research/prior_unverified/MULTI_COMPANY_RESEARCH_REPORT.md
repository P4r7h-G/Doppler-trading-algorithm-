# Multi-Company Doppler vs GARCH Backtest

## Executive conclusion

The multi-company test does **not** validate a unique Doppler-frequency forecasting edge yet.

It does show very clearly that the current Doppler-based models outperform a plain GARCH(1,1) forecast around scheduled earnings releases under QLIKE. However, this is not enough to attribute the gain to Doppler-frequency deformation: a simple point-in-time historical earnings-variance forecast is generally stronger, and adding the Doppler variables to a conventional state model does not materially improve immediate post-earnings forecasts.

The strongest interpretation is therefore:

1. Plain GARCH is badly misspecified for known earnings catalysts because it does not explicitly model the scheduled variance jump.
2. Event awareness explains a large portion of the apparent Doppler-vs-GARCH advantage.
3. The present daily-frequency Doppler features have **not yet demonstrated incremental information** beyond ordinary state variables / historical earnings variance.
4. This remains worth testing with intraday data, where the intended Doppler concept -- frequency deformation as time-to-catalyst approaches zero -- can be measured much more directly.

## Companies

The full panel contains AAPL plus the 11 newly supplied NASDAQ price histories:

AAPL, AMAT, AVGO, COST, CSCO, LIN, LRCX, META, MU, NVDA, TSLA, WMT.

The main robustness result excludes AAPL so that the previous Apple-only result cannot drive the conclusion. LIN has a shorter price history (starting in 2023), so it contributes fewer observations where a 500-observation GARCH warm-up is required.

## Price data

The supplied NASDAQ files contain daily Open, High, Low, Close/Last and Volume. Most histories run from September 2016 through 9 September 2026; LIN begins in March 2023. The series were normalized and checked for missing/duplicate observations and obvious split discontinuities.

## Earnings alignment

For the timing-aware test:

- after-close announcements: predictors include the earnings-date regular-session close; target begins with the next trading session;
- before-open announcements: predictors stop at the previous trading-session close; target begins with the earnings-date session.

WMT and LIN are before-open reporters in the tested recent history. The other panel names are generally after-close reporters. One historical Broadcom exception (26 May 2022, before open) was explicitly corrected in the training data.

## Doppler features

The model uses a non-overlapping 60-day baseline spectrum and 20-day recent spectrum.

Core variables include:

- Jensen-Shannon spectral deformation D;
- first and second differences V_D and A_D;
- signed log spectral-centroid shift (financial redshift/blueshift);
- signed frequency velocity and acceleration;
- high-/low-band power shifts;
- spectral entropy and high-frequency power.

The state-only benchmark uses recent 20-day realized volatility and a 60-day volume z-score.

All regression models use point-in-time scaling and ticker fixed effects with Ridge regularization. No random train/test split is used.

## Development-period test: 2023-2025

Training begins with 2020-2022 and expands annually. The clean main robustness panel excludes AAPL.

### Immediate squared-return variance target (new 11, 123 common OOS events)

| Model | QLIKE (lower is better) |
|---|---:|
| Historical earnings mean | -4.057 |
| Historical earnings median | -3.280 |
| Earnings-calibrated GARCH | -2.866 |
| State-only Ridge | -1.880 |
| Doppler + GARCH Ridge | -1.822 |
| Doppler System Ridge | -1.764 |
| Doppler Core Ridge | -1.402 |
| Plain GARCH(1,1) | 4.144 |

Doppler Core minus GARCH mean QLIKE difference: **-5.546**, calendar-quarter cluster-bootstrap 95% interval **[-7.388, -3.826]**.

But Doppler System minus State-only: **+0.108**, 95% interval **[-0.332, +0.635]**. There is no evidence that adding Doppler features improves the state model.

Doppler Core also underperforms the historical earnings mean by +2.547 QLIKE, with a 95% interval [1.464, 3.752].

### First reaction-session gap + Parkinson target (new 11, 123 common OOS events)

| Model | QLIKE |
|---|---:|
| Historical earnings mean | -4.129 |
| Historical earnings median | -3.647 |
| Doppler System Ridge | -3.534 |
| Doppler + GARCH Ridge | -3.524 |
| State-only Ridge | -3.494 |
| Earnings-calibrated GARCH | -3.434 |
| Doppler Core Ridge | -3.389 |
| Plain GARCH(1,1) | 4.844 |

Doppler Core beats plain GARCH very strongly: QLIKE difference **-8.233**, quarter-cluster 95% interval **[-10.339, -6.014]**.

But Doppler System vs State-only is only **-0.035**, 95% interval **[-0.121, +0.044]** -- again not statistically distinguishable from zero.

## True 2026 holdout

After fixing the timing-aware specification, 2026 was held out and the model was trained only on 2020-2025. The price files support 34 completed earnings events through 9 September 2026 (31 excluding AAPL).

### Immediate squared-return target, excluding AAPL (31 events)

| Model | QLIKE |
|---|---:|
| Historical earnings mean | -3.543 |
| State-only Ridge | -2.442 |
| Doppler System Ridge | -2.387 |
| Doppler + GARCH Ridge | -2.281 |
| Historical earnings median | -2.072 |
| Doppler Core Ridge | -2.024 |
| Earnings-calibrated GARCH | -1.993 |
| Plain GARCH(1,1) | 6.288 |

Doppler Core vs GARCH: **-8.312 QLIKE**, ordinary event bootstrap 95% interval **[-14.286, -3.221]**.

Doppler System vs State-only: **+0.055**, 95% interval **[-0.396, +0.477]**. The frequency variables again do not show incremental immediate-event value.

Historical earnings mean beats Doppler Core by **1.519 QLIKE**, with 95% interval **[0.457, 2.755]**.

### Immediate gap + Parkinson target, excluding AAPL (31 events)

| Model | QLIKE |
|---|---:|
| Historical earnings mean | -3.622 |
| Earnings-calibrated GARCH | -3.363 |
| State-only Ridge | -3.362 |
| Historical earnings median | -3.118 |
| Doppler System Ridge | -3.100 |
| Doppler + GARCH Ridge | -2.928 |
| Doppler Core Ridge | -2.138 |
| Plain GARCH(1,1) | 5.898 |

Again, Doppler Core strongly beats plain GARCH but loses to the event-aware and state baselines.

## Longer horizons

Across 3-day and 5-day realized variance, Doppler Core continues to beat plain GARCH in the 2023-2025 sample. The advantage decays with horizon. The Doppler System does not reliably beat State-only.

An exploratory event-aware regression found one small effect at the 5-day horizon: EventBaseline + State + Doppler beat EventBaseline + State by about -0.055 QLIKE in the ex-AAPL panel, with a quarter-cluster interval approximately [-0.112, -0.010]. This was a secondary analysis motivated after viewing the main results and should **not** be treated as confirmatory evidence.

## What the experiment actually tells us

The statement "Doppler beats GARCH" is true for these data under QLIKE, but scientifically incomplete.

Plain GARCH is not told that an earnings release is imminent. A historical earnings mean is told, in effect, what variance Apple/NVIDIA/Meta/etc. typically experience on earnings. That simple event-aware benchmark is difficult to beat and remains strongest on the immediate targets, including the untouched 2026 holdout.

Therefore the current evidence is:

- **Against plain GARCH:** strong and repeatable advantage for Doppler-based/event-aware models.
- **Against a simple state model:** no reliable incremental Doppler benefit at the immediate horizon.
- **Against historical earnings variance:** current Doppler model loses.
- **Unique Doppler hypothesis:** not validated yet.

## Recommended next experiment

Do not tune the daily Doppler model further on this panel. That risks data mining.

Instead:

1. Preserve these daily specifications as frozen benchmarks.
2. Obtain intraday 5-minute OHLCV (preferably trades/quotes) and exact announcement timestamps.
3. Create multiple causal snapshots at T-4h, T-2h, T-1h, T-30m, T-15m and T-5m.
4. Measure D(tau), signed frequency shift, dD/dtau and d2D/dtau2 as tau -> 0.
5. Compare against an **earnings-aware** volatility baseline, not plain GARCH alone.
6. Test incremental loss reduction from adding frequency features to EventBaseline + State + GARCH.
7. Keep a new set of companies or later events fully untouched for final validation.

That experiment tests the actual Doppler hypothesis -- acceleration/compression of market-frequency structure as a fixed catalyst approaches -- rather than merely testing whether a generic spectral model knows earnings are volatile.

## Event timing sources

Price files: user-supplied NASDAQ historical data.

Event dates/timing were checked against SEC-derived historical earnings pages and company IR where available. Source map is included in `event_source_map.csv`. Key references include:

- https://quant500.com/earnings-date/AMAT
- https://quant500.com/earnings-date/AVGO
- https://quant500.com/earnings-date/COST
- https://quant500.com/earnings-date/CSCO
- https://quant500.com/earnings-date/LIN
- https://quant500.com/earnings-date/LRCX
- https://quant500.com/earnings-date/META
- https://quant500.com/earnings-date/MU
- https://quant500.com/earnings-date/NVDA
- https://quant500.com/earnings-date/TSLA
- https://quant500.com/earnings-date/WMT
- Apple event dates/source URLs are contained in the prior Apple research event file.

## Important limitations

- Daily bars are a coarse proxy for a frequency-deformation hypothesis that is naturally intraday.
- Historical event timing was verified most carefully for the test/holdout years; one known AVGO 2022 timing exception was corrected. A production-grade study should use exact point-in-time announcement timestamps for every event.
- The company set is selected rather than a point-in-time broad universe; results are not a deployable trading strategy backtest.
- No transaction costs, option prices, IV expectations, or trading rule are evaluated here. This is a volatility-forecast comparison, not evidence of tradable profitability.
- Bootstrap uncertainty is exploratory and does not correct for every form of cross-sectional dependence or multiple testing.
