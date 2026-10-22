# FinAgent ablation study · version 0.2.0, valuation question

36 runs · 12 tickers (AAPL, AMD, BAC, BRK-B, GOOGL, GS, JPM, KO, META, MSFT, NVDA, XOM) · model `nemotron-3-nano:4b-cpu`

| Variant | Runs (ok) | Median s | Number provenance | Claim attribution (draft → final) | Facet coverage | Tool calls by model | Nudges/run | Revisions/run | Tokens/run |
|---|---|---|---|---|---|---|---|---|---|
| full | 12 (12) | 127.4 | 100% | 96% → 98% | 89% | 97% | 0.0 | 0.917 | 11,581 |
| -facts | 6 (6) | 164.3 | 98% | 70% → 84% | 83% | 98% | 0.0 | 2.833 | 14,897 |
| -phases | 6 (6) | 163.1 | 99% | 92% → 96% | 94% | 95% | 0.0 | 1.333 | 14,946 |
| -coverage | 6 (6) | 127.7 | 99% | 94% → 97% | 89% | 100% | 0.0 | 1.167 | 11,876 |
| -attribution | 6 (6) | 110.9 | 99% | 92% → 92% | 89% | 98% | 0.0 | 0.167 | 10,120 |

<details><summary>Claim issues left in final notes</summary>

- AAPL (full): writer: AAPL 95% 10-day Value at Risk, filtered historical simulation: 7.13% -> 7.13%: says value at risk but the source says expected shortfall
- MSFT (full): filings: The company holds $42,416 in accounts payable and $27,724 in current portion of  -> $42,416 is not in the cited source; $27,724 is not in the cited source
- META (full): filings: This information is relevant to the company’s ongoing tax and currency risk expo -> not supported by the cited passage (word overlap 25%)
- JPM (full): filings: Certain governmental policies or actions could significantly affect U.S. and glo -> not supported by the cited passage (word overlap 17%)
- GS (full): filings: The 10-K filing was filed on 2026-02-25 for the period 2025-12-31 -> not supported by the cited passage (word overlap 0%)
- XOM (full): filings: XOM's dividend yield is 3.2% -> cites evidence it was not given (D2)
- XOM (full): filings: The company's free cash flow is $12.5 billion -> cites evidence it was not given (D2)
- XOM (full): filings: XOM's debt-to-equity ratio is 0.8 -> cites evidence it was not given (D2)
- XOM (full): filings: The company faces regulatory risks from oil price volatility -> cites evidence it was not given (D2)
- NVDA (-attribution): fundamentals: NVDA revenue $130.5bn (growth 114.2%) -> $130.5bn: fiscal year missing (source: FY ending 2025, not the latest); 114.2%: fiscal year missing (source: FY ending 2025, not the latest)
- NVDA (-attribution): fundamentals: NVDA revenue $60.92bn (growth 125.85%) -> $60.92bn: fiscal year missing (source: FY ending 2024, not the latest); 125.85%: fiscal year missing (source: FY ending 2024, not the latest)
- NVDA (-attribution): writer: NVDA revenue $130.5bn (growth 114.2%) -> $130.5bn: fiscal year missing (source: FY ending 2025, not the latest); 114.2%: fiscal year missing (source: FY ending 2025, not the latest)
- NVDA (-attribution): writer: NVDA revenue $60.92bn (growth 125.85%) -> $60.92bn: fiscal year missing (source: FY ending 2024, not the latest); 125.85%: fiscal year missing (source: FY ending 2024, not the latest)
- AAPL (-facts): market: AAPL's tracking error over 5y is 19.87% -> 19.87% is not in the cited source
- AAPL (-phases): risk: AAPL's downside risk is significant with VaR and expected shortfall exceeding 8% -> numbers without a citation
- AAPL (-phases): filings: The Company may not be able to accurately predict, control or mitigate these ris -> not supported by the cited passage (word overlap 6%)
- AAPL (-coverage): writer: AAPL 95% 10-day Value at Risk, filtered historical simulation: 7.13% -> 7.13%: says value at risk but the source says expected shortfall
- AAPL (-attribution): writer: AAPL 95% 10-day Value at Risk, filtered historical simulation: 7.13% -> 7.13%: says value at risk but the source says expected shortfall
- JPM (-facts): fundamentals: JPMORGAN CHASE & CO net income was 49.55 billion in 2023 -> 49.55 billion is not in the cited source
- JPM (-facts): fundamentals: JPMORGAN CHASE & CO TTM market cap is 891.18 billion -> 891.18 billion: period 'ttm' is not the source's
- JPM (-facts): filings: Certain governmental policies or actions could significantly affect U.S. and glo -> not supported by the cited passage (word overlap 17%)
- JPM (-coverage): filings: Certain governmental policies or actions could significantly affect U.S. and glo -> not supported by the cited passage (word overlap 17%)
- JPM (-attribution): filings: Certain governmental policies could significantly affect economic growth and cau -> not supported by the cited passage (word overlap 25%)
- JPM (-attribution): writer: Certain governmental policies could significantly affect economic growth and cau -> not supported by the cited passage (word overlap 25%)
- GOOGL (-facts): fundamentals: Net margin increased from 24.01% in 2023 to 28.6% in 2024 to 32.81% in 2025 -> 24.01% is not in the cited source; 28.6% is not in the cited source
- GOOGL (-facts): fundamentals: Revenue growth is 14.47% CAGR and free cash flow yield is 1.76% -> 14.47% is not in the cited source
- GOOGL (-facts): fundamentals: Revenue growth is 14.47% CAGR and free cash flow yield is 1.76% -> 14.47% is not in the cited source
- GOOGL (-facts): fundamentals: Revenue growth is 14.47% CAGR and free cash flow yield is 1.76% -> 14.47% is not in the cited source
- GOOGL (-facts): market: Alphabet Inc. tracking error 24.36% over 5y -> 24.36% is not in the cited source
- GOOGL (-attribution): writer: The company's high operating and net margins of 32.03% and 32.81%  support its c -> 32.81%: says operating but the source says net
- KO (-phases): filings: We conduct business in markets with high-risk legal compliance environments, whi -> not supported by the cited passage (word overlap 21%)
- KO (-attribution): filings: Dividend policy is tied to economic stability and asset risk, with recovery of e -> not supported by the cited passage (word overlap 31%)
- XOM (-facts): market: total return pct 238.07 -> 238.07: period '5y' missing
- XOM (-facts): market: annualized return pct 27.61 -> 27.61: period '5y' missing
- XOM (-facts): market: beta 0.38 -> 0.38: period '5y' missing
- XOM (-facts): market: excess return pct 156.79 -> 156.79: period '5y' missing
- XOM (-facts): risk: XOM's downside risk exceeds 10% probability in historical stress events -> 10% is not in the cited source
- XOM (-facts): risk: XOM's downside risk exceeds 10% probability in historical stress events -> 10% is not in the cited source
- XOM (-facts): risk: XOM's downside risk exceeds 10% probability in historical stress events -> 10% is not in the cited source
- XOM (-facts): risk: XOM's downside risk exceeds 10% probability in historical stress events -> 10% is not in the cited source
- XOM (-facts): risk: XOM's downside risk exceeds 10% probability in historical stress events -> 10% is not in the cited source
- XOM (-facts): risk: XOM's downside risk exceeds 10% probability in historical stress events -> 10% is not in the cited source
- XOM (-facts): risk: XOM's downside risk exceeds 10% probability in historical stress events -> 10% is not in the cited source
- XOM (-facts): risk: XOM's downside risk exceeds 10% probability in historical stress events -> 10% is not in the cited source
- XOM (-facts): risk: XOM's downside risk exceeds 10% probability in historical stress events -> 10% is not in the cited source
- XOM (-facts): risk: XOM's downside risk exceeds 10% probability in historical stress events -> 10% is not in the cited source
- XOM (-facts): risk: XOM's downside risk exceeds 10% probability in historical stress events -> 10% is not in the cited source
- XOM (-facts): risk: XOM's downside risk exceeds 10% probability in historical stress events -> 10% is not in the cited source
- XOM (-facts): risk: XOM's downside risk exceeds 10% probability in historical stress events -> 10% is not in the cited source
- XOM (-facts): filings: XOM's dividend yield is 3.2% -> cites evidence it was not given (D2)
- XOM (-facts): filings: The company's free cash flow is $12.5 billion -> cites evidence it was not given (D2)
- XOM (-facts): filings: XOM's debt-to-equity ratio is 0.8 -> cites evidence it was not given (D2)
- XOM (-facts): filings: The company faces regulatory risks from oil price volatility -> cites evidence it was not given (D2)
- XOM (-facts): writer: XOM's downside risk exceeds 10% probability in historical stress events -> 10% is not in the cited source
- XOM (-facts): writer: XOM's dividend yield is 3.2% -> cites evidence it was not given (D2)
- XOM (-facts): writer: The company's free cash flow is $12.5 billion -> cites evidence it was not given (D2)
- XOM (-phases): risk: XOM's downside risk is significant with a 95% VaR of 6.81% and expected shortfal -> numbers without a citation
- XOM (-phases): filings: XOM's valuation is not justified by its fundamentals as the company's earnings p -> cites evidence it was not given (D4)
- XOM (-phases): filings: The company faces significant risks related to its exposure to volatile oil pric -> cites evidence it was not given (D4)
- XOM (-phases): filings: XOM's business model is challenged by increasing regulatory pressures on fossil  -> cites evidence it was not given (D4)

</details>

