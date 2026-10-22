# FinAgent pilot · runs produced with 0.2.2, scored with 0.2.3

24 runs · 12 tickers (AAPL, AMD, BAC, BRK-B, GOOGL, GS, JPM, KO, META, MSFT, NVDA, XOM) · model `nemotron-3-nano:4b-cpu`

| Variant | Runs (ok) | Median s | Number provenance | Claim attribution (draft → final) | Facet coverage | Tool calls by model | Nudges/run | Revisions/run | Tokens/run |
|---|---|---|---|---|---|---|---|---|---|
| full | 24 (24) | 126.2 | 100% | 97% → 98% | 88% | 93% | 0.0 | 0.708 | 10,951 |

<details><summary>Claim issues left in final notes</summary>

- AAPL (full): risk: AAPL exhibits historical Value at Risk and expected shortfall values with low pr -> numbers without a citation
- AMD (full): writer: AMD is relatively risky for a long-term investor due to its high beta of 2.31  a -> 2.31: period '3y' missing
- BAC (full): risk: BAC exhibits significant downside risk with historical Value at Risk and expecte -> numbers without a citation
- BAC (full): writer: BAC is moderately risky for a long-term investor, with historical Value at Risk  -> 10%: period '10d' missing; 10%: says expected shortfall but the source says value at risk
- BRK-B (full): fundamentals: Revenue $249.71bn (growth -2.04%) -> $249.71bn: fiscal year missing (source: FY ending 2024, not the latest); -2.04%: fiscal year missing (source: FY ending 2024, not the latest)
- BRK-B (full): fundamentals: Revenue $254.91bn (growth 22.69%) -> $254.91bn: fiscal year missing (source: FY ending 2023, not the latest); 22.69%: fiscal year missing (source: FY ending 2023, not the latest)
- BRK-B (full): fundamentals: Revenue $207.77bn (growth 8.64%) -> $207.77bn: fiscal year missing (source: FY ending 2022, not the latest); 8.64%: fiscal year missing (source: FY ending 2022, not the latest)
- BRK-B (full): fundamentals: Revenue $191.25bn (growth 14.71%) -> $191.25bn: fiscal year missing (source: FY ending 2021, not the latest); 14.71%: fiscal year missing (source: FY ending 2021, not the latest)
- BRK-B (full): writer: BRK-B has shown lower volatility (17.14%) and lower beta (0.58) than the S&P 500 -> 17.14%: period '5y' missing; 0.58: period '5y' missing
- GS (full): filings: The 10-K does not explicitly flag any contingent liabilities or specific legal r -> not supported by the cited passage (word overlap 17%)
- GS (full): writer: The 10-K does not explicitly flag any contingent liabilities or specific legal r -> not supported by the cited passage (word overlap 17%)
- GS (full): writer: The stock's premium return of 18.2%  is not fully justified by its risk-adjusted -> 18.2%: period '1y' missing; -3.61%: period '1y' missing
- JPM (full): filings: Certain governmental policies or actions could significantly affect U.S. and glo -> not supported by the cited passage (word overlap 17%)
- MSFT (full): filings: The company holds $42,416 in accounts payable and $27,724 in current portion of  -> $42,416 is not in the cited source; $27,724 is not in the cited source

</details>

