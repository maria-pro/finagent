"""FinAgent web UI: run the agent team live, or replay a saved run.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finagent import analytics as an
from finagent.accel import array_backend
from finagent.config import load_settings, shared_ui, ui_api_key
from finagent.data.prices import adjusted_close
from finagent.runs import build_record, list_runs, load_run, save_run
from finagent.tools import ToolContext, parse_tickers

st.set_page_config(page_title="FinAgent", page_icon="📈", layout="wide")

# Reference categorical palette (fixed order) - light and dark steps.
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"],
}
NEUTRAL = {"light": "#8a8985", "dark": "#9a9992"}
AGENT_ICON = {"market": "📈", "fundamentals": "🧾", "risk": "🛡️", "filings": "📄", "writer": "✍️"}
EXAMPLES = [
    ("NVDA", "Is NVIDIA's valuation justified by its fundamentals, and what are the main risks?"),
    ("NVDA, AMD", "Compare NVIDIA and AMD on growth, valuation and downside risk."),
    ("JPM", "How risky is JPMorgan for a long-term investor, and what does its 10-K flag?"),
]


def md(text: str) -> str:
    """Streamlit renders $...$ as LaTeX; escape dollar signs so currency stays currency."""
    return text.replace("$", "\\$")


def theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def colors() -> tuple[list[str], str]:
    mode = theme()
    return SERIES[mode], NEUTRAL[mode]


def style(fig: go.Figure, title: str, y_title: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15}},
        height=height,
        margin={"l": 10, "r": 10, "t": 48, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.18},
        yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


# --------------------------------------------------------------------------- sidebar


def sidebar():
    st.sidebar.header("Setup")
    profile = st.sidebar.radio(
        "Model endpoint",
        ["local", "nim"],
        format_func=lambda p: "Local CPU (Ollama / llama.cpp)" if p == "local" else "NVIDIA API catalog",
    )
    base = load_settings(profile)
    shared = shared_ui()
    with st.sidebar.expander("Endpoint & models", expanded=False):
        # In a shared deployment the endpoints are fixed by whoever runs the server.
        base_url = st.text_input("Base URL", base.base_url, disabled=shared)
        orchestrator = st.text_input("Orchestrator model (planner, writer)", base.orchestrator_model, disabled=shared)
        specialist = st.text_input("Specialist model (sub-agents)", base.specialist_model, disabled=shared)
        typed_key = ""
        if profile == "nim" and not shared:
            # Never pre-filled: a key held by the server must not be sent to the browser.
            typed_key = st.text_input("API key", "", type="password", help="Leave empty to use the key set on the server.")
            if base.api_key:
                st.caption("A key is set on the server. It is used only with the endpoint it is configured for.")
        reasoning = st.toggle("Reasoning (thinking) mode", base.reasoning, help="Slower; off is recommended on CPU.")
        concurrency = st.slider("Parallel agents", 1, 4, base.max_concurrency, help="Use 1 on a CPU.")
    settings = base.with_overrides(
        base_url=base_url, orchestrator_model=orchestrator, specialist_model=specialist,
        api_key=ui_api_key(base, base_url, typed_key), reasoning=reasoning, max_concurrency=concurrency,
    )

    st.sidebar.header("Saved runs")
    # Shared deployments list only the shipped examples and this visitor's own runs.
    labels = {f"{p.parent.name}/{p.stem}": p for p in list_runs(("examples",) if shared else ("reports", "examples"))}
    labels.update({f"this session/{label}": label for label in st.session_state.get("my_runs", {})})
    choice = None
    if labels:
        picked = st.sidebar.selectbox("Replay a run", ["—"] + list(labels), help="Show a previous run without calling any model.")
        if picked != "—":
            choice = labels[picked]
    if not shared:
        st.sidebar.caption("Runs you make are saved to ./reports and can be replayed here.")
    else:
        st.sidebar.caption("Your runs stay in this browser session; download them to keep them.")
    return settings, choice


def badges(settings):
    local = settings.is_local
    cols = st.columns(4)
    cols[0].metric("Specialist model", settings.specialist_model.split("/")[-1])
    cols[1].metric("Inference", "Local CPU" if local else "NVIDIA API catalog")
    cols[2].metric("Risk engine", array_backend()[1].split(" (")[0])
    cols[3].metric("Data leaves machine", "Market data only" if local else "Prompts + data")


# --------------------------------------------------------------------------- live run


class LiveTrace:
    """Renders workflow events into one status box per agent while the run is in progress."""

    def __init__(self):
        self.orchestrator = st.status("Orchestrator", expanded=True)
        self.boxes: dict[str, object] = {}
        self.events: list[dict] = []

    def __call__(self, event: dict):
        self.events.append(event)
        kind, agent = event["event"], event.get("agent", "")
        if kind == "guard":
            self.orchestrator.write(("✅ Guardrail passed" if event["allowed"] else f"⛔ Refused: {event['reason']}") + f" · {event['rail']}")
        elif kind == "plan":
            self.orchestrator.write(f"🗺️ Plan ({event['source']}): " + " · ".join(t["agent"] for t in event["tasks"]))
            for t in event["tasks"]:
                self.orchestrator.caption(f"{t['agent']}: {t['focus']}")
        elif kind == "agent_start":
            self.boxes[agent] = st.status(f"{AGENT_ICON.get(agent, '🤖')} {event['title']} — working", expanded=False)
        elif kind == "llm" and agent in self.boxes:
            calls = ", ".join(event["tool_calls"]) or "writes findings"
            self.boxes[agent].write(f"model · {event['latency_s']:.1f}s · {event['prompt_tokens']}→{event['completion_tokens']} tokens → {calls}")
        elif kind == "tool" and agent in self.boxes:
            how = "" if event["chosen_by"].startswith("model") else f" _(added by {event['chosen_by']})_"
            err = f" ⚠️ {event['error']}" if event.get("error") else ""
            args = ", ".join(f"{k}={v}" for k, v in event["args"].items())
            self.boxes[agent].write(md(f"**{event['id']}** `{event['tool']}({args})` · {event['seconds']}s{how}{err}"))
        elif kind == "agent_done" and agent in self.boxes:
            self.boxes[agent].update(label=f"{AGENT_ICON.get(agent, '🤖')} {agent.title()} — done in {event['seconds']}s", state="complete")
        elif kind == "verify":
            note = f" · untraced: {', '.join(event['unverified'])}" if event["unverified"] else ""
            when = " (after revision)" if event.get("after_revision") else ""
            self.orchestrator.write(f"🔎 Fact-check {agent}{when}: {event['verified']}/{event['checked']} numbers traced{note}")
        elif kind == "revise":
            self.orchestrator.write(f"↺ {agent} revising its findings")
        elif kind == "done":
            if "writer" in self.boxes:
                self.boxes["writer"].update(label="✍️ Lead Analyst — done", state="complete")
            self.orchestrator.update(label="Orchestrator — report assembled", state="complete", expanded=False)


def run_live(settings, tickers: list[str], question: str) -> dict | None:
    from finagent.graph import run_research
    from finagent.llm import OpenAICompatLLM

    trace = LiveTrace()
    try:
        final = run_research(OpenAICompatLLM(settings), ToolContext(settings), tickers, question, on_event=trace)
    except Exception as err:
        st.error(f"The run failed: {type(err).__name__}: {err}. Check the endpoint in the sidebar (run `finagent doctor`).")
        return None
    if shared_ui():
        # Nothing is written to the shared reports/ folder: the run stays with this visitor.
        record = json.loads(json.dumps(build_record(final, trace.events, settings), default=str))
        label = f"{datetime.now():%H%M%S}-{'-'.join(tickers).lower()}"
        st.session_state.setdefault("my_runs", {})[label] = record
        return record
    path = save_run(final, trace.events, settings)
    record = load_run(path)
    record["_path"] = str(path)
    return record


# --------------------------------------------------------------------------- results


def price_charts(settings, tickers: list[str]):
    series_colors, neutral = colors()
    try:
        prices = {t: adjusted_close(settings, t, "5y") for t in tickers + ["SPY"]}
    except Exception as err:
        st.info(f"Charts unavailable: {err}")
        return
    window = st.segmented_control("Window", ["1y", "3y", "5y"], default="1y", key="window") or "1y"
    days = {"1y": 365, "3y": 1095, "5y": 1826}[window]

    fig = go.Figure()
    dd = go.Figure()
    for i, (t, s) in enumerate(prices.items()):
        s = s[s.index >= s.index[-1] - pd.Timedelta(days=days)]
        is_bench = t == "SPY"
        color = neutral if is_bench else series_colors[i % len(series_colors)]
        line = {"color": color, "width": 2, "dash": "dot" if is_bench else "solid"}
        indexed = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(x=indexed.index, y=indexed, name=t, line=line, hovertemplate="%{y:.1f}"))
        if not is_bench:
            fig.add_annotation(x=indexed.index[-1], y=indexed.iloc[-1], text=t, showarrow=False, xanchor="left", xshift=6)
            draw = an.drawdown_series(s) * 100
            dd.add_trace(go.Scatter(x=draw.index, y=draw, name=t, line={"color": color, "width": 2}, hovertemplate="%{y:.1f}%"))
    c1, c2 = st.columns(2)
    c1.plotly_chart(style(fig, f"Growth of 100 ({window}, vs SPY)", "Indexed value"), width="stretch")
    c2.plotly_chart(style(dd, f"Drawdown from running peak ({window})", "%"), width="stretch")


def risk_chart(settings, record: dict):
    ev = next((e for e in record.get("evidence", []) if e["tool"] == "value_at_risk" and not e.get("error")), None)
    if not ev:
        return
    series_colors, _ = colors()
    r = ev["result"]
    tickers = list(r["portfolio"])
    weights = np.array(list(r["portfolio"].values()))
    frame = pd.concat([adjusted_close(settings, t, "5y") for t in tickers], axis=1, join="inner").pct_change().dropna()
    xp, backend = array_backend()
    sim = an.filtered_historical_simulation(frame.to_numpy(), weights, horizon_days=r["horizon_days"], n_paths=100_000,
                                            confidence=r["confidence_pct"] / 100, xp=xp, backend=backend, keep_samples=True)
    hist = r.get("historical_var_pct")
    fig = go.Figure(go.Histogram(x=sim.samples * 100, nbinsx=120, marker={"color": series_colors[0]}, name="simulated", hovertemplate="%{x:.1f}%: %{y} paths<extra></extra>"))
    fig.add_vline(x=-sim.var * 100, line={"color": "#e34948", "width": 2})
    fig.add_annotation(x=-sim.var * 100, y=1, yref="paper", text=f"FHS VaR {r['confidence_pct']:.0f}%: −{sim.var:.1%}", showarrow=False, xanchor="right", xshift=-6)
    if hist is not None:
        fig.add_vline(x=-hist, line={"color": colors()[1], "width": 2, "dash": "dot"})
        fig.add_annotation(x=-hist, y=0.88, yref="paper", text=f"historical VaR: −{hist:.1f}%", showarrow=False, xanchor="left", xshift=6)
    fig.update_xaxes(range=[float(np.percentile(sim.samples, 0.1)) * 100, float(np.percentile(sim.samples, 99.9)) * 100])
    label = " + ".join(f"{w:.0%} {t}" for t, w in r["portfolio"].items())
    title = (f"{r['horizon_days']}-day returns, filtered historical simulation · {label} · current volatility "
             f"{sim.current_vol_annual:.0%} vs 5y average {sim.average_vol_annual:.0%} [{ev['id']}]")
    st.plotly_chart(style(fig, title, "Paths", 320).update_layout(hovermode="closest", xaxis_title="%"), width="stretch")


def fundamentals_charts(record: dict):
    rows = []
    for ev in record.get("evidence", []):
        if ev["tool"] == "financial_statements" and not ev.get("error"):
            for fy in ev["result"].get("fiscal_years", []):
                rows.append({"ticker": ev["result"]["ticker"], **fy})
    if not rows:
        return
    df = pd.DataFrame(rows).drop_duplicates(["ticker", "fiscal_year_end"])
    # Companies name a fiscal year after the calendar year it ends in (NVIDIA's FY2026 ends January 2026).
    df["fy"] = "FY" + df["fiscal_year_end"].str[:4]
    order = sorted(df["fy"].unique())
    series_colors, _ = colors()
    rev, margin = go.Figure(), go.Figure()
    for i, (t, g) in enumerate(df.groupby("ticker", sort=False)):
        c = series_colors[i % len(series_colors)]
        ends = g["fiscal_year_end"]
        rev.add_trace(go.Bar(x=g["fy"], y=g["revenue_bn"], name=t, marker={"color": c}, customdata=ends,
                             hovertemplate=f"{t} · year to %{{customdata}}: $%{{y:.1f}}bn<extra></extra>"))
        margin.add_trace(go.Scatter(x=g["fy"], y=g["operating_margin_pct"], name=t, mode="lines+markers", customdata=ends,
                                    line={"color": c, "width": 2}, marker={"size": 8},
                                    hovertemplate=f"{t} · year to %{{customdata}}: %{{y:.1f}}%<extra></extra>"))
    for fig in (rev, margin):
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=order)
    c1, c2 = st.columns(2)
    c1.plotly_chart(style(rev, "Revenue by fiscal year (SEC filings)", "$bn").update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, barcornerradius=4, hovermode="closest"), width="stretch")
    c2.plotly_chart(style(margin, "Operating margin by fiscal year", "%").update_layout(hovermode="closest"), width="stretch")


def timeline(record: dict):
    events = record.get("events", [])
    if not events:
        return
    t0 = events[0]["t"]
    spans, starts = [], {}
    for e in events:
        if e["event"] == "agent_start":
            starts[e["agent"]] = e["t"]
        elif e["event"] == "llm":
            spans.append({"lane": e["agent"], "start": e["t"] - e["latency_s"] - t0, "dur": e["latency_s"], "kind": "model"})
        elif e["event"] == "tool":
            spans.append({"lane": e["agent"], "start": e["t"] - e["seconds"] - t0, "dur": max(e["seconds"], 0.2), "kind": "tool"})
    plan_llm = [c for c in record.get("llm_calls", []) if c["agent"] == "planner"]
    first_plan = next((e for e in events if e["event"] == "plan"), None)
    if plan_llm and first_plan:
        spans.append({"lane": "planner", "start": first_plan["t"] - plan_llm[0]["latency_s"] - t0, "dur": plan_llm[0]["latency_s"], "kind": "model"})
    if not spans:
        return
    series_colors, neutral = colors()
    lanes = ["planner"] + [t["agent"] for t in record.get("plan", [])] + ["writer"]
    fig = go.Figure()
    for kind, color in (("model", series_colors[0]), ("tool", series_colors[1])):
        part = [s for s in spans if s["kind"] == kind]
        fig.add_trace(go.Bar(y=[s["lane"] for s in part], x=[s["dur"] for s in part], base=[s["start"] for s in part], orientation="h",
                             name="language model call" if kind == "model" else "tool call", marker={"color": color},
                             hovertemplate="%{y}: %{x:.1f}s<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(lanes)))
    st.plotly_chart(style(fig, "Agent timeline (seconds from start)", "", 300).update_layout(barmode="overlay", hovermode="closest", xaxis_title="seconds"), width="stretch")


def show_result(settings, record: dict):
    stats = record.get("stats") or {}
    if stats.get("refused"):
        st.warning(record.get("report", "Refused."))
        return
    prov = stats.get("provenance", {})
    quality = stats.get("quality") or {}
    cols = st.columns(6)
    plain = {"delta_color": "off", "delta_arrow": "off"}
    cols[0].metric("Numbers traced to tools", f"{prov.get('verified', 0)}/{prov.get('checked', 0)}", f"{prov.get('score', 0):.0%}", **plain)
    if quality:
        cols[1].metric("Claims match sources", f"{quality['claims_supported']}/{quality['claims_checked']}", f"{quality['claim_attribution']:.0%}", **plain)
    cols[2].metric("Tool calls", stats.get("tool_calls", 0), f"{stats.get('tool_calls_chosen_by_model', 0)} chosen by the models", **plain)
    cols[3].metric("LLM calls", stats.get("llm_calls", 0), f"{stats.get('revisions', 0)} revision(s)", **plain)
    cols[4].metric("Tokens", f"{stats.get('prompt_tokens', 0) + stats.get('completion_tokens', 0):,}")
    cols[5].metric("End-to-end", f"{stats.get('wall_seconds', 0):.0f}s", f"{stats.get('model_seconds', 0):.0f}s model time", **plain)

    tabs = st.tabs(["Research note", "Charts", "Evidence ledger", "Fact-check", "Agent timeline"])
    with tabs[0]:
        st.markdown(md(record.get("report", "")))
        c1, c2 = st.columns(2)
        stem = f"finagent-{'-'.join(record['tickers']).lower()}-{datetime.now():%Y%m%d}"
        c1.download_button("Download note (.md)", record.get("report", ""), f"{stem}.md", "text/markdown")
        c2.download_button("Download run record (.json)", json.dumps(record, indent=2, default=str), f"{stem}.json", "application/json")
    with tabs[1]:
        price_charts(settings, record["tickers"])
        risk_chart(settings, record)
        fundamentals_charts(record)
    with tabs[2]:
        ledger = pd.DataFrame(
            [{"id": e["id"], "agent": e["agent"], "tool": e["tool"], "arguments": json.dumps(e["args"]),
              "chosen by": e["chosen_by"], "seconds": e["seconds"],
              "data retrieved": min((s["fetched_at"] for s in e.get("sources") or []), default="–"),
              "stale": "⚠ yes" if e.get("stale") else "", "error": e.get("error") or ""} for e in record.get("evidence", [])]
        )
        st.dataframe(ledger, width="stretch", hide_index=True)
        for e in record.get("evidence", []):
            with st.expander(f"{e['id']} · {e['tool']}"):
                st.json(e["result"])
    with tabs[3]:
        st.caption("Two checks run on everything a model writes. Number provenance: every number must trace to a tool result "
                   "the agent received. Claim attribution: every cited claim must keep its source's company, period, metric and "
                   "direction (10-K points are checked by wording). Agents with problems get one revision; anything left is marked ⚠ in the note.")

        def check_row(name, check, revised):
            p, a = check.get("provenance", {}), check.get("attribution") or {}
            return {"agent": name, "numbers traced": f"{p.get('verified', 0)}/{p.get('checked', 0)}",
                    "claims matching source": f"{a.get('supported', 0)}/{a.get('claims', 0)}" if a else "–",
                    "issues": "; ".join(a.get("issues", [])) if a else "", "revised": revised}

        rows = [check_row(f["title"], f, bool(f.get("revised"))) for f in (record.get("findings") or {}).values()]
        if record.get("writer_check"):
            rows.append(check_row("Lead Analyst", record["writer_check"], "writer" in (record.get("revisions") or [])))
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if quality.get("claim_issues"):
            st.warning("Claim issues left in the final note:\n\n" + "\n".join(f"- {md(i)}" for i in quality["claim_issues"]))
        if record.get("narrative_draft") and record.get("narrative_draft") != record.get("narrative"):
            with st.expander("Lead Analyst: first draft vs revision"):
                a, b = st.columns(2)
                a.markdown("**First draft**\n\n" + md(record["narrative_draft"]))
                b.markdown("**After fact-check**\n\n" + md(record["narrative"]))
        for f in (record.get("findings") or {}).values():
            if f.get("revised"):
                with st.expander(f"{f['title']}: first draft vs revision"):
                    a, b = st.columns(2)
                    a.markdown("**First draft**\n\n" + md(f.get("draft", "")))
                    b.markdown("**After fact-check**\n\n" + md(f["text"]))
    with tabs[4]:
        timeline(record)
        calls = pd.DataFrame(record.get("llm_calls", []))
        if not calls.empty:
            st.dataframe(calls, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- page


def main():
    settings, saved = sidebar()
    st.title("FinAgent")
    st.caption("Grounded multi-agent equity research on a small NVIDIA Nemotron model. "
               "Four specialist agents call real data tools; a fact-checker traces every number back to its source.")
    badges(settings)

    if "example" not in st.session_state:
        st.session_state.example = EXAMPLES[0]
    with st.form("ask"):
        c1, c2 = st.columns([1, 4])
        tickers_text = c1.text_input("Tickers", st.session_state.example[0])
        question = c2.text_input("Research question", st.session_state.example[1])
        submitted = st.form_submit_button("Run the agent team", type="primary")
    ex_cols = st.columns(len(EXAMPLES))
    for col, (t, q) in zip(ex_cols, EXAMPLES):
        if col.button(f"{t}: {q[:48]}…", key=f"ex-{t}"):
            st.session_state.example = (t, q)
            st.rerun()

    if submitted:
        if settings.is_local:
            st.info("Running on CPU: expect a few minutes. The trace below updates as each agent works.")
        record = run_live(settings, parse_tickers(tickers_text), question)
        if record:
            st.session_state.record = record
    elif isinstance(saved, str):  # a run from this browser session
        st.session_state.record = st.session_state["my_runs"][saved]
    elif saved is not None:
        st.session_state.record = load_run(saved)

    if st.session_state.get("record"):
        st.divider()
        show_result(settings, st.session_state.record)  # has download buttons for the note and the run record


main()
