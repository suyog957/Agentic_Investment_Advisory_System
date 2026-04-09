"""
Streamlit UI for the Agentic Investment Advisory System.

Shows the full three-layer interaction:
  Client <-> Advisor  (chat bubbles)
  Advisor -> Analyst  (research panel with tool calls)
  Analyst -> Advisor  (structured report card)
"""

import streamlit as st
from groq import RateLimitError
from schemas import ClientProfile
from orchestrator import Orchestrator
from events import Event

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Investment Advisory System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — professional, no emojis, bank-style ─────────────────────────────────
st.markdown("""
<style>
/* Global font */
html, body, [class*="css"] { font-family: 'Segoe UI', Arial, sans-serif; }

/* Analyst research block */
.analyst-header {
    background: #1a2e4a;
    color: #e8edf3;
    padding: 10px 16px;
    border-radius: 4px 4px 0 0;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.analyst-body {
    background: #f4f7fb;
    border: 1px solid #c9d6e3;
    border-top: none;
    border-radius: 0 0 4px 4px;
    padding: 14px 18px;
    margin-bottom: 18px;
}

/* Section label inside analyst body */
.section-label {
    font-weight: 700;
    color: #1a2e4a;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 10px 0 4px 0;
    border-bottom: 1px solid #c9d6e3;
    padding-bottom: 3px;
}

/* Session complete banner */
.complete-banner {
    background: #1a2e4a;
    color: #ffffff;
    padding: 14px 20px;
    border-radius: 4px;
    text-align: center;
    font-size: 1rem;
    font-weight: 600;
    margin: 24px 0 8px 0;
    letter-spacing: 0.02em;
}
.complete-banner small {
    font-weight: 400;
    font-size: 0.85rem;
    opacity: 0.85;
}

/* Rate-limit warning */
.rate-limit-box {
    background: #fff8e1;
    border-left: 4px solid #f59e0b;
    padding: 14px 18px;
    border-radius: 0 4px 4px 0;
    margin: 16px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar — Client Profile ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Client Profile")
    st.divider()

    name = st.text_input("Full Name", value="Sarah Chen")
    age = st.number_input("Age", min_value=18, max_value=80, value=42)
    assets = st.number_input("Investable Assets (USD)", min_value=10_000, value=350_000, step=10_000)
    income = st.number_input("Annual Income (USD)", min_value=0, value=145_000, step=5_000)
    risk = st.selectbox("Risk Tolerance", ["conservative", "moderate", "aggressive"], index=1)
    horizon = st.number_input("Investment Horizon (years)", min_value=1, max_value=40, value=20)

    st.divider()
    st.markdown("**Investment Goals**")
    goal_retirement = st.checkbox("Retirement", value=True)
    goal_education = st.checkbox("Children's Education", value=True)

    st.markdown("**Constraints**")
    no_crypto = st.checkbox("Exclude cryptocurrency", value=True)
    etf_pref = st.checkbox("Prefer ETFs", value=True)
    no_leverage = st.checkbox("Exclude leveraged funds", value=True)

    st.divider()
    existing = st.text_area(
        "Existing Investments",
        value="401(k) with $120,000 balance, checking/savings $30,000 emergency fund",
    )
    context = st.text_area(
        "Additional Context",
        value=(
            "Has two children aged 8 and 11. Currently maxing out 401(k). "
            "Wants to retire at 65. Needs ~$80,000/year for college per child "
            "starting in 7 and 10 years."
        ),
    )

    st.divider()
    max_turns = st.slider("Max conversation turns", min_value=4, max_value=16, value=10)


# ── Build profile from sidebar ─────────────────────────────────────────────────
def build_profile() -> ClientProfile:
    goals = []
    if goal_retirement:
        goals.append("retirement at age 65")
    if goal_education:
        goals.append("children's college education")
    if not goals:
        goals = ["wealth preservation"]

    constraints = []
    if no_crypto:
        constraints.append("no cryptocurrency")
    if etf_pref:
        constraints.append("prefer ETFs over individual stocks")
    if no_leverage:
        constraints.append("no leveraged funds")
    if not constraints:
        constraints = ["none"]

    return ClientProfile(
        name=name,
        age=age,
        assets=float(assets),
        annual_income=float(income),
        risk_tolerance=risk,
        investment_horizon_years=horizon,
        goals=goals,
        constraints=constraints,
        liquidity_need="low",
        existing_investments=existing,
        additional_context=context,
    )


# ── Render helpers ─────────────────────────────────────────────────────────────

def render_analyst_block(analyst_events: list[Event]) -> None:
    """Render a grouped set of analyst events as a collapsible research panel."""
    task_event = next((e for e in analyst_events if e.kind == "analyst_task"), None)
    report_event = next((e for e in analyst_events if e.kind == "analyst_report"), None)

    # Pair tool_call with the immediately following tool_result
    tool_pairs: list[tuple[Event, Event]] = []
    for i, e in enumerate(analyst_events):
        if e.kind == "tool_call" and i + 1 < len(analyst_events) and analyst_events[i + 1].kind == "tool_result":
            tool_pairs.append((e, analyst_events[i + 1]))

    objective = task_event.data["task"].objective if task_event else "Research"

    st.markdown(
        f'<div class="analyst-header">Analyst Research &nbsp;|&nbsp; {objective}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="analyst-body">', unsafe_allow_html=True)

    # Research questions
    if task_event and task_event.data["task"].research_questions:
        with st.expander("Research Questions", expanded=False):
            for q in task_event.data["task"].research_questions:
                st.markdown(f"- {q}")

    # Tool calls
    if tool_pairs:
        st.markdown('<p class="section-label">Tool Calls</p>', unsafe_allow_html=True)
        for call_evt, result_evt in tool_pairs:
            tool_name = call_evt.data["tool"]
            args = call_evt.data["args"]
            result = result_evt.data["result"]

            label = "Knowledge Base" if tool_name == "search_knowledge_base" else "Web Search"
            query = args.get("query", str(args))

            with st.expander(f"{label}  —  {query}", expanded=False):
                st.caption(f"Tool: `{tool_name}`")
                st.json(args)
                st.divider()
                st.caption("Result")
                st.text(result[:2000] + (" [truncated]" if len(result) > 2000 else ""))

    # Analyst report
    if report_event:
        report = report_event.data["report"]
        st.markdown('<p class="section-label">Research Report</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.caption("Market Summary")
            st.write(report.market_summary)

            if report.allocation_breakdown:
                st.caption("Recommended Allocation")
                st.bar_chart(report.allocation_breakdown, height=200)

        with col2:
            if report.etf_suggestions:
                st.caption("Suggested ETFs")
                for etf in report.etf_suggestions:
                    st.markdown(
                        f"**{etf.get('ticker', '?')}** &nbsp; {etf.get('name', '')}  \n"
                        f"<small style='color:#555'>{etf.get('rationale', '')}</small>",
                        unsafe_allow_html=True,
                    )
                st.write("")

            if report.risks:
                st.caption("Key Risks")
                for risk_item in report.risks:
                    st.markdown(f"- {risk_item}")

        if report.recommendations:
            st.caption("Recommendations")
            for rec in report.recommendations:
                st.markdown(f"- {rec}")

        if report.sources:
            st.caption(f"Sources: {' · '.join(report.sources)}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_events(events: list[Event]) -> None:
    """Render the full event list — chat turns with inline analyst panels."""
    pending_analyst: list[Event] = []

    def flush_analyst() -> None:
        if pending_analyst:
            render_analyst_block(list(pending_analyst))
            pending_analyst.clear()

    for event in events:
        if event.kind == "client":
            flush_analyst()
            with st.chat_message("user"):
                st.markdown(f"**{event.data['name']}**")
                st.write(event.data["content"])

        elif event.kind == "advisor":
            flush_analyst()
            with st.chat_message("assistant"):
                st.markdown("**Financial Advisor**")
                st.write(event.data["content"])

        elif event.kind in ("analyst_task", "tool_call", "tool_result", "analyst_report"):
            pending_analyst.append(event)

        elif event.kind == "session_complete":
            flush_analyst()
            reason = event.data.get("reason", "")
            turns = event.data.get("turns", "?")
            st.markdown(
                f'<div class="complete-banner">Session Complete &mdash; {turns} turn(s)'
                f'<br><small>{reason}</small></div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Disclaimer: This is an educational simulation. "
                "It does not constitute financial advice."
            )

    flush_analyst()


# ── Main page ──────────────────────────────────────────────────────────────────
st.markdown("## Investment Advisory System")
st.caption("Multi-agent simulation — Client / Advisor / Analyst")
st.divider()

# Session state
if "events" not in st.session_state:
    st.session_state.events = []
if "error_msg" not in st.session_state:
    st.session_state.error_msg = ""

col_btn1, col_btn2, _ = st.columns([1, 1, 4])

with col_btn1:
    start = st.button("Start Session", type="primary", use_container_width=True)
with col_btn2:
    if st.button("Reset", use_container_width=True):
        st.session_state.events = []
        st.session_state.error_msg = ""
        st.rerun()

# Show rate-limit or other error from previous run
if st.session_state.error_msg:
    st.markdown(
        f'<div class="rate-limit-box"><strong>Session could not complete</strong><br>{st.session_state.error_msg}</div>',
        unsafe_allow_html=True,
    )

if start:
    st.session_state.events = []
    st.session_state.error_msg = ""

    import config
    config.MAX_TURNS = max_turns

    profile = build_profile()

    try:
        with st.spinner("Running multi-agent advisory session. This typically takes 1-3 minutes."):
            session = Orchestrator(profile=profile)
            st.session_state.events = session.collect_events()
    except RateLimitError as e:
        # Parse the retry-after hint from the error message if present
        msg = str(e)
        retry_hint = ""
        import re
        m = re.search(r"try again in ([\d]+m[\d.]+s)", msg)
        if m:
            retry_hint = f" Please retry in {m.group(1)}."
        st.session_state.error_msg = (
            f"The Groq API daily token limit has been reached.{retry_hint} "
            f"Any results collected before the limit was hit are shown below."
        )
    except Exception as e:
        st.session_state.error_msg = f"Unexpected error: {e}"

    st.rerun()

# Render whatever events we have (partial results are fine)
if st.session_state.events:
    st.markdown("### Advisory Session")
    render_events(st.session_state.events)
elif not st.session_state.error_msg:
    st.info(
        "Configure the client profile in the sidebar, then click **Start Session**."
    )
    st.markdown(
        "The session will show the full three-layer interaction: "
        "client and advisor conversation turns, the research tasks the advisor "
        "delegates to the analyst, every tool call the analyst makes (knowledge "
        "base and web search), and the structured report the analyst returns."
    )
