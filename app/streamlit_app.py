from __future__ import annotations
import streamlit as st
from app.api_client import call_query, APIError
from app.helpers import compute_em, strip_citations

API_BASE = "http://localhost:8000"
COLLECTION = "hotpot_dev_500"
TIMEOUT_S = 120
STRATEGIES: list[str] = ["top_k", "symmetric", "asymmetric"]

COMPARISON_KEYWORDS: list[str] = [
    # attribute comparisons
    "who is taller", "who is older", "who is younger", "who is bigger",
    "who was born", "who died", "who had", "who has",
    # both/same patterns
    "both", "the same", "same nationality", "same city",
    "were both", "are both", "did both",
    # shared / belong
    "shared", "belong to", "were formed",
    # explicit compare
    "what is the same", "compare",
    "more than", "less than",
]


def classify_question(question: str) -> tuple[str, str]:
    """Return (question_type, strategy). Comparison → top_k, bridge → asymmetric."""
    q_lower = question.lower()
    for kw in COMPARISON_KEYWORDS:
        if kw in q_lower:
            return "comparison", "top_k"
    return "bridge", "asymmetric"

st.set_page_config(page_title="MergeRAG", layout="wide")
st.markdown("""
<style>
    *:not([data-testid="stIconMaterial"]):not(.material-icons):not(.material-symbols-rounded):not(.material-symbols-outlined) {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    }
    .block-container { padding-top: 2rem; }
    [data-testid="stSidebar"] { display: none; }
    code { color: black !important; background-color: #f0f0f0 !important; word-break: break-all; }
    .stExpander { margin-bottom: 4px !important; }
    [data-testid="stExpander"] summary p { word-break: break-all; }
    [data-testid="stCodeBlock"] pre, [data-testid="stCode"] pre {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
</style>
""", unsafe_allow_html=True)


def render_column(strategy: str, data: dict | None, error: str | None, gold: str) -> None:
    st.subheader(strategy)
    st.markdown("---")

    if error:
        st.error(error)
        return

    if data is None:
        st.markdown("_waiting..._")
        return

    # Stats
    lat = data["latency_ms"] / 1000
    em_str = str(int(compute_em(data["answer"], gold))) if gold.strip() else "—"
    st.markdown(f"EM: {em_str}  |  lat: {lat:.1f}s  |  tokens: {data['token_count']}")

    # Answer
    st.markdown("**ANSWER**")
    st.code(strip_citations(data["answer"]), language=None)

    # Citations
    if data["citations"]:
        st.markdown("**CITATIONS**")
        for c in data["citations"]:
            st.markdown(" · ".join(c["chunk_ids"]))

    # Final context
    st.markdown("**FINAL CONTEXT**")
    for item in data["final_context"]:
        if item["type"] == "chunk":
            label = f"{item['id'][:60]}  (score: {item['score']:.3f})"
        else:
            label = f"{item['id'][:60]}  [merged]"
        with st.expander(label):
            if item["type"] == "merged":
                st.caption(f"type: merged")
            st.text(item["text"])
            if item["type"] == "merged":
                st.markdown(f"_sources: {', '.join(item['source_chunk_ids'])}_")

    # Merge plan — skip for top_k and when no ops
    if strategy != "top_k" and data.get("merge_plan") and data["merge_plan"]["operations"]:
        st.markdown("**MERGE PLAN**")
        for op in data["merge_plan"]["operations"]:
            if op["type"] == "symmetric":
                st.markdown(f"`{op['primary_id']} ++ {op['secondary_id']}`")
            else:
                st.markdown(f"`{op['primary_id']} → {op['secondary_id']}`")


# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown("# MERGERAG")
st.markdown(f"collection: `{COLLECTION}`")
st.markdown("---")

query = st.text_input("Query", placeholder="Enter a multi-hop question...")
gold = st.text_input("Gold answer (optional)", placeholder="Leave blank to skip EM scoring")
smart_route = st.toggle("Smart Route", key="smart_route")
run = st.button("Run", disabled=not query.strip())

if run and query.strip():
    st.session_state["results"] = {}
    st.session_state["errors"] = {}
    st.session_state["gold"] = gold
    st.session_state["smart_route_active"] = smart_route

    if smart_route:
        q_type, strategy = classify_question(query)
        st.session_state["smart_route_meta"] = {"type": q_type, "strategy": strategy}
        st.markdown(f"`detected: {q_type} | strategy: {strategy}`")
        with st.spinner(f"running {strategy}..."):
            try:
                data = call_query(query, strategy, COLLECTION, API_BASE, TIMEOUT_S)
                st.session_state["results"][strategy] = data
                st.session_state["errors"][strategy] = None
            except APIError as e:
                st.session_state["results"][strategy] = None
                st.session_state["errors"][strategy] = str(e)
        render_column(
            strategy,
            st.session_state["results"].get(strategy),
            st.session_state["errors"].get(strategy),
            gold,
        )
    else:
        st.session_state["smart_route_meta"] = None
        cols = st.columns(3)
        for i, strategy in enumerate(STRATEGIES):
            with cols[i]:
                with st.spinner(f"running {strategy}..."):
                    try:
                        data = call_query(query, strategy, COLLECTION, API_BASE, TIMEOUT_S)
                        st.session_state["results"][strategy] = data
                        st.session_state["errors"][strategy] = None
                    except APIError as e:
                        st.session_state["results"][strategy] = None
                        st.session_state["errors"][strategy] = str(e)
                render_column(
                    strategy,
                    st.session_state["results"].get(strategy),
                    st.session_state["errors"].get(strategy),
                    gold,
                )

elif "results" in st.session_state:
    if st.session_state.get("smart_route_active"):
        meta = st.session_state.get("smart_route_meta") or {}
        q_type = meta.get("type", "")
        strategy = meta.get("strategy", "")
        if q_type and strategy:
            st.markdown(f"`detected: {q_type} | strategy: {strategy}`")
        render_column(
            strategy,
            st.session_state["results"].get(strategy),
            st.session_state["errors"].get(strategy),
            st.session_state.get("gold", ""),
        )
    else:
        cols = st.columns(3)
        for i, strategy in enumerate(STRATEGIES):
            with cols[i]:
                render_column(
                    strategy,
                    st.session_state["results"].get(strategy),
                    st.session_state["errors"].get(strategy),
                    st.session_state.get("gold", ""),
                )
