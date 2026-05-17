from __future__ import annotations
import streamlit as st
from app.api_client import call_query, APIError
from app.helpers import compute_em, strip_citations

API_BASE = "http://localhost:8000"
COLLECTION = "hotpot_dev_500"
TIMEOUT_S = 120
STRATEGIES: list[str] = ["top_k", "symmetric", "asymmetric"]

DEFAULTS: dict = {
    "smart_route": False,
    "retriever_choice": "chroma",
    "top_n": 10,
    "top_k": 5,
    "strong_k": 5,
    "token_budget": 2048,
    "asymmetric_max_ops": 1,
}


def _apply_defaults() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

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

# Seed defaults on first render
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.pop("_reset", False):
    _apply_defaults()

query = st.text_input("Query", placeholder="Enter a multi-hop question...")
gold = st.text_input("Gold answer (optional)", placeholder="Leave blank to skip EM scoring")

col_a, col_b = st.columns(2)
with col_a:
    smart_route = st.toggle("Smart Route", key="smart_route")
with col_b:
    retriever_choice = st.radio(
        "Retriever",
        options=["chroma", "hybrid"],
        horizontal=True,
        key="retriever_choice",
    )

with st.expander("Pipeline params", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("top_n (retrieval pool)", min_value=1, max_value=100, step=1, key="top_n")
        st.number_input("token_budget", min_value=128, max_value=8192, step=128, key="token_budget")
    with c2:
        st.number_input("top_k (final context)", min_value=1, max_value=50, step=1, key="top_k")
        st.number_input("asymmetric_max_ops", min_value=0, max_value=10, step=1, key="asymmetric_max_ops")
    with c3:
        st.number_input("strong_k", min_value=1, max_value=50, step=1, key="strong_k")

run_col, reset_col = st.columns([1, 1])
with run_col:
    run = st.button("Run", disabled=not query.strip(), use_container_width=True)
with reset_col:
    if st.button("Reset to defaults", use_container_width=True):
        st.session_state["_reset"] = True
        st.rerun()

top_n = st.session_state["top_n"]
top_k = st.session_state["top_k"]
strong_k = st.session_state["strong_k"]
token_budget = st.session_state["token_budget"]
asymmetric_max_ops = st.session_state["asymmetric_max_ops"]

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
                data = call_query(
                    query, strategy, COLLECTION, API_BASE, TIMEOUT_S,
                    retriever=retriever_choice,
                    top_n=top_n, top_k=top_k, strong_k=strong_k,
                    token_budget=token_budget,
                    asymmetric_max_ops=asymmetric_max_ops,
                )
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
                        data = call_query(
                    query, strategy, COLLECTION, API_BASE, TIMEOUT_S,
                    retriever=retriever_choice,
                    top_n=top_n, top_k=top_k, strong_k=strong_k,
                    token_budget=token_budget,
                    asymmetric_max_ops=asymmetric_max_ops,
                )
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
