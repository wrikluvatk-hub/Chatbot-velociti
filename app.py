import os
import duckdb
import pandas as pd
import streamlit as st

from chatbot import (
    get_live_schema,
    get_valid_columns,
    get_column_table_map,
    generate_sql,
    is_safe_select,
    uses_only_known_columns,
    verify_literal_values,
    ask_gemini,
    EXPLAIN_SYSTEM_PROMPT,
    DB_FILE,
)

st.set_page_config(page_title="Velociti Shift Intelligence", page_icon="📊", layout="wide")

# ============================================================
# Styling -- premium, modern, LLM-chat-like
# ============================================================

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px;}
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        border-radius: 18px;
        padding: 1.5rem 1.75rem;
        color: white;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
    }
    .hero-card h1 {margin-bottom: 0.25rem; font-size: 1.6rem;}
    .hero-card p {margin-bottom: 0; color: #dbe4f5; font-size: 0.95rem;}
    .info-card {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        background: #f8fafc;
        margin-bottom: 0.75rem;
        color: #0f172a;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        height: 100%;
    }
    .info-card strong {color: #0f172a; font-size: 0.95rem;}
    .info-card .sub {color: #475569; font-size: 0.85rem; line-height: 1.5;}
    .stChatMessage {border-radius: 14px;}
     div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem 1rem;
    }
    div[data-testid="stMetric"] label {
        color: #475569 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0f172a !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: #0f172a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Startup checks -- production-friendly error handling
# ============================================================

if not os.getenv("GEMINI_API_KEY"):
    st.error("🔑 GEMINI_API_KEY is not configured.")
    st.info(
        "Local development: add `GEMINI_API_KEY=your_key_here` to a `.env` file in this folder.\n\n"
        "Deployed on Streamlit Community Cloud: add it under **App settings → Secrets** instead "
        "(never commit real keys to a repo)."
    )
    st.stop()

if not os.path.exists(DB_FILE):
    st.error(f"📦 Database file '{DB_FILE}' was not found.")
    st.info("Run `python build_database.py` first so the app has data to query.")
    st.stop()


@st.cache_resource
def get_connection():
    con = duckdb.connect(DB_FILE, read_only=True)
    schema_text = get_live_schema(con)
    valid_columns = get_valid_columns(con)
    col_map = get_column_table_map(con)
    return con, schema_text, valid_columns, col_map


try:
    con, schema_text, valid_columns, col_map = get_connection()
except Exception as e:
    st.error(f"Could not open the database connection: {e}")
    st.stop()


@st.cache_data(ttl=600)
def get_date_bounds():
    row = con.execute("SELECT MIN(shift_date), MAX(shift_date) FROM shifts").fetchone()
    return row[0], row[1]


@st.cache_data(ttl=600)
def get_filter_options():
    customers = con.execute(
        "SELECT DISTINCT customer_name FROM shifts WHERE customer_name IS NOT NULL ORDER BY 1"
    ).fetchdf()["customer_name"].tolist()
    facilities = con.execute(
        "SELECT DISTINCT facility_name FROM shifts WHERE facility_name IS NOT NULL ORDER BY 1"
    ).fetchdf()["facility_name"].tolist()
    return customers, facilities


min_date, max_date = get_date_bounds()
all_customers, all_facilities = get_filter_options()


def build_where(date_range, categories, customer, facility):
    clauses, params = ["1=1"], []
    if date_range and len(date_range) == 2:
        clauses.append("shift_date >= ? AND shift_date <= ?")
        params += [date_range[0], date_range[1]]
    if categories:
        placeholders = ",".join(["?"] * len(categories))
        clauses.append(f"shift_category IN ({placeholders})")
        params += categories
    if customer and customer != "All":
        clauses.append("customer_name = ?")
        params.append(customer)
    if facility and facility != "All":
        clauses.append("facility_name = ?")
        params.append(facility)
    return " AND ".join(clauses), params


def filters_as_text(date_range, categories, customer, facility):
    parts = []
    if date_range and len(date_range) == 2:
        parts.append(f"date range {date_range[0]} to {date_range[1]}")
    if categories and len(categories) < 2:
        parts.append(f"shift category = {categories[0]}")
    if customer and customer != "All":
        parts.append(f"customer = {customer}")
    if facility and facility != "All":
        parts.append(f"facility = {facility}")
    return "; ".join(parts) if parts else "no filters applied (all data)"


# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("### 📊 Velociti Shift Intelligence")
    st.caption("AI-powered analysis of scheduled, unscheduled & self-performed shifts")
    st.divider()

    mode = st.radio(
        "Mode", ["💬 Ask Anything", "📈 Quick Analytics", "🩺 Data Health"], label_visibility="collapsed"
    )

    st.divider()
    st.markdown("#### Filters")
    st.caption("Scopes Quick Analytics, and is given to chat as context")
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    category_filter = st.multiselect("Shift category", ["Scheduled", "Unscheduled"], default=["Scheduled", "Unscheduled"])
    customer_filter = st.selectbox("Customer", ["All"] + all_customers)
    facility_filter = st.selectbox("Facility", ["All"] + all_facilities)

    st.divider()
    response_style = st.radio(
        "Answer style",
        ["Just the number", "Full analysis (answer + context + insight)"],
        index=1,
        help="Full analysis uses one extra API call per question.",
    )
    concise_mode = response_style == "Just the number"

    st.divider()
    st.markdown("#### Session")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()
    st.caption(f"💾 {len(st.session_state.history)} exchange(s) in memory")

    with st.expander("ℹ️ About this app"):
        st.caption(
            "Questions are translated into SQL by an LLM, run against your real DuckDB database, "
            "and every filter value is verified against the actual data before an answer is shown -- "
            "so results are grounded in fact, not guesses."
        )


# ============================================================
# Quick Analytics -- pure SQL, zero AI, zero hallucination risk
# ============================================================

def render_quick_analytics():
    st.markdown(
        f"""<div class="hero-card"><h1>📈 Quick Analytics</h1>
        <p>Deterministic SQL dashboard -- no AI involved, so every number here is exact.</p></div>""",
        unsafe_allow_html=True,
    )
    st.caption(f"Scope: {filters_as_text(date_range, category_filter, customer_filter, facility_filter)}")

    where, params = build_where(date_range, category_filter, customer_filter, facility_filter)

    kpi_df = con.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN shift_status='Completed' THEN 1 ELSE 0 END) AS completed,
            SUM(CASE WHEN shift_status='NoShow' THEN 1 ELSE 0 END) AS noshow,
            SUM(CASE WHEN shift_status='Scheduled' THEN 1 ELSE 0 END) AS upcoming
        FROM shifts WHERE {where}
        """,
        params,
    ).fetchdf()

    total = int(kpi_df["total"][0]) if kpi_df["total"][0] else 0
    completed = int(kpi_df["completed"][0] or 0)
    noshow = int(kpi_df["noshow"][0] or 0)
    upcoming = int(kpi_df["upcoming"][0] or 0)
    noshow_rate = round(100 * noshow / total, 1) if total else 0
    completion_rate = round(100 * completed / total, 1) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total shifts", f"{total:,}")
    c2.metric("Completion rate", f"{completion_rate}%")
    c3.metric("No-show rate", f"{noshow_rate}%")
    c4.metric("Upcoming (future)", f"{upcoming:,}")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Shifts over time**")
        trend = con.execute(
            f"SELECT date_trunc('month', shift_date) AS month, COUNT(*) AS shifts FROM shifts WHERE {where} GROUP BY month ORDER BY month",
            params,
        ).fetchdf()
        if not trend.empty:
            st.line_chart(trend.set_index("month")["shifts"])
        else:
            st.info("No data in this range.")

    with col_b:
        st.markdown("**Status breakdown**")
        status_df = con.execute(
            f"SELECT shift_status, COUNT(*) AS cnt FROM shifts WHERE {where} GROUP BY shift_status ORDER BY cnt DESC",
            params,
        ).fetchdf()
        if not status_df.empty:
            st.bar_chart(status_df.set_index("shift_status")["cnt"])
        else:
            st.info("No data in this range.")

    st.divider()
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Top 5 facilities by shift volume**")
        top_fac = con.execute(
            f"SELECT facility_name, COUNT(*) AS shifts FROM shifts WHERE {where} GROUP BY facility_name ORDER BY shifts DESC LIMIT 5",
            params,
        ).fetchdf()
        st.dataframe(top_fac, hide_index=True, use_container_width=True)

    with col_d:
        st.markdown("**Top 5 customers by shift volume**")
        top_cust = con.execute(
            f"SELECT customer_name, COUNT(*) AS shifts FROM shifts WHERE {where} GROUP BY customer_name ORDER BY shifts DESC LIMIT 5",
            params,
        ).fetchdf()
        st.dataframe(top_cust, hide_index=True, use_container_width=True)


# ============================================================
# Data Health
# ============================================================

def render_data_health():
    st.markdown(
        """<div class="hero-card"><h1>🩺 Data Health</h1>
        <p>Structural checks to catch data issues before they affect analysis.</p></div>""",
        unsafe_allow_html=True,
    )
    min_d, max_d = get_date_bounds()
    st.info(f"Data covers **{min_d}** to **{max_d}**.")

    orphan_count = con.execute(
        """SELECT COUNT(DISTINCT s.employee_number) FROM shifts s
           LEFT JOIN jobs j ON s.employee_number = j.employee_number
           WHERE j.employee_number IS NULL"""
    ).fetchone()[0]
    null_dates = con.execute("SELECT COUNT(*) FROM shifts WHERE shift_date IS NULL").fetchone()[0]
    null_customer = con.execute("SELECT COUNT(*) FROM shifts WHERE customer_name IS NULL").fetchone()[0]
    null_facility = con.execute("SELECT COUNT(*) FROM shifts WHERE facility_name IS NULL").fetchone()[0]
    total_shifts = con.execute("SELECT COUNT(*) FROM shifts").fetchone()[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Employees in shifts but not in roster", f"{orphan_count:,}")
    c2.metric("Shifts with missing date", f"{null_dates:,}")
    c3.metric("Total shift records", f"{total_shifts:,}")

    st.divider()
    st.markdown("**Missing-value check on key columns**")
    st.dataframe(
        pd.DataFrame({
            "column": ["shift_date", "customer_name", "facility_name"],
            "missing_count": [null_dates, null_customer, null_facility],
            "missing_pct": [
                round(100 * null_dates / total_shifts, 2) if total_shifts else 0,
                round(100 * null_customer / total_shifts, 2) if total_shifts else 0,
                round(100 * null_facility / total_shifts, 2) if total_shifts else 0,
            ],
        }),
        hide_index=True, use_container_width=True,
    )

    st.divider()
    st.markdown("**Status code reference**")
    st.dataframe(
        con.execute("SELECT shift_status, COUNT(*) AS cnt FROM shifts GROUP BY shift_status ORDER BY cnt DESC").fetchdf(),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "⚠️ Status codes 'I' and 'B' have unconfirmed business meaning as of this build -- "
        "treat any analysis involving them with caution until clarified."
    )


# ============================================================
# Ask Anything -- self-healing AI pipeline + richer responses
# ============================================================

def run_pipeline(question: str, filter_context: str):
    augmented_question = f"{question}\n\n[Active UI filters -- apply unless the question overrides them: {filter_context}]"

    try:
        sql = generate_sql(augmented_question, schema_text, history=st.session_state.history)
    except RuntimeError as e:
        return f"❌ {e}", None, None

    result_df = None
    last_error = None
    unverified_problems = None

    for _ in range(4):
        if not is_safe_select(sql):
            return "❌ Refused — not a safe read-only SELECT statement.", sql, None

        ok, bad_cols = uses_only_known_columns(sql, valid_columns)
        if not ok:
            unverified_problems = None
            last_error = f"Query references unknown column(s): {bad_cols}. Use only columns from the schema."
            try:
                sql = generate_sql(augmented_question, schema_text, previous_error=last_error,
                                    previous_sql=sql, use_cache=False, history=st.session_state.history)
            except RuntimeError as e:
                return f"❌ {e}", sql, None
            continue

        value_ok, problems = verify_literal_values(sql, con, col_map)
        if not value_ok:
            unverified_problems = problems
            hints = "; ".join(
                f"column '{p['column']}' filtered on '{p['value']}' but that value does not exist -- "
                f"real values include: {p['real_sample_values']}"
                for p in problems
            )
            last_error = f"Filter value(s) not found in the real data: {hints}"
            try:
                sql = generate_sql(augmented_question, schema_text, previous_error=last_error,
                                    previous_sql=sql, use_cache=False, history=st.session_state.history)
            except RuntimeError as e:
                return f"❌ {e}", sql, None
            continue

        try:
            result_df = con.execute(sql).fetchdf()
            last_error = None
            unverified_problems = None
            break
        except Exception as e:
            unverified_problems = None
            last_error = str(e)
            try:
                sql = generate_sql(augmented_question, schema_text, previous_error=last_error,
                                    previous_sql=sql, use_cache=False, history=st.session_state.history)
            except RuntimeError as e2:
                return f"❌ {e2}", sql, None

    if unverified_problems:
        lines = ["I couldn't verify the filter value(s) in your question against the real data, so I won't guess:"]
        for p in unverified_problems:
            lines.append(f"- Something matching **'{p['value']}'** doesn't exist in `{p['column']}`. Real values include: {p['real_sample_values']}")
        st.session_state.history.append({"question": question, "sql": sql, "result_summary": "unverified"})
        return "\n".join(lines), sql, None

    if last_error is not None:
        return f"❌ Still failing after retries: {last_error}", sql, None

    if len(result_df) == 0:
        st.session_state.history.append({"question": question, "sql": sql, "result_summary": "no rows returned"})
        return "The query ran successfully but returned no rows — nothing matched your criteria.", sql, None

    result_preview = result_df.head(50).to_string(index=False)
    row_count_note = f"(showing first 50 of {len(result_df)} rows)" if len(result_df) > 50 else ""

    is_scalar = result_df.shape == (1, 1)
    scalar_value = None
    if is_scalar:
        v = result_df.iloc[0, 0]
        if v is None or (isinstance(v, float) and v != v):
            st.session_state.history.append({"question": question, "sql": sql, "result_summary": "no matching data"})
            return "No matching data was found for this question.", sql, None
        scalar_value = v

    if concise_mode:
        st.session_state.history.append({"question": question, "sql": sql, "result_summary": str(scalar_value if is_scalar else result_preview[:200])})
        return (f"**{scalar_value}**" if is_scalar else f"```\n{result_preview}\n```"), sql, scalar_value

    try:
        narrative_prompt = f"""{EXPLAIN_SYSTEM_PROMPT}

Respond like a sharp, friendly data analyst colleague giving a business stakeholder a clear
answer. Structure your response in three short parts, written as flowing prose (not headers,
not bullet lists):
1. The direct answer, with the key number(s) in **bold**.
2. One sentence of context on what this actually means in plain terms.
3. One sentence of useful insight or takeaway if one is genuinely supported by the data
   (a notable share, a comparison, a trend) -- omit this part entirely if there isn't a
   meaningful one; never invent a takeaway just to fill space.
Keep the whole thing to 3-4 sentences total.

Question: {question}

SQL used: {sql}

Result {row_count_note}:
{result_preview}"""
        explanation = ask_gemini(narrative_prompt)
        st.session_state.history.append({"question": question, "sql": sql, "result_summary": explanation[:300]})
        return explanation, sql, scalar_value
    except RuntimeError as e:
        st.session_state.history.append({"question": question, "sql": sql, "result_summary": result_preview[:300]})
        return f"⚠️ Couldn't generate a written explanation ({e})\n\n```\n{result_preview}\n```", sql, scalar_value


def render_ask_anything():
    st.markdown(
        """<div class="hero-card"><h1>Ask your workforce data anything</h1>
        <p>Natural language in, verified SQL-backed answers out.</p></div>""",
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        cols = st.columns(3)
        cards = [
            ("💡 Try asking", "How many completed shifts were there last month?"),
            ("📊 Or compare", "Which employee had the most no-shows this year?"),
            ("🏢 Or explore", "Show me shifts by facility, top 5."),
        ]
        for col, (title, example) in zip(cols, cards):
            with col:
                st.markdown(
                    f"""<div class="info-card"><strong>{title}</strong>
                    <div class="sub">"{example}"</div></div>""",
                    unsafe_allow_html=True,
                )

    st.caption(f"Scope: {filters_as_text(date_range, category_filter, customer_filter, facility_filter)}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("metric") is not None:
                st.metric("Answer", msg["metric"])
            if "sql" in msg:
                with st.expander("Show SQL used"):
                    st.code(msg["sql"], language="sql")

    question = st.chat_input("Ask a question about your shift data...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        filter_context = filters_as_text(date_range, category_filter, customer_filter, facility_filter)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                answer, sql_used, scalar_value = run_pipeline(question, filter_context)
            st.markdown(answer)
            if scalar_value is not None and not concise_mode:
                st.metric("Answer", scalar_value)
            if sql_used:
                with st.expander("Show SQL used"):
                    st.code(sql_used, language="sql")

        entry = {"role": "assistant", "content": answer, "metric": scalar_value if not concise_mode else None}
        if sql_used:
            entry["sql"] = sql_used
        st.session_state.messages.append(entry)


# ============================================================
# Route
# ============================================================

if mode == "💬 Ask Anything":
    render_ask_anything()
elif mode == "📈 Quick Analytics":
    render_quick_analytics()
else:
    render_data_health()