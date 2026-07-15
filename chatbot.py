import os
import re
import time
import json
import hashlib
import duckdb
from datetime import datetime, timezone
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DB_FILE = "velociti.duckdb"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
CACHE_FILE = "sql_cache.json"
LOG_FILE = "qa_log.jsonl"

BUSINESS_NOTES = """
BUSINESS CONTEXT (read carefully before writing SQL):
- shift_category = 'Scheduled' means the shift was planned as part of a recurring schedule.
  'Unscheduled' means it was an ad-hoc/one-off shift.
- shift_status = 'Scheduled' (as a STATUS, not category) means the shift is in the FUTURE and
  has not happened yet. Do not count these as completed/worked shifts unless the user asks
  about future or upcoming shifts specifically.
- shift_status = 'NoShow' means the employee did not show up. This is NOT a completed shift.
- shift_status = 'Completed' means the shift was worked and finished normally.
- shift_status = 'InProgress' means the shift is currently happening right now.
- Unless the user's question implies otherwise, "worked shifts" or "completed shifts" should
  filter to shift_status = 'Completed' only.
- Always filter dates using shift_date (a DATE column), not shift_start_time, unless the user
  asks about a specific time of day.
- The `jobs` table is a reference/roster table (one row per job-employee assignment) — it does
  NOT contain individual shift instances. Use `shifts` for anything about actual worked/missed time.
- To join jobs and shifts for questions like "employees assigned to a job who never worked a shift",
  join on job_number and employee_number.
- There is no such thing as a "null shift" as a status — shifts either have a status like
  Completed/NoShow/Scheduled/InProgress, or a column value itself may be NULL (missing data).
  If a user asks about "null or absent" shifts, interpret "absent" as shift_status = 'NoShow',
  and separately check for actual NULL values in shift_status if that's relevant.
- The `jobs` table is a partial roster/reference snapshot and may NOT include every
  facility a customer has (it only lists current job assignments). For any question
  about customers, facilities, or their relationships/counts, always use the `shifts`
  table as the source of truth, not `jobs` — unless the question specifically asks
  about job assignments, roles, or scheduling setup (recurring pattern, schedule days, etc.).
- "Least" / "fewest" / "lowest" means ORDER BY ... ASC. "Most" / "top" / "highest" means
  ORDER BY ... DESC. Do not mix these up.
- When grouping by job, always exclude NULL/blank job_name groups unless the user asks
  specifically about missing data, since a NULL group is not a real job.
- company_name is ALWAYS 'Velociti Services' (Velociti's own name, since Velociti is the
  company operating this system) -- it is NEVER a client/customer. For any question about a
  customer, client, county, or organization Velociti works for, use customer_name instead.
- For filtering on free-text name columns (customer_name, facility_name, job_name, role_name,
  employee_name), use ILIKE instead of = for case-insensitive matching, since you cannot be
  certain of the exact stored capitalization. Example: WHERE role_name ILIKE '%cleaner%'
  (not = 'cleaner'). Do NOT use ILIKE for the enum-style status/category columns
  (shift_status, shift_category, etc.) -- use their exact values as listed in the schema above.
"""

FEW_SHOT_EXAMPLES = """
EXAMPLES OF CORRECT REASONING (follow this pattern):

Q: "Which job has the least shifts recorded?"
A: The word "least" means smallest count, so sort ascending and take 1.
   SELECT job_name, COUNT(*) AS shift_count
   FROM shifts
   WHERE job_name IS NOT NULL
   GROUP BY job_name
   ORDER BY shift_count ASC
   LIMIT 1

Q: "How many employees never showed up for a shift in 2025?"
A: "Never showed up" = NoShow status. Count DISTINCT employees, not rows, and filter by year
   using shift_date.
   SELECT COUNT(DISTINCT employee_number)
   FROM shifts
   WHERE shift_status = 'NoShow'
     AND shift_date >= '2025-01-01' AND shift_date < '2026-01-01'

Q: "What percentage of shifts were completed vs no-show last month?"
A: This needs a ratio over ALL shifts, not just completed_shifts view, so filter shift_status
   manually with a CASE/FILTER rather than using a single pre-filtered view.
   SELECT
     shift_status,
     COUNT(*) AS cnt,
     ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
   FROM shifts
   WHERE shift_date >= date_trunc('month', CURRENT_DATE - INTERVAL 1 MONTH)
     AND shift_date < date_trunc('month', CURRENT_DATE)
   GROUP BY shift_status

Q: "How many employees work for Maricopa County?"
A: "Maricopa County" is a CUSTOMER, not the company (company_name is always 'Velociti Services').
   Use customer_name with ILIKE for case-insensitive matching.
   SELECT COUNT(DISTINCT employee_number)
   FROM shifts
   WHERE customer_name ILIKE 'maricopa county'
"""


def get_live_schema(con) -> str:
    """Pulls the REAL schema straight from the database so it's always accurate —
    never relies on a hand-typed description that could drift out of sync."""
    lines = []
    for table in ["jobs", "shifts"]:
        lines.append(f"\nTABLE: {table}")
        cols = con.execute(f"DESCRIBE {table}").fetchdf()
        for _, row in cols.iterrows():
            lines.append(f"  - {row['column_name']} ({row['column_type']})")

    for col in ["shift_status", "employee_shift_status", "shift_category", "schedule_type", "shift_type"]:
        vals = con.execute(f"SELECT DISTINCT {col} FROM shifts WHERE {col} IS NOT NULL LIMIT 20").fetchdf()
        vals_list = vals[col].tolist()
        lines.append(f"\nActual distinct values in shifts.{col}: {vals_list}")

    return "\n".join(lines)


def get_valid_columns(con) -> set:
    """Used to sanity-check AI-generated SQL for hallucinated column names before we
    even try to run it — catches a class of errors without burning an extra API call."""
    cols = set()
    for table in ["jobs", "shifts"]:
        df = con.execute(f"DESCRIBE {table}").fetchdf()
        cols.update(c.lower() for c in df["column_name"].tolist())
    return cols


def uses_only_known_columns(sql: str, valid_columns: set) -> tuple:
    """Cheap local check for obviously-hallucinated column names, so we don't
    waste an API call round-trip discovering it from a DB error."""
    lowered_sql = sql.lower()
    sql_without_strings = re.sub(r"'[^']*'", " ", lowered_sql)
    tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", sql_without_strings))
    sql_keywords = {
        "select", "from", "where", "group", "by", "order", "limit", "as", "and", "or",
        "not", "null", "is", "in", "on", "join", "left", "right", "inner", "outer",
        "count", "sum", "avg", "min", "max", "distinct", "case", "when", "then", "else",
        "end", "asc", "desc", "having", "over", "partition", "date_trunc", "interval",
        "current_date", "round", "try_cast", "cast", "date", "timestamp", "boolean",
        "bigint", "double", "varchar", "shifts", "jobs", "completed_shifts", "noshow_shifts",
        "upcoming_shifts", "in_progress_shifts", "shift_hierarchy", "month", "year", "day",
        "extract", "true", "false", "like", "ilike", "between", "filter", "with",
    }
    aliases = set(re.findall(r"\bas\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql_without_strings))

    candidate_cols = tokens - sql_keywords - valid_columns - aliases
    candidate_cols = {c for c in candidate_cols if not c.isdigit()}
    return (len(candidate_cols) == 0), sorted(candidate_cols)


def format_history(history: list, max_turns: int = 3) -> str:
    """Formats recent Q&A turns so follow-up questions ('that', 'it', 'now break
    down by...') can be resolved. Kept short on purpose -- more history = more
    tokens burned per question, which matters given the free-tier quota."""
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = ['CONVERSATION HISTORY (use this to resolve follow-up questions like "that", "it", "also", "now break down by..."):']
    for i, turn in enumerate(recent, 1):
        lines.append(f'{i}. Q: {turn["question"]}')
        lines.append(f'   SQL used: {turn["sql"]}')
        lines.append(f'   Result summary: {turn["result_summary"]}')
    return "\n".join(lines)


def get_column_table_map(con) -> dict:
    """Maps each column name to (table, type) so we know where and how to verify
    literal values against real data. Prefers 'shifts' when a column exists in both."""
    m = {}
    for table in ["jobs", "shifts"]:
        cols = con.execute(f"DESCRIBE {table}").fetchdf()
        for _, row in cols.iterrows():
            col = row["column_name"].lower()
            typ = row["column_type"]
            if col not in m or table == "shifts":
                m[col] = (table, typ)
    return m


def verify_literal_values(sql: str, con, col_map: dict) -> tuple:
    """SELF-HEALING CHECK: extracts every `column = 'value'` / `column ILIKE 'value'`
    filter from the SQL and verifies the value actually exists somewhere in that real
    column -- no guessing. Catches hallucinated values (e.g. a made-up status or
    category) before they ever produce a silently-wrong or empty answer. Returns
    (ok, problems); each problem includes real sample values so the AI can self-correct
    with facts instead of another guess."""
    pattern = re.compile(r"(\w+)\s*(=|ILIKE|LIKE)\s*'([^']*)'", re.IGNORECASE)
    problems = []
    for col, op, value in pattern.findall(sql):
        col_lower = col.lower()
        if col_lower not in col_map:
            continue
        table, typ = col_map[col_lower]
        if "VARCHAR" not in typ.upper():
            continue
        clean_value = value.strip("%")
        if not clean_value:
            continue
        try:
            count = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} ILIKE ?", [f"%{clean_value}%"]
            ).fetchone()[0]
        except Exception:
            continue
        if count == 0:
            words = [w for w in clean_value.split() if len(w) > 2]
            suggestions = []
            if words:
                longest_word = max(words, key=len)
                try:
                    sugg_df = con.execute(
                        f"SELECT DISTINCT {col} FROM {table} WHERE {col} ILIKE ? LIMIT 5",
                        [f"%{longest_word}%"],
                    ).fetchdf()
                    suggestions = sugg_df[col].dropna().tolist()
                except Exception:
                    pass
            if not suggestions:
                try:
                    sugg_df = con.execute(f"SELECT DISTINCT {col} FROM {table} LIMIT 10").fetchdf()
                    suggestions = sugg_df[col].dropna().tolist()
                except Exception:
                    pass
            problems.append({"column": col, "value": value, "real_sample_values": suggestions})
    return (len(problems) == 0), problems


def build_sql_system_prompt(schema_text: str) -> str:
    return f"""You are a SQL expert writing DuckDB SQL queries against a workforce shift database.

Here is the exact, live schema of the database:
{schema_text}

{BUSINESS_NOTES}

{FEW_SHOT_EXAMPLES}

Rules:
- Output ONLY the raw SQL query. No explanation, no markdown fences, no commentary.
- Only write SELECT queries. Never INSERT, UPDATE, DELETE, DROP, ALTER, ATTACH, CREATE, or PRAGMA.
- Use only column names that appear in the schema above — never invent column names.
- If the question is ambiguous, make the most reasonable assumption based on the business context above.
- Prefer aggregating (COUNT, SUM, AVG, GROUP BY) directly in SQL rather than returning raw rows,
  when the question asks for a total, count, or summary.
- Pre-built views are available for convenience: completed_shifts, noshow_shifts, upcoming_shifts,
  in_progress_shifts — each pre-filtered to the correct shift_status. Prefer using these views
  over manually filtering shift_status yourself, UNLESS the question needs to compare across
  multiple statuses at once (e.g. percentages, breakdowns) — in that case filter shift_status
  manually on the base `shifts` table instead.
"""


EXPLAIN_SYSTEM_PROMPT = """You are a helpful analyst. You will be given a user's question,
the SQL query that was run, and the exact result from the database.
Explain the result in clear plain English. Only state facts directly supported by the result
data — never add numbers or claims not present in the result. If the result was truncated,
mention that clearly. Keep it concise."""


def is_safe_select(sql: str) -> bool:
    cleaned = sql.strip().rstrip(";")
    if ";" in cleaned:
        return False
    if not cleaned.lower().lstrip().startswith("select"):
        return False
    forbidden = ["insert", "update", "delete", "drop", "alter", "attach", "create", "pragma", "copy"]
    lowered = cleaned.lower()
    return not any(re.search(rf"\b{word}\b", lowered) for word in forbidden)


def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass


def _cache_key(question: str) -> str:
    normalized = question.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


def log_interaction(question: str, sql, result_summary, status: str) -> None:
    """Appends every Q&A to a permanent audit log -- one JSON record per line.
    Use this to review what's actually been asked, and to mine real questions
    that went wrong into new test_suite.py / test_local_logic.py regression cases."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "sql": sql,
        "result_summary": str(result_summary)[:500],
        "status": status,
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def ask_gemini(prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return response.text.strip()
        except Exception as e:
            error_str = str(e)

            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                if "PerDay" in error_str or "per day" in error_str.lower():
                    raise RuntimeError(
                        "Your Gemini API daily free quota is used up for today. "
                        "It resets at midnight Pacific time (~12:30 PM IST). Options:\n"
                        "  1. Wait until it resets.\n"
                        "  2. Switch models: set GEMINI_MODEL=gemini-2.5-flash-lite in your .env file.\n"
                        "  3. Enable billing on your Google AI Studio project.\n"
                    ) from e
                wait_seconds = 15
                match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", error_str)
                if match:
                    wait_seconds = int(match.group(1)) + 2
                print(f"⏳ Rate limit hit — waiting {wait_seconds}s before retrying (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_seconds)

            elif "UNAVAILABLE" in error_str or "503" in error_str:
                wait_seconds = 8
                print(f"⏳ Gemini servers are busy (high demand) — waiting {wait_seconds}s before retrying (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_seconds)

            else:
                raise
    raise RuntimeError("Gemini retries exhausted (rate limit or server overload) — try again shortly.")


def clean_sql(raw_sql: str) -> str:
    cleaned = raw_sql.strip()
    cleaned = re.sub(r"^```sql\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    return cleaned.strip()


def generate_sql(question: str, schema_text: str, previous_error: str = None,
                  previous_sql: str = None, use_cache: bool = True, history: list = None) -> str:
    cache = _load_cache() if use_cache else {}
    key = _cache_key(question)

    if use_cache and previous_error is None and not history and key in cache:
        print("💾 (using cached SQL for this exact question — no API call needed)")
        return cache[key]

    system_prompt = build_sql_system_prompt(schema_text)
    history_text = format_history(history) if history else ""

    if previous_error:
        prompt = (
            f"{system_prompt}\n\n{history_text}\n\nQuestion: {question}\n\n"
            f"Your previous attempt:\n{previous_sql}\n\n"
            f"That query failed with this error:\n{previous_error}\n\n"
            f"Write a corrected SQL query."
        )
    else:
        prompt = f"{system_prompt}\n\n{history_text}\n\nQuestion: {question}"

    sql = clean_sql(ask_gemini(prompt))

    if use_cache and previous_error is None and not history:
        cache[key] = sql
        _save_cache(cache)

    return sql


def answer_question(question: str, con, schema_text: str, valid_columns: set, col_map: dict, history: list = None) -> None:
    if history is None:
        history = []

    print("\n🧠 Thinking...")

    try:
        sql = generate_sql(question, schema_text, history=history)
    except RuntimeError as e:
        print(f"\n❌ {e}\n")
        log_interaction(question, None, None, f"error: {e}")
        return

    result_df = None
    last_error = None
    unverified_problems = None

    for attempt in range(4):
        print(f"\n📝 SQL (attempt {attempt + 1}):\n{sql}\n")

        if not is_safe_select(sql):
            print("❌ Refused — not a safe read-only SELECT statement.")
            log_interaction(question, sql, None, "error: unsafe SQL refused")
            return

        ok, bad_cols = uses_only_known_columns(sql, valid_columns)
        if not ok:
            unverified_problems = None
            last_error = f"Query references unknown column(s): {bad_cols}. Use only columns from the schema."
            print(f"⚠️  {last_error}\nAsking Gemini to correct it...")
            try:
                sql = generate_sql(question, schema_text, previous_error=last_error, previous_sql=sql,
                                    use_cache=False, history=history)
            except RuntimeError as e:
                print(f"\n❌ {e}\n")
                log_interaction(question, sql, None, f"error: {e}")
                return
            continue

        value_ok, problems = verify_literal_values(sql, con, col_map)
        if not value_ok:
            unverified_problems = problems
            hints = "; ".join(
                f"column '{p['column']}' filtered on '{p['value']}' but that value does not exist -- "
                f"real values in this column include: {p['real_sample_values']}"
                for p in problems
            )
            last_error = f"Filter value(s) not found in the real data: {hints}"
            print(f"⚠️  {last_error}\nAsking Gemini to correct it with real values...")
            try:
                sql = generate_sql(question, schema_text, previous_error=last_error, previous_sql=sql,
                                    use_cache=False, history=history)
            except RuntimeError as e:
                print(f"\n❌ {e}\n")
                log_interaction(question, sql, None, f"error: {e}")
                return
            continue

        try:
            result_df = con.execute(sql).fetchdf()
            last_error = None
            unverified_problems = None
            break
        except Exception as e:
            unverified_problems = None
            last_error = str(e)
            print(f"⚠️  SQL error: {last_error}\nAsking Gemini to correct it...")
            try:
                sql = generate_sql(question, schema_text, previous_error=last_error, previous_sql=sql,
                                    use_cache=False, history=history)
            except RuntimeError as e2:
                print(f"\n❌ {e2}\n")
                log_interaction(question, sql, None, f"error: {e2}")
                return

    if unverified_problems:
        print("\n💬 I couldn't verify the filter value(s) in your question against the real data, so I won't guess:")
        for p in unverified_problems:
            print(f"   - You mentioned something matching '{p['value']}', but that doesn't exist in {p['column']}.")
            print(f"     Real values found in that column: {p['real_sample_values']}")
        print()
        log_interaction(question, sql, unverified_problems, "unverified: could not confirm filter values")
        return

    if last_error is not None:
        print(f"❌ Still failing after retries: {last_error}")
        log_interaction(question, sql, None, f"error: {last_error}")
        return

    if len(result_df) == 0:
        print("The query ran successfully but returned no rows — nothing matched your criteria.")
        history.append({"question": question, "sql": sql, "result_summary": "no rows returned"})
        log_interaction(question, sql, "no rows returned", "success")
        return

    result_preview = result_df.head(50).to_string(index=False)
    row_count_note = f"(showing first 50 of {len(result_df)} rows)" if len(result_df) > 50 else ""

    if result_df.shape == (1, 1):
        value = result_df.iloc[0, 0]
        if value is None or (isinstance(value, float) and value != value):
            print("\n💬 Answer: No matching data was found for this question.\n")
            history.append({"question": question, "sql": sql, "result_summary": "no matching data"})
            log_interaction(question, sql, "no matching data", "success")
        else:
            print(f"\n💬 Answer: {value}\n")
            history.append({"question": question, "sql": sql, "result_summary": str(value)})
            log_interaction(question, sql, value, "success")
    else:
        try:
            explanation = ask_gemini(
                f"{EXPLAIN_SYSTEM_PROMPT}\n\nQuestion: {question}\n\nSQL used: {sql}\n\n"
                f"Result {row_count_note}:\n{result_preview}"
            )
            print(f"\n💬 Answer:\n{explanation}\n")
            history.append({"question": question, "sql": sql, "result_summary": explanation[:300]})
            log_interaction(question, sql, explanation, "success")
        except RuntimeError as e:
            print(f"\n⚠️ Couldn't generate a written explanation ({e})\n")
            print(f"💬 Raw result:\n{result_preview}\n")
            history.append({"question": question, "sql": sql, "result_summary": result_preview[:300]})
            log_interaction(question, sql, result_preview, f"partial: {e}")

    print("-" * 60)


def main():
    con = duckdb.connect(DB_FILE, read_only=True)
    print(f"Using model: {MODEL_NAME}")
    print("Loading live database schema...")
    schema_text = get_live_schema(con)
    valid_columns = get_valid_columns(con)
    col_map = get_column_table_map(con)
    conversation_history = []
    print("Velociti Shift Analysis Chatbot — type 'exit' to quit.\n")
    while True:
        question = input("Ask a question about your shift data: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer_question(question, con, schema_text, valid_columns, col_map, history=conversation_history)
    con.close()


if __name__ == "__main__":
    main()