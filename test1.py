import duckdb
from chatbot import get_live_schema, generate_sql, is_safe_select

DB_FILE = "velociti.duckdb"

TEST_CASES = [
    {
        "question": "How many shifts had a NoShow status in September 2025?",
        "ground_truth_sql": """
            SELECT COUNT(*) FROM shifts
            WHERE shift_status = 'NoShow'
              AND shift_date >= '2025-09-01' AND shift_date < '2025-10-01'
        """,
    },
    {
        "question": "How many total scheduled shifts are there?",
        "ground_truth_sql": "SELECT COUNT(*) FROM shifts WHERE shift_category = 'Scheduled'",
    },
    {
        "question": "How many total unscheduled shifts are there?",
        "ground_truth_sql": "SELECT COUNT(*) FROM shifts WHERE shift_category = 'Unscheduled'",
    },
    {
        "question": "What is the top facility by number of shifts?",
        "ground_truth_sql": """
            SELECT facility_name, COUNT(*) AS cnt FROM shifts
            GROUP BY facility_name ORDER BY cnt DESC LIMIT 1
        """,
    },
    {
        "question": "How many unique employees are there in the shifts data?",
        "ground_truth_sql": "SELECT COUNT(DISTINCT employee_number) FROM shifts",
    },
    {
        "question": "How many shifts were completed in 2025?",
        "ground_truth_sql": """
            SELECT COUNT(*) FROM shifts
            WHERE shift_status = 'Completed'
              AND shift_date >= '2025-01-01' AND shift_date < '2026-01-01'
        """,
    },
    {
        "question": "Which customer has the most facilities?",
        "ground_truth_sql": """
            SELECT customer_name, COUNT(DISTINCT facility_name) AS facility_count
            FROM shifts GROUP BY customer_name ORDER BY facility_count DESC LIMIT 1
        """,
    },
]


def get_scalar_or_first_row(df):
    if df.shape == (1, 1):
        return df.iloc[0, 0]
    return tuple(df.iloc[0]) if len(df) > 0 else None


def run_tests():
    con = duckdb.connect(DB_FILE, read_only=True)
    schema_text = get_live_schema(con)
    passed, failed = 0, 0

    for case in TEST_CASES:
        question = case["question"]
        truth_value = get_scalar_or_first_row(con.execute(case["ground_truth_sql"]).fetchdf())
        ai_sql = generate_sql(question, schema_text)

        if not is_safe_select(ai_sql):
            print(f"❌ FAIL (unsafe SQL): {question}")
            failed += 1
            continue
        try:
            ai_value = get_scalar_or_first_row(con.execute(ai_sql).fetchdf())
        except Exception as e:
            print(f"❌ FAIL (SQL error): {question}\n   Error: {e}\n   SQL: {ai_sql}")
            failed += 1
            continue

        if str(ai_value) == str(truth_value):
            print(f"✅ PASS: {question}  (= {truth_value})")
            passed += 1
        else:
            print(f"❌ FAIL: {question}\n   Expected: {truth_value}\n   Got: {ai_value}\n   AI SQL: {ai_sql}")
            failed += 1

    print(f"\n--- {passed} passed, {failed} failed out of {len(TEST_CASES)} ---")
    con.close()


if __name__ == "__main__":
    run_tests()