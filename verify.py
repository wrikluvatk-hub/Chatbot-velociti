import duckdb

con = duckdb.connect("velociti.duckdb", read_only=True)

result = con.execute("""
    SELECT COUNT(*) AS noshow_count
    FROM shifts
    WHERE shift_status = 'NoShow'
      AND shift_date >= '2025-09-01'
      AND shift_date < '2025-10-01'
""").fetchone()

print("NoShow shifts in September 2025:", result[0])

# Bonus: break it down by scheduled vs unscheduled, so you can sanity-check the split too
breakdown = con.execute("""
    SELECT shift_category, COUNT(*) AS cnt
    FROM shifts
    WHERE shift_status = 'NoShow'
      AND shift_date >= '2025-09-01'
      AND shift_date < '2025-10-01'
    GROUP BY shift_category
""").fetchdf()
print(breakdown)

con.close()