import duckdb

con = duckdb.connect("velociti.duckdb", read_only=True)

print("Date range of all shifts:")
print(con.execute("SELECT MIN(shift_date), MAX(shift_date) FROM shifts").fetchdf())

print("\nShift status breakdown:")
print(con.execute("""
    SELECT shift_category, shift_status, COUNT(*) as cnt
    FROM shifts
    GROUP BY shift_category, shift_status
    ORDER BY shift_category, cnt DESC
""").fetchdf())

print("\nTop 5 facilities by total shifts:")
print(con.execute("""
    SELECT facility_name, COUNT(*) as shift_count
    FROM shifts
    GROUP BY facility_name
    ORDER BY shift_count DESC
    LIMIT 5
""").fetchdf())

con.close()