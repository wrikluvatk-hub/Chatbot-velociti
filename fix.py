from chatbot import uses_only_known_columns, get_valid_columns
import duckdb

con = duckdb.connect('velociti.duckdb', read_only=True)
cols = get_valid_columns(con)

test_sql = "SELECT COUNT(*) FROM shifts WHERE shift_status != 'Completed'"
result = uses_only_known_columns(test_sql, cols)

print(result)