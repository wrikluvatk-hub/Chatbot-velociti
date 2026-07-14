import duckdb
import os

DB_FILE = "velociti.duckdb"
JOBS_CSV = "VELOCITI_V_SelPerformed_Jobs.csv"
SCHEDULED_CSV = "VELOCITI_V_SelPerformed_Scheduled_Shift_Details.csv"
UNSCHEDULED_CSV = "VELOCITI_V_SelPerformed_Unscheduled_Shift_Details.csv"

for f in [JOBS_CSV, SCHEDULED_CSV, UNSCHEDULED_CSV]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Could not find {f} — make sure it's in this same folder.")

if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

con = duckdb.connect(DB_FILE)

print("Loading jobs (master reference) table...")
con.execute(f"""
    CREATE TABLE jobs AS
    SELECT
        TRY_CAST(jobNumber AS BIGINT)          AS job_number,
        jobName                                AS job_name,
        jobType                                AS job_type,
        TRY_CAST(jobStartDate AS DATE)          AS job_start_date,
        jobStatus                              AS job_status,
        customerName                            AS customer_name,
        facilityName                            AS facility_name,
        vendorName                              AS company_name,
        employeeName                            AS employee_name,
        TRY_CAST(employeeNumber AS BIGINT)      AS employee_number,
        roleName                                AS role_name,
        TRY_CAST(isSupervisor AS BOOLEAN)       AS is_supervisor,
        workOrderId                              AS work_order_id,
        woScheduleMasterId                       AS wo_schedule_master_id,
        userId                                   AS user_id,
        scheduleType                            AS schedule_type,
        shiftType                               AS shift_type,
        scheduleName                            AS schedule_name,
        scheduleDays                            AS schedule_days,
        TRY_CAST(scheduleStartDate AS DATE)     AS schedule_start_date,
        TRY_CAST(scheduleEndDate AS DATE)       AS schedule_end_date,
        scheduleStartTime                       AS schedule_start_time,
        scheduleEndTime                         AS schedule_end_time
    FROM read_csv_auto('{JOBS_CSV}', ALL_VARCHAR=TRUE)
""")

def load_shift_file(csv_path: str, category: str):
    print(f"Loading {category} shifts from {csv_path} ...")
    con.execute("""
        CREATE TABLE IF NOT EXISTS shifts (
            job_number BIGINT, job_name VARCHAR, customer_name VARCHAR,
            facility_name VARCHAR, company_name VARCHAR, employee_name VARCHAR,
            employee_number BIGINT, role_name VARCHAR, is_supervisor BOOLEAN,
            work_order_id VARCHAR, wo_schedule_master_id VARCHAR, user_id VARCHAR,
            shift_category VARCHAR, schedule_type VARCHAR, shift_type VARCHAR,
            shift_date DATE, shift_start_time TIMESTAMP, shift_end_time TIMESTAMP,
            shift_status VARCHAR, employee_shift_status VARCHAR,
            in_time_local TIMESTAMP, out_time_local TIMESTAMP,
            total_work_minutes DOUBLE, total_minutes_on_site DOUBLE,
            hours_type_description VARCHAR,
            break_start_time TIMESTAMP,
            break_end_time TIMESTAMP
        )
    """)
    con.execute(f"""
        INSERT INTO shifts
        SELECT
            TRY_CAST(jobNumber AS BIGINT), jobName, customerName, facilityName, vendorName,
            employeeName, TRY_CAST(employeeNumber AS BIGINT), roleName,
            TRY_CAST(isSupervisor AS BOOLEAN), workOrderId, woScheduleMasterId, userId,
            '{category}' AS shift_category, scheduleType, shiftType,
            TRY_CAST(shiftDate AS DATE), TRY_CAST(shiftStartTime AS TIMESTAMP),
            TRY_CAST(shiftEndTime AS TIMESTAMP),TRY_CAST(breakStartTime AS TIMESTAMP),
            TRY_CAST(breakEndTime AS TIMESTAMP), shiftStatus, employeeShiftStatus,
            TRY_CAST(inTimeLocal AS TIMESTAMP), TRY_CAST(outTimeLocal AS TIMESTAMP),
            TRY_CAST(totalWorkMinutes AS DOUBLE), TRY_CAST(totalMinutesOnSite AS DOUBLE),
            hoursTypeDescription
        FROM read_csv_auto('{csv_path}', ALL_VARCHAR=TRUE)
    """)

con.execute("DROP TABLE IF EXISTS shifts")
load_shift_file(SCHEDULED_CSV, "Scheduled")
load_shift_file(UNSCHEDULED_CSV, "Unscheduled")

print("Building hierarchy view...")
con.execute("""
    CREATE OR REPLACE VIEW shift_hierarchy AS
    SELECT company_name, customer_name, facility_name, job_number, job_name,
        employee_number, employee_name, role_name, work_order_id, wo_schedule_master_id,
        shift_category, shift_date, shift_start_time, shift_end_time,
        shift_status, employee_shift_status, total_work_minutes, total_minutes_on_site
    FROM shifts
""")
print("Building semantic views (pre-defined business logic, so the AI doesn't have to guess each time)...")
con.execute("CREATE OR REPLACE VIEW completed_shifts AS SELECT * FROM shifts WHERE shift_status = 'Completed'")
con.execute("CREATE OR REPLACE VIEW noshow_shifts AS SELECT * FROM shifts WHERE shift_status = 'NoShow'")
con.execute("CREATE OR REPLACE VIEW upcoming_shifts AS SELECT * FROM shifts WHERE shift_status = 'Scheduled'")
con.execute("CREATE OR REPLACE VIEW in_progress_shifts AS SELECT * FROM shifts WHERE shift_status = 'InProgress'")
n_jobs = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
n_shifts = con.execute("SELECT COUNT(*) FROM shifts").fetchone()[0]
print(f"\n--- Build complete --- jobs: {n_jobs}, shifts: {n_shifts}")
con.close()