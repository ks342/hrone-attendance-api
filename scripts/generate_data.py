"""Fill MONGO_DB with a ~100k-record dataset that looks like the grader's: overnight shifts, weekend
records, missing weekdays, ABSENT/LEAVE days, half days, mid-month and future joiners, employees with
no logs at all, and legacy records without half_day/history. Derived fields are computed with the
app's own helpers so they are consistent with R1-R5.

    python scripts/generate_data.py            # drops and refills employees + attendance_logs
"""
import random
import sys
import time as clock
from datetime import date, datetime, timedelta
from pathlib import Path

from pymongo.errors import BulkWriteError, PyMongoError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import main  # noqa: E402

random.seed(7)
DEPARTMENTS = ["Engineering", "Sales", "Support", "Finance", "HR", "Operations"]
SHIFTS = [("09:30", "18:30"), ("09:00", "18:00"), ("10:00", "19:00"), ("22:00", "06:00"), ("14:00", "22:00")]
FIRST_DAY, LAST_DAY = date(2026, 1, 1), date(2026, 9, 30)
N_EMPLOYEES = 560


def make_employees():
    emps = []
    for i in range(1, N_EMPLOYEES + 1):
        start, end = random.choice(SHIFTS)
        roll = random.random()
        if roll < 0.08:  # joins during the period (some mid-month)
            joined = FIRST_DAY + timedelta(days=random.randint(30, 250))
        elif roll < 0.10:  # joins after the data ends: never in any headcount yet
            joined = LAST_DAY + timedelta(days=random.randint(1, 60))
        else:
            joined = date(2025, random.randint(1, 12), random.randint(1, 28))
        emps.append({
            "emp_code": f"EMP{i:04d}", "name": f"Employee {i}", "email": f"emp{i}@example.com",
            "department": random.choice(DEPARTMENTS), "shift_start": start, "shift_end": end,
            "joined_on": joined.isoformat(), "created_at": main.now_utc(),
        })
    return emps


def make_record(emp, day):
    roll = random.random()
    if roll < 0.04:
        status = random.choice(["ABSENT", "LEAVE"])
        doc = {"emp_code": emp["emp_code"], "date": day.isoformat(), "status": status, "punch_in": None,
               "punch_out": None, "work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False,
               "history": []}
    else:
        status = random.choices(["PRESENT", "WFH", "ON_DUTY"], [0.8, 0.15, 0.05])[0]
        start, _ = main.shift_bounds(day, emp)
        punch_in = start + timedelta(seconds=random.randint(-1800, 3600 if random.random() < 0.2 else 600))
        punch_out = None
        if random.random() > 0.01:  # a few open records
            hours = random.uniform(3, 5) if random.random() < 0.05 else random.uniform(8, 10)
            punch_out = punch_in + timedelta(seconds=int(hours * 3600))
        doc = {"emp_code": emp["emp_code"], "date": day.isoformat(), "status": status, "punch_in": punch_in,
               "punch_out": punch_out, **main.derived_fields(status, punch_in, punch_out, emp, day), "history": []}
    if random.random() < 0.05:  # legacy shape
        doc.pop("half_day")
        doc.pop("history")
    return doc


def insert(batch):
    """insert_many that survives flaky connections; a retried batch may hit rows that already
    made it, which the unique (emp_code, date) index rejects and we ignore."""
    for attempt in range(8):
        try:
            main.logs.insert_many(batch, ordered=False)
            return
        except BulkWriteError as exc:
            if all(e["code"] == 11000 for e in exc.details["writeErrors"]):
                return
            raise
        except PyMongoError:
            clock.sleep(2 * (attempt + 1))
    raise RuntimeError("giving up on a batch after 8 attempts")


def main_():
    emps = make_employees()
    main.employees.drop()
    main.logs.drop()
    main.ensure_indexes()  # before loading, so a retried batch cannot duplicate rows
    main.employees.insert_many([dict(e) for e in emps])
    no_logs = set(random.sample([e["emp_code"] for e in emps], 15))
    batch, total, t0 = [], 0, clock.time()
    for emp in emps:
        if emp["emp_code"] in no_logs:
            continue
        day = max(FIRST_DAY, date.fromisoformat(emp["joined_on"]))
        while day <= LAST_DAY:
            weekend = day.weekday() >= 5
            if (not weekend and random.random() < 0.95) or (weekend and random.random() < 0.04):
                batch.append(make_record(emp, day))
            day += timedelta(days=1)
            if len(batch) >= 2000:
                insert(batch)
                total += len(batch)
                batch = []
    if batch:
        insert(batch)
        total += len(batch)
    main.ensure_indexes()
    print(f"{len(emps)} employees, {total} attendance records in {clock.time() - t0:.0f}s -> db '{main.db.name}'")


if __name__ == "__main__":
    main_()
