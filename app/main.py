"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience; real environment variables win)

Layout of this file:
  1. config and database
  2. time / rounding helpers and the business rules R1-R5
  3. request models and validation
  4. serialization (MongoDB document -> API shape)
  5. endpoints: system, employees, attendance
"""
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import AfterValidator, BaseModel, Field, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

log = logging.getLogger("attendance")

# --------------------------------------------------------------------------- #
# 1. Config and database
# --------------------------------------------------------------------------- #
load_dotenv(override=False)  # .env only fills in what the real environment does not set

MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = os.environ.get("MONGO_DB")
if not MONGO_URI or not MONGO_DB:
    raise RuntimeError("MONGO_URI and MONGO_DB must be set (environment or .env)")

# tz_aware=True: PyMongo hands back UTC-aware datetimes instead of naive ones, so every
# instant we read can be compared with and converted like the ones we create.
client = MongoClient(MONGO_URI, tz_aware=True, tzinfo=timezone.utc, serverSelectionTimeoutMS=5000)
db = client[MONGO_DB]
employees = db["employees"]
logs = db["attendance_logs"]

IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

PRESENCE = ("PRESENT", "WFH", "ON_DUTY")
NON_PRESENCE = ("ABSENT", "LEAVE")
Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]

INDEXES = {
    employees: [
        # identity of an employee; also what makes concurrent duplicate creates fail cleanly
        ([("emp_code", ASCENDING)], {"name": "uniq_emp_code", "unique": True}),
        # GET /employees?department=... sorted by emp_code
        ([("department", ASCENDING), ("emp_code", ASCENDING)], {"name": "department_emp_code"}),
        # headcount (R9): employees with joined_on <= end of period, optionally per department
        ([("joined_on", ASCENDING), ("department", ASCENDING)], {"name": "joined_on_department"}),
    ],
    logs: [
        # one record per employee per day: race-safe punch-in, PATCH lookup, per-employee month scans, $lookup
        ([("emp_code", ASCENDING), ("date", ASCENDING)], {"name": "uniq_emp_code_date", "unique": True}),
        # GET /attendance default order, and every month / range scan in the analytics
        ([("date", DESCENDING), ("emp_code", ASCENDING)], {"name": "date_desc_emp_code"}),
        # GET /attendance?status=... keeps the same order without an in-memory sort
        ([("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)], {"name": "status_date_emp_code"}),
        # punch-out: latest punch_in at or before the punch-out instant
        ([("emp_code", ASCENDING), ("punch_in", DESCENDING)], {"name": "emp_code_punch_in"}),
    ],
}
_indexes_ready = False


def ensure_indexes() -> None:
    """Create every index we rely on. create_index is idempotent, so this is safe on every start."""
    global _indexes_ready
    for collection, specs in INDEXES.items():
        for keys, options in specs:
            collection.create_index(keys, **options)
    _indexes_ready = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ensure_indexes()
    except PyMongoError as exc:  # MongoDB not reachable yet: /health retries and reports 503 meanwhile
        log.warning("index creation deferred: %s", exc)
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# 2. Helpers: time, rounding, business rules
# --------------------------------------------------------------------------- #
def to_ms(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    if dt.tzinfo is None:  # defensive: anything naive coming out of MongoDB is UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - EPOCH) // timedelta(milliseconds=1)


def from_ms(ms: int) -> datetime:
    """Epoch ms -> aware UTC datetime, truncated to whole seconds (R1)."""
    return EPOCH + timedelta(seconds=ms // 1000)


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def half_up(value, places: int) -> float:
    """R4/R8 rounding: half-up on the decimal value, not float round() (which is half-even on binary floats)."""
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def parse_hhmm(value: str) -> time:
    h, m = value.split(":")
    return time(int(h), int(m))


def is_overnight(shift_start: str, shift_end: str) -> bool:
    return parse_hhmm(shift_end) <= parse_hhmm(shift_start)


def attendance_date(punch_in: datetime, emp: dict) -> date:
    """R1: the IST calendar date of the punch-in; for overnight shifts, a punch-in before
    shift_end belongs to the shift that started the previous day."""
    local = punch_in.astimezone(IST)
    day = local.date()
    if is_overnight(emp["shift_start"], emp["shift_end"]) and local.time() < parse_hhmm(emp["shift_end"]):
        day -= timedelta(days=1)
    return day


def shift_bounds(day: date, emp: dict) -> tuple[datetime, datetime]:
    """Start and end instants of the shift that starts on `day` (IST)."""
    start = datetime.combine(day, parse_hhmm(emp["shift_start"]), tzinfo=IST)
    end_day = day + timedelta(days=1) if is_overnight(emp["shift_start"], emp["shift_end"]) else day
    end = datetime.combine(end_day, parse_hhmm(emp["shift_end"]), tzinfo=IST)
    return start, end


def compute_late_minutes(punch_in: datetime, emp: dict, day: date) -> int:
    """R2: late only when strictly more than 10:00 after shift start; then floor minutes since shift start."""
    start, _ = shift_bounds(day, emp)
    seconds = int((punch_in - start).total_seconds())
    return seconds // 60 if seconds > 600 else 0


def compute_overtime(punch_out: datetime, emp: dict, day: date) -> int:
    """R3: floor minutes after shift end, counted only from 30 upwards."""
    _, end = shift_bounds(day, emp)
    minutes = int((punch_out - end).total_seconds()) // 60
    return minutes if minutes >= 30 else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: seconds / 3600, 2 decimals half-up."""
    seconds = int((punch_out - punch_in).total_seconds())
    return float((Decimal(seconds) / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def derived_fields(status: str, punch_in: Optional[datetime], punch_out: Optional[datetime], emp: dict, day: date) -> dict:
    """All four derived fields for a record, per the table in DATA_MODEL.md."""
    if status not in PRESENCE or punch_in is None:
        return {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    fields = {
        "work_hours": None,
        "late_minutes": compute_late_minutes(punch_in, emp, day),
        "overtime_minutes": 0,
        "half_day": False,
    }
    if punch_out is not None:
        fields["work_hours"] = compute_work_hours(punch_in, punch_out)
        fields["overtime_minutes"] = compute_overtime(punch_out, emp, day)
        fields["half_day"] = fields["work_hours"] < 4.5  # R5 uses the rounded value
    return fields


def unprocessable(msg: str, loc: list) -> HTTPException:
    """A 422 shaped like FastAPI's own validation errors, for checks that need the database."""
    return HTTPException(422, [{"type": "value_error", "loc": loc, "msg": msg}])


def get_employee(emp_code: str) -> dict:
    emp = employees.find_one({"emp_code": emp_code}, {"_id": 0})
    if emp is None:
        raise HTTPException(404, f"employee {emp_code} not found")
    return emp


# --------------------------------------------------------------------------- #
# 3. Request models and validation
# --------------------------------------------------------------------------- #
def _check_date(value: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ValueError("must be a valid calendar date YYYY-MM-DD") from None
    return value


# Strings on purpose: a calendar date is not an instant (DATA_MODEL.md section 1).
DateStr = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$"), AfterValidator(_check_date)]
MonthStr = Annotated[str, Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]
HHMM = Annotated[str, Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
# strict: rejects floats, numeric strings and booleans; the bounds reject seconds-looking values
EpochMillis = Annotated[int, Field(strict=True, ge=100_000_000_000, le=4_102_444_800_000)]
Page = Annotated[int, Query(ge=1)]
PageSize = Annotated[int, Query(ge=1, le=100)]


class EmployeeIn(BaseModel):
    emp_code: Annotated[str, Field(pattern=r"^EMP\d{4,6}$")]
    name: Annotated[str, Field(min_length=1, max_length=100)]
    email: Annotated[str, Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]
    department: Annotated[str, Field(min_length=1, max_length=50)]
    shift_start: HHMM = "09:30"
    shift_end: HHMM = "18:30"
    joined_on: DateStr

    @model_validator(mode="after")
    def _shift_not_empty(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMillis] = None
    status: PresenceStatus = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMillis] = None


class RegularizeIn(BaseModel):
    """Derived fields a client might send are simply ignored (extra fields are dropped)."""

    status: Optional[Status] = None
    punch_in: Optional[EpochMillis] = None
    punch_out: Optional[EpochMillis] = None
    reason: Annotated[str, Field(min_length=5, max_length=200)]
    regularized_by: Annotated[str, Field(min_length=1, max_length=50)]

    @model_validator(mode="before")
    @classmethod
    def _no_explicit_nulls(cls, data):
        # "omitted" means unchanged; an explicit null is not a valid value for any of these
        if isinstance(data, dict):
            for field in ("status", "punch_in", "punch_out"):
                if field in data and data[field] is None:
                    raise ValueError(f"{field} may be omitted but not null")
        return data


# --------------------------------------------------------------------------- #
# 4. Serialization
# --------------------------------------------------------------------------- #
TIME_FIELDS = ("punch_in", "punch_out")


def employee_out(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": to_ms(doc.get("created_at")),
    }


def history_out(entry: dict) -> dict:
    changes = {}
    for field, change in (entry.get("changes") or {}).items():
        if field in TIME_FIELDS:
            change = {"from": to_ms(change.get("from")), "to": to_ms(change.get("to"))}
        changes[field] = change
    return {"at": to_ms(entry.get("at")), "by": entry.get("by"), "reason": entry.get("reason"), "changes": changes}


def record_out(doc: dict) -> dict:
    """Stored record -> AttendanceRecord. Legacy documents may omit half_day / history / minutes."""
    work_hours = doc.get("work_hours")
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": to_ms(doc.get("punch_in")),
        "punch_out": to_ms(doc.get("punch_out")),
        "work_hours": float(work_hours) if work_hours is not None else None,
        "late_minutes": int(doc.get("late_minutes") or 0),
        "overtime_minutes": int(doc.get("overtime_minutes") or 0),
        "half_day": bool(doc.get("half_day") or False),
        "history": [history_out(h) for h in doc.get("history") or []],
    }


# --------------------------------------------------------------------------- #
# 5a. System
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        if not _indexes_ready:
            ensure_indexes()
    except PyMongoError:
        raise HTTPException(503, "database unavailable")
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# 5b. Employees
# --------------------------------------------------------------------------- #
@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = datetime.now(timezone.utc).replace(microsecond=0)
    try:
        employees.insert_one(doc)  # the unique index decides, not a find-then-insert
    except DuplicateKeyError:
        raise HTTPException(409, f"emp_code {body.emp_code} already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(department: Optional[str] = None, page: Page = 1, page_size: PageSize = 20):
    q = {"department": department} if department is not None else {}
    total = employees.count_documents(q)
    cursor = employees.find(q, {"_id": 0}).sort("emp_code", ASCENDING).skip((page - 1) * page_size).limit(page_size)
    return {"items": [employee_out(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


# --------------------------------------------------------------------------- #
# 5c. Attendance
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = get_employee(body.emp_code)
    ts = from_ms(body.punched_at) if body.punched_at is not None else now_utc()
    day = attendance_date(ts, emp)
    doc = {
        "emp_code": body.emp_code,
        "date": day.isoformat(),
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        **derived_fields(body.status, ts, None, emp, day),
        "history": [],
    }
    try:
        logs.insert_one(doc)  # unique (emp_code, date): of N simultaneous inserts exactly one succeeds
    except DuplicateKeyError:
        raise HTTPException(409, f"{body.emp_code} already punched in for {day.isoformat()}")
    return record_out(doc)


ATTENDANCE_SORT = [("date", DESCENDING), ("emp_code", ASCENDING)]


def attendance_query(emp_code, date_from, date_to, status) -> dict:
    if date_from and date_to and date_from > date_to:
        raise unprocessable("date_from must not be after date_to", ["query", "date_from"])
    q: dict[str, Any] = {}
    if emp_code is not None:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status is not None:
        q["status"] = status
    return q


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Annotated[Optional[DateStr], Query()] = None,
    date_to: Annotated[Optional[DateStr], Query()] = None,
    status: Optional[Status] = None,
    page: Page = 1,
    page_size: PageSize = 20,
):
    q = attendance_query(emp_code, date_from, date_to, status)
    total = logs.count_documents(q) if q else logs.estimated_document_count()
    cursor = logs.find(q, {"_id": 0}).sort(ATTENDANCE_SORT).skip((page - 1) * page_size).limit(page_size)
    return {"items": [record_out(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


MAX_SHIFT = timedelta(hours=24)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = get_employee(body.emp_code)
    ts = from_ms(body.punched_at) if body.punched_at is not None else now_utc()
    # most recent punch-in at or before this instant; works across midnight for overnight shifts
    rec = logs.find_one({"emp_code": body.emp_code, "punch_in": {"$lte": ts}}, sort=[("punch_in", DESCENDING)])
    if rec is None:
        raise HTTPException(404, f"no punch-in found for {body.emp_code}")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, f"record for {rec['date']} is already punched out")
    if ts <= rec["punch_in"]:
        raise unprocessable("punched_at must be after punch_in", ["body", "punched_at"])
    if ts - rec["punch_in"] > MAX_SHIFT:
        raise unprocessable("punched_at is more than 24 hours after punch_in", ["body", "punched_at"])

    day = date.fromisoformat(rec["date"])
    update = {"punch_out": ts, **derived_fields(rec["status"], rec["punch_in"], ts, emp, day)}
    # the punch_out: null condition makes this the single winner among simultaneous punch-outs
    updated = logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None},
        {"$set": update},
        projection={"_id": 0},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, f"record for {rec['date']} is already punched out")
    return record_out(updated)


@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, date: DateStr, body: RegularizeIn):
    emp = get_employee(emp_code)
    rec = logs.find_one({"emp_code": emp_code, "date": date})
    if rec is None:
        raise HTTPException(404, f"no attendance record for {emp_code} on {date}")
    day = datetime.strptime(date, "%Y-%m-%d").date()

    status = body.status or rec["status"]
    if status in NON_PRESENCE:
        if body.punch_in is not None or body.punch_out is not None:
            raise unprocessable(f"{status} clears the punch times; do not send punch_in/punch_out", ["body"])
        new_in = new_out = None
    else:
        new_in = from_ms(body.punch_in) if body.punch_in is not None else rec.get("punch_in")
        new_out = from_ms(body.punch_out) if body.punch_out is not None else rec.get("punch_out")
        if new_in is None:
            raise unprocessable(f"status {status} needs a punch_in", ["body", "punch_in"])
        if body.punch_in is not None and attendance_date(new_in, emp) != day:
            raise unprocessable(f"punch_in must stay on attendance date {date}", ["body", "punch_in"])
        if new_out is not None and not (new_in < new_out <= new_in + MAX_SHIFT):
            raise unprocessable("punch_out must be after punch_in and within 24 hours", ["body", "punch_out"])

    before = {
        "status": rec["status"],
        "punch_in": rec.get("punch_in"),
        "punch_out": rec.get("punch_out"),
        "work_hours": rec.get("work_hours"),
        "late_minutes": rec.get("late_minutes") or 0,
        "overtime_minutes": rec.get("overtime_minutes") or 0,
        "half_day": rec.get("half_day") or False,
    }
    after = {"status": status, "punch_in": new_in, "punch_out": new_out, **derived_fields(status, new_in, new_out, emp, day)}
    changes = {k: {"from": before[k], "to": after[k]} for k in after if before[k] != after[k]}
    if not changes:
        raise unprocessable("the request does not change anything", ["body"])

    entry = {"at": now_utc(), "by": body.regularized_by, "reason": body.reason, "changes": changes}
    # Optimistic concurrency: only apply if nobody changed the record since we read it, i.e. the
    # editable fields are as read and history still has exactly the length we saw.
    n = len(rec.get("history") or [])
    guard = {
        "_id": rec["_id"],
        "status": rec["status"],
        "punch_in": rec.get("punch_in"),
        "punch_out": rec.get("punch_out"),
        f"history.{n}": {"$exists": False},
    }
    if n:
        guard[f"history.{n - 1}"] = {"$exists": True}
    updated = logs.find_one_and_update(
        guard,
        {"$set": after, "$push": {"history": entry}},
        projection={"_id": 0},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "the record was changed by another request; reload and retry")
    return record_out(updated)


# --------------------------------------------------------------------------- #
# 6. Analytics
#
# Every analytics endpoint is a `build_*` function returning (collection, pipeline). The endpoint
# runs it and the explain endpoint explains exactly the same thing, so the two cannot drift apart.
# Rounding is done inside the pipeline on Decimal128 (exact), because $round is half-to-even.
# --------------------------------------------------------------------------- #
def month_bounds(month: str) -> tuple[date, date]:
    first = date.fromisoformat(f"{month}-01")
    next_first = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
    return first, next_first - timedelta(days=1)


def weekdays_between(start: date, end: date) -> int:
    """Mon-Fri days in [start, end] (R7). 0 when start > end."""
    if start > end:
        return 0
    full_weeks, rest = divmod((end - start).days + 1, 7)
    return full_weeks * 5 + sum(1 for i in range(rest) if (start.weekday() + i) % 7 < 5)


def x_half_up(expr, places: int) -> dict:
    """Aggregation expression: half-up rounding on Decimal128, returned as a double (null stays null)."""
    scale = 10**places
    return {"$toDouble": {"$divide": [{"$floor": {"$add": [{"$multiply": [{"$toDecimal": expr}, scale]}, 0.5]}}, scale]}}


def x_weekday(date_expr) -> dict:
    """True when a 'YYYY-MM-DD' string falls on Mon-Fri."""
    return {"$lte": [{"$isoDayOfWeek": {"$dateFromString": {"dateString": date_expr, "format": "%Y-%m-%d"}}}, 5]}


def x_presence_weight(log: str) -> dict:
    """1 for a presence record, 0.5 for a presence half day (R5, R6), 0 otherwise."""
    return {
        "$cond": [
            {"$in": [f"{log}.status", list(PRESENCE)]},
            {"$cond": [{"$eq": [{"$ifNull": [f"{log}.half_day", False]}, True]}, 0.5, 1]},
            0,
        ]
    }


def x_is_late(log: str) -> dict:
    return {"$gt": [{"$ifNull": [f"{log}.late_minutes", 0]}, 0]}


def x_count(array: str, cond: dict) -> dict:
    return {"$size": {"$filter": {"input": array, "as": "l", "cond": cond}}}


def build_employee_monthly(emp_code: str, month: str):
    first, last = month_bounds(month)
    pipeline = [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}},
        {
            "$group": {
                "_id": None,
                # R7: present_days only counts Mon-Fri records; late/overtime count every record
                "present_days": {"$sum": {"$cond": [x_weekday("$date"), x_presence_weight("$$CURRENT"), 0]}},
                "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                "late_count": {"$sum": {"$cond": [x_is_late("$$CURRENT"), 1, 0]}},
                "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
            }
        },
    ]
    return logs, pipeline


def build_department_summary(month: str, department: Optional[str]):
    first, last = month_bounds(month)
    # Start from employees so that people with zero logs still make the headcount (R9).
    match: dict[str, Any] = {"joined_on": {"$lte": last.isoformat()}}
    if department is not None:
        match["department"] = department
    month_logs = [
        {"$match": {"date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}},
        {"$project": {"_id": 0, "date": 1, "status": 1, "half_day": 1, "late_minutes": 1, "work_hours": 1}},
    ]
    pipeline = [
        {"$match": match},
        {"$lookup": {"from": logs.name, "localField": "emp_code", "foreignField": "emp_code",
                     "pipeline": month_logs, "as": "logs"}},
        {
            "$project": {
                "_id": 0,
                "department": 1,
                "present_days": {"$sum": {"$map": {"input": "$logs", "as": "l", "in": {
                    "$cond": [x_weekday("$$l.date"), x_presence_weight("$$l"), 0]}}}},
                # avg_work_hours is an average over records, so carry the records, not a per-person mean
                "worked": {"$filter": {"input": "$logs", "as": "l", "cond": {
                    "$and": [{"$in": ["$$l.status", list(PRESENCE)]}, {"$isNumber": "$$l.work_hours"}]}}},
                "late_count": x_count("$logs", x_is_late("$$l")),
                "total_late_minutes": {"$sum": {"$map": {"input": "$logs", "as": "l", "in": {"$ifNull": ["$$l.late_minutes", 0]}}}},
                "leave_count": x_count("$logs", {"$eq": ["$$l.status", "LEAVE"]}),
                "on_duty_count": x_count("$logs", {"$eq": ["$$l.status", "ON_DUTY"]}),
            }
        },
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "present_days": {"$sum": "$present_days"},
                "work_hours_sum": {"$sum": {"$sum": {"$map": {"input": "$worked", "as": "w", "in": {"$toDecimal": "$$w.work_hours"}}}}},
                "work_hours_n": {"$sum": {"$size": "$worked"}},
                "late_count": {"$sum": "$late_count"},
                "total_late_minutes": {"$sum": "$total_late_minutes"},
                "leave_count": {"$sum": "$leave_count"},
                "on_duty_count": {"$sum": "$on_duty_count"},
            }
        },
        {"$sort": {"_id": 1}},
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": 1,
                "present_days": x_half_up("$present_days", 2),
                "avg_work_hours": {"$cond": [
                    {"$gt": ["$work_hours_n", 0]},
                    x_half_up({"$divide": ["$work_hours_sum", "$work_hours_n"]}, 2),
                    None,
                ]},
                "late_count": 1,
                "total_late_minutes": {"$toLong": "$total_late_minutes"},
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
    ]
    return employees, pipeline


def build_late_leaderboard(month: str, limit: int, department: Optional[str]):
    first, last = month_bounds(month)
    pipeline = [
        {"$match": {"date": {"$gte": first.isoformat(), "$lte": last.isoformat()}, "late_minutes": {"$gt": 0}}},
        {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": "$late_minutes"}, "late_count": {"$sum": 1}}},
        # logs whose employee does not exist drop out at the $unwind
        {"$lookup": {"from": employees.name, "localField": "_id", "foreignField": "emp_code",
                     "pipeline": [{"$project": {"_id": 0, "name": 1, "department": 1}}], "as": "emp"}},
        {"$unwind": "$emp"},
    ]
    if department is not None:
        pipeline.append({"$match": {"emp.department": department}})  # rank inside the department only
    pipeline += [
        # $rank is standard competition ranking: ties share a rank and the next one is skipped (1, 2, 2, 4)
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},  # limit applies after ranking, so ties at the cutoff all stay
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {"$project": {"_id": 0, "rank": 1, "emp_code": "$_id", "name": "$emp.name", "department": "$emp.department",
                      "total_late_minutes": {"$toLong": "$total_late_minutes"}, "late_count": 1}},
    ]
    return logs, pipeline


def build_department_trend(department: str, start: date, end: date):
    start_s, end_s = start.isoformat(), end.isoformat()
    range_logs = [
        {"$match": {"date": {"$gte": start_s, "$lte": end_s}}},
        {"$project": {"_id": 0, "date": 1, "status": 1, "half_day": 1, "late_minutes": 1}},
    ]
    pipeline = [
        {"$match": {"department": department, "joined_on": {"$lte": end_s}}},
        {"$lookup": {"from": logs.name, "localField": "emp_code", "foreignField": "emp_code",
                     "pipeline": range_logs, "as": "logs"}},
        # one event for the day each employee enters the headcount, plus one per log
        {"$project": {"_id": 0, "events": {"$concatArrays": [
            [{"date": {"$max": ["$joined_on", start_s]}, "joins": 1, "present": 0, "late": 0}],
            {"$map": {"input": "$logs", "as": "l", "in": {
                "date": "$$l.date", "joins": 0, "present": x_presence_weight("$$l"),
                "late": {"$cond": [x_is_late("$$l"), 1, 0]}}}},
        ]}}},
        {"$unwind": "$events"},
        {"$replaceWith": "$events"},
        # anchor on the first day so $densify has a row to fill from even when there is no data at all
        {"$unionWith": {"pipeline": [{"$documents": [{"date": start_s, "joins": 0, "present": 0, "late": 0}]}]}},
        {"$group": {"_id": "$date", "joins": {"$sum": "$joins"}, "present": {"$sum": "$present"}, "late": {"$sum": "$late"}}},
        {"$set": {"day": {"$dateFromString": {"dateString": "$_id", "format": "%Y-%m-%d"}}}},
        # gap filling: exactly one row per calendar day in [start, end]
        {"$densify": {"field": "day", "range": {"step": 1, "unit": "day", "bounds": [
            datetime.combine(start, time(), tzinfo=timezone.utc),
            datetime.combine(end + timedelta(days=1), time(), tzinfo=timezone.utc),
        ]}}},
        {"$setWindowFields": {"sortBy": {"day": 1}, "output": {"headcount": {
            "$sum": {"$ifNull": ["$joins", 0]}, "window": {"documents": ["unbounded", "current"]}}}}},
        {"$set": {
            "is_working_day": {"$lte": [{"$isoDayOfWeek": "$day"}, 5]},
            "present_count": {"$ifNull": ["$present", 0]},
            "late_count": {"$ifNull": ["$late", 0]},
        }},
        {"$set": {"attendance_rate": {"$cond": [
            {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
            x_half_up({"$divide": [{"$toDecimal": "$present_count"}, "$headcount"]}, 4),
            None,
        ]}}},
        # $avg skips nulls; only rows inside the range exist, so the first rows get a shorter window
        {"$setWindowFields": {"sortBy": {"day": 1}, "output": {"moving_avg_7d": {
            "$avg": "$attendance_rate", "window": {"documents": [-6, 0]}}}}},
        {"$project": {
            "_id": 0,
            "date": {"$dateToString": {"date": "$day", "format": "%Y-%m-%d"}},
            "is_working_day": 1,
            "headcount": 1,
            "present_count": 1,
            "late_count": 1,
            "attendance_rate": 1,
            "moving_avg_7d": x_half_up("$moving_avg_7d", 4),
        }},
    ]
    return employees, pipeline


def trend_range(start: str, end: str) -> tuple[date, date]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if last < first:
        raise unprocessable("to must not be before from", ["query", "to"])
    if (last - first).days + 1 > 92:
        raise unprocessable("the range may span at most 92 days", ["query", "to"])
    return first, last


Month = Annotated[MonthStr, Query()]
Limit = Annotated[int, Query(ge=1, le=50)]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: Month):
    emp = get_employee(emp_code)
    collection, pipeline = build_employee_monthly(emp_code, month)
    row = next(collection.aggregate(pipeline), None) or {}
    first, last = month_bounds(month)
    working_days = weekdays_between(max(first, date.fromisoformat(emp["joined_on"])), last)
    present_days = Decimal(str(row.get("present_days", 0)))
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": half_up(present_days, 2),
        "leave_days": int(row.get("leave_days", 0)),
        "late_count": int(row.get("late_count", 0)),
        "total_late_minutes": int(row.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(row.get("total_overtime_minutes", 0)),
        "attendance_pct": half_up(present_days * 100 / working_days, 2) if working_days else None,
    }


@app.get("/analytics/departments/summary")
def department_summary(month: Month, department: Optional[str] = None):
    collection, pipeline = build_department_summary(month, department)
    return {"month": month, "items": list(collection.aggregate(pipeline))}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(month: Month, limit: Limit = 10, department: Optional[str] = None):
    collection, pipeline = build_late_leaderboard(month, limit, department)
    return {"month": month, "items": list(collection.aggregate(pipeline))}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    start: Annotated[DateStr, Query(alias="from")],
    end: Annotated[DateStr, Query(alias="to")],
):
    first, last = trend_range(start, end)
    if employees.find_one({"department": department}, {"_id": 1}) is None:
        raise HTTPException(404, f"department {department} not found")
    collection, pipeline = build_department_trend(department, first, last)
    return {"department": department, "items": list(collection.aggregate(pipeline))}
