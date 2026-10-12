"""End-to-end tests against a real MongoDB (MONGO_URI from the environment / .env).

They use their own database (MONGO_DB_TEST, default attendance_test) and wipe it, so never point
them at data you care about.

    pytest -q
"""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from dotenv import load_dotenv

load_dotenv()
os.environ["MONGO_DB"] = os.environ.get("MONGO_DB_TEST", "attendance_test")

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))


def ms(ist_iso: str) -> int:
    """'2026-07-06T09:40:01' (IST) -> epoch ms."""
    return int(datetime.fromisoformat(ist_iso).replace(tzinfo=IST).timestamp() * 1000)


def utc(ist_iso: str) -> datetime:
    return datetime.fromisoformat(ist_iso).replace(tzinfo=IST).astimezone(timezone.utc)


@pytest.fixture(scope="module")
def api():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(autouse=True)
def clean_db():
    main.employees.delete_many({})
    main.logs.delete_many({})
    main.ensure_indexes()
    yield


def make_emp(api, code="EMP0001", dept="Engineering", start="09:30", end="18:30", joined="2026-01-05"):
    r = api.post("/employees", json={"emp_code": code, "name": "Test " + code, "email": f"{code.lower()}@x.io",
                                     "department": dept, "shift_start": start, "shift_end": end, "joined_on": joined})
    assert r.status_code == 201, r.text
    return r.json()


# --------------------------------------------------------------------------- employees
def test_health(api):
    assert api.get("/health").json() == {"status": "ok"}


def test_create_employee_shape_and_validation(api):
    e = make_emp(api)
    assert set(e) == {"emp_code", "name", "email", "department", "shift_start", "shift_end", "joined_on", "created_at"}
    assert isinstance(e["created_at"], int) and e["created_at"] > 10**11
    assert api.post("/employees", json={**e, "created_at": None}).status_code == 409
    bad = [{"emp_code": "E1"}, {"email": "nope"}, {"shift_start": "24:00"}, {"joined_on": "2026-02-30"},
           {"shift_start": "09:30", "shift_end": "09:30"}, {"name": ""}]
    base = {"emp_code": "EMP0002", "name": "A", "email": "a@b.co", "department": "D", "joined_on": "2026-01-01"}
    for patch in bad:
        assert api.post("/employees", json={**base, **patch}).status_code == 422, patch


def test_concurrent_duplicate_create(api):
    body = {"emp_code": "EMP0042", "name": "A", "email": "a@b.co", "department": "D", "joined_on": "2026-01-01"}
    with ThreadPoolExecutor(8) as pool:
        codes = sorted(pool.map(lambda _: TestClient(main.app).post("/employees", json=body).status_code, range(8)))
    assert codes == [201] + [409] * 7


def test_list_employees_filter_sort_paginate(api):
    for code, dept in [("EMP0003", "Sales"), ("EMP0001", "Sales"), ("EMP0002", "Ops")]:
        make_emp(api, code, dept)
    r = api.get("/employees", params={"department": "Sales", "page": 1, "page_size": 1}).json()
    assert r["total"] == 2 and [e["emp_code"] for e in r["items"]] == ["EMP0001"]
    r = api.get("/employees", params={"department": "Sales", "page": 2, "page_size": 1}).json()
    assert [e["emp_code"] for e in r["items"]] == ["EMP0003"]
    for params in [{"page": 0}, {"page_size": 0}, {"page_size": 101}]:
        assert api.get("/employees", params=params).status_code == 422


# --------------------------------------------------------------------------- punch-in / punch-out
@pytest.mark.parametrize("t,late", [("09:40:00", 0), ("09:40:01", 10), ("10:15:59", 45), ("09:00:00", 0)])
def test_late_minutes_r2(api, t, late):
    make_emp(api)
    r = api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms(f"2026-07-06T{t}")})
    assert r.status_code == 201 and r.json()["late_minutes"] == late


def test_punch_in_record_shape_and_truncation(api):
    make_emp(api)
    r = api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06T09:40:00") + 900,
                                               "status": "WFH"}).json()
    assert r == {"emp_code": "EMP0001", "date": "2026-07-06", "status": "WFH", "punch_in": ms("2026-07-06T09:40:00"),
                 "punch_out": None, "work_hours": None, "late_minutes": 0, "overtime_minutes": 0,
                 "half_day": False, "history": []}


def test_punch_in_date_is_ist_not_utc(api):
    make_emp(api, start="06:00", end="14:00")
    r = api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-07T01:00:00")})
    assert r.json()["date"] == "2026-07-07"  # 2026-07-06T19:30Z


def test_punch_in_errors(api):
    make_emp(api)
    good = ms("2026-07-06T09:30:00")
    assert api.post("/attendance/punch-in", json={"emp_code": "EMP0999", "punched_at": good}).status_code == 404
    for value in [good // 1000, float(good), str(good), True, 5_000_000_000_000]:
        assert api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": value}).status_code == 422
    for status in ["ABSENT", "LEAVE", "present"]:
        r = api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": good, "status": status})
        assert r.status_code == 422
    assert api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": good}).status_code == 201
    assert api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": good + 3600_000}).status_code == 409


def test_concurrent_punch_in(api):
    make_emp(api)
    body = {"emp_code": "EMP0001", "punched_at": ms("2026-07-06T09:30:00")}
    with ThreadPoolExecutor(10) as pool:
        codes = sorted(pool.map(lambda _: TestClient(main.app).post("/attendance/punch-in", json=body).status_code, range(10)))
    assert codes == [201] + [409] * 9


def test_punch_out_rules(api):
    make_emp(api)
    api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06T10:05:00")})
    r = api.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06T19:10:00")})
    assert r.status_code == 200
    body = r.json()
    assert (body["work_hours"], body["late_minutes"], body["overtime_minutes"], body["half_day"]) == (9.08, 35, 40, False)
    assert body["history"] == []
    again = api.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06T19:20:00")})
    assert again.status_code == 409


def test_punch_out_overtime_threshold_and_half_day(api):
    make_emp(api)
    api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06T09:30:00")})
    r = api.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06T18:59:59")}).json()
    assert r["overtime_minutes"] == 0  # 29:59 after shift end
    make_emp(api, "EMP0002")
    api.post("/attendance/punch-in", json={"emp_code": "EMP0002", "punched_at": ms("2026-07-06T09:30:00")})
    # 4h29m42s = 4.495h -> 4.50 half-up (float round gives 4.49): not a half day
    r = api.post("/attendance/punch-out", json={"emp_code": "EMP0002", "punched_at": ms("2026-07-06T13:59:42")}).json()
    assert (r["work_hours"], r["half_day"]) == (4.5, False)


def test_overnight_shift(api):
    make_emp(api, "EMP0005", "Support", "22:00", "06:00")
    r = api.post("/attendance/punch-in", json={"emp_code": "EMP0005", "punched_at": ms("2026-07-06T21:55:00")}).json()
    assert (r["date"], r["late_minutes"]) == ("2026-07-06", 0)
    r = api.post("/attendance/punch-out", json={"emp_code": "EMP0005", "punched_at": ms("2026-07-07T06:40:00")}).json()
    assert (r["date"], r["work_hours"], r["overtime_minutes"]) == ("2026-07-06", 8.75, 40)
    # a punch-in after midnight but before shift_end belongs to the previous day's shift
    r = api.post("/attendance/punch-in", json={"emp_code": "EMP0005", "punched_at": ms("2026-07-08T00:30:00")}).json()
    assert (r["date"], r["late_minutes"]) == ("2026-07-07", 150)


def test_punch_out_errors(api):
    make_emp(api)
    assert api.post("/attendance/punch-out", json={"emp_code": "EMP0999"}).status_code == 404
    assert api.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06T18:00:00")}).status_code == 404
    api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06T09:30:00")})
    too_late = api.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-07T09:30:01")})
    assert too_late.status_code == 422
    same = api.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06T09:30:00")})
    assert same.status_code == 422


def test_concurrent_punch_out(api):
    make_emp(api)
    api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06T09:30:00")})
    body = {"emp_code": "EMP0001", "punched_at": ms("2026-07-06T18:30:00")}
    with ThreadPoolExecutor(10) as pool:
        codes = sorted(pool.map(lambda _: TestClient(main.app).post("/attendance/punch-out", json=body).status_code, range(10)))
    assert codes == [200] + [409] * 9


# --------------------------------------------------------------------------- regularize
def test_regularize_recomputes_and_appends_history(api):
    make_emp(api)
    api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-07T10:05:00")})
    api.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-07T18:30:00")})
    r = api.patch("/attendance/EMP0001/2026-07-07", json={"punch_in": ms("2026-07-07T09:28:00"), "reason": "biometric glitch",
                                                          "regularized_by": "hr.admin", "late_minutes": 99})
    assert r.status_code == 200, r.text
    rec = r.json()
    assert (rec["late_minutes"], rec["work_hours"]) == (0, 9.03)
    (entry,) = rec["history"]
    assert entry["by"] == "hr.admin" and entry["reason"] == "biometric glitch" and isinstance(entry["at"], int)
    assert entry["changes"] == {"punch_in": {"from": ms("2026-07-07T10:05:00"), "to": ms("2026-07-07T09:28:00")},
                                "late_minutes": {"from": 35, "to": 0}, "work_hours": {"from": 8.42, "to": 9.03}}
    stored = main.logs.find_one({"emp_code": "EMP0001"})
    assert isinstance(stored["history"][0]["changes"]["punch_in"]["from"], datetime)

    r = api.patch("/attendance/EMP0001/2026-07-07", json={"status": "LEAVE", "reason": "was on leave", "regularized_by": "hr"})
    rec = r.json()
    assert (rec["punch_in"], rec["punch_out"], rec["work_hours"], rec["late_minutes"]) == (None, None, None, 0)
    assert len(rec["history"]) == 2


def test_regularize_errors(api):
    make_emp(api)
    api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-07T09:30:00")})
    base = {"reason": "fixing it", "regularized_by": "hr"}
    url = "/attendance/EMP0001/2026-07-07"
    assert api.patch("/attendance/EMP0999/2026-07-07", json=base).status_code == 404
    assert api.patch("/attendance/EMP0001/2026-07-08", json={**base, "status": "WFH"}).status_code == 404
    assert api.patch("/attendance/EMP0001/2026-13-01", json={**base, "status": "WFH"}).status_code == 422
    assert api.patch(url, json={**base, "status": "PRESENT"}).status_code == 422  # changes nothing
    assert api.patch(url, json={**base, "status": "ABSENT", "punch_in": ms("2026-07-07T09:30:00")}).status_code == 422
    assert api.patch(url, json={**base, "punch_in": ms("2026-07-08T09:30:00")}).status_code == 422  # other day
    assert api.patch(url, json={**base, "punch_out": ms("2026-07-07T09:00:00")}).status_code == 422
    assert api.patch(url, json={**base, "punch_in": None}).status_code == 422
    assert api.patch(url, json={"status": "WFH", "reason": "shrt", "regularized_by": "hr"}).status_code == 422
    api.patch(url, json={**base, "status": "ABSENT"})
    assert api.patch(url, json={**base, "status": "PRESENT"}).status_code == 422  # presence without punch_in


def test_concurrent_regularize_keeps_history(api):
    make_emp(api)
    api.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-07T09:30:00")})
    statuses = ["WFH", "ON_DUTY", "PRESENT", "WFH", "ON_DUTY", "PRESENT", "WFH", "ON_DUTY"]

    def patch(s):
        return TestClient(main.app).patch("/attendance/EMP0001/2026-07-07",
                                          json={"status": s, "reason": "concurrent", "regularized_by": s}).status_code

    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(patch, statuses))
    assert set(codes) <= {200, 409, 422}
    history = main.logs.find_one({"emp_code": "EMP0001"})["history"]
    assert len(history) == codes.count(200) >= 1


# --------------------------------------------------------------------------- list attendance
def test_list_attendance(api):
    seed_analytics()
    r = api.get("/attendance", params={"page_size": 100}).json()
    keys = [(i["date"], i["emp_code"]) for i in r["items"]]
    expected = sorted(keys, key=lambda k: k[1])  # stable sorts: emp_code asc, then date desc
    assert keys == sorted(expected, key=lambda k: k[0], reverse=True)
    assert r["total"] == main.logs.count_documents({}) == len(keys)
    page2 = api.get("/attendance", params={"page": 2, "page_size": 3}).json()
    assert [(i["date"], i["emp_code"]) for i in page2["items"]] == keys[3:6]
    r = api.get("/attendance", params={"emp_code": "EMP1001", "date_from": "2026-07-07", "date_to": "2026-07-11"}).json()
    assert [i["date"] for i in r["items"]] == ["2026-07-11", "2026-07-08", "2026-07-07"] and r["total"] == 3
    legacy = api.get("/attendance", params={"emp_code": "EMP1001", "date_from": "2026-07-13", "date_to": "2026-07-13"}).json()
    assert legacy["items"][0]["half_day"] is False and legacy["items"][0]["history"] == []
    assert api.get("/attendance", params={"status": "LEAVE"}).json()["total"] == 1
    assert api.get("/attendance", params={"date_from": "2026-07-10", "date_to": "2026-07-01"}).status_code == 422
    assert api.get("/attendance", params={"status": "late"}).status_code == 422


# --------------------------------------------------------------------------- analytics
def log(code, day, status="PRESENT", wh=None, late=0, ot=0, half=False, legacy=False):
    doc = {"emp_code": code, "date": day, "status": status,
           "punch_in": utc(f"{day}T09:30:00") if status not in ("ABSENT", "LEAVE") else None,
           "punch_out": None, "work_hours": wh, "late_minutes": late, "overtime_minutes": ot}
    if not legacy:
        doc.update(half_day=half, history=[])
    return doc


def seed_analytics():
    """July 2026: Jul 1 is a Wednesday, so July has 23 weekdays."""
    emp = lambda code, dept, joined: {"emp_code": code, "name": "N" + code, "email": "x@y.z", "department": dept,
                                      "shift_start": "09:30", "shift_end": "18:30", "joined_on": joined,
                                      "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}
    main.employees.insert_many([
        emp("EMP1001", "Engineering", "2026-01-01"),
        emp("EMP1002", "Engineering", "2026-07-15"),  # mid-month joiner
        emp("EMP1003", "Engineering", "2026-01-01"),  # no logs at all
        emp("EMP1004", "Engineering", "2026-08-10"),  # joins after July
        emp("EMP2001", "Sales", "2026-01-01"),
        emp("EMP2002", "Sales", "2026-01-01"),
        emp("EMP2003", "Sales", "2026-01-01"),
        emp("EMP3001", "Future", "2026-09-01"),
    ])
    main.logs.insert_many([
        log("EMP1001", "2026-07-06", wh=9.0),
        log("EMP1001", "2026-07-07", wh=4.0, late=20, half=True),
        log("EMP1001", "2026-07-08", "LEAVE"),
        log("EMP1001", "2026-07-11", wh=5.0, late=15, ot=40),  # Saturday
        log("EMP1001", "2026-07-13", "WFH", wh=8.01, legacy=True),
        log("EMP1001", "2026-06-30", wh=8.0, late=500),  # previous month
        log("EMP1002", "2026-07-20", "ON_DUTY", wh=8.5),
        log("EMP1002", "2026-07-21", "ABSENT"),
        log("EMP2001", "2026-07-06", wh=8.0, late=30),
        log("EMP2002", "2026-07-07", wh=8.0, late=12),
        log("EMP2002", "2026-07-08", wh=4.49, late=18, half=True),
        log("EMP2003", "2026-07-09", wh=8.97, late=10),
        log("EMP9999", "2026-07-09", wh=8.0, late=100),  # orphan log
    ])


def test_employee_monthly(api):
    seed_analytics()
    get = lambda code: api.get(f"/analytics/employees/{code}/monthly", params={"month": "2026-07"})
    assert get("EMP1001").json() == {"emp_code": "EMP1001", "month": "2026-07", "working_days": 23, "present_days": 2.5,
                                     "leave_days": 1, "late_count": 2, "total_late_minutes": 35,
                                     "total_overtime_minutes": 40, "attendance_pct": 10.87}
    e2 = get("EMP1002").json()
    assert (e2["working_days"], e2["present_days"], e2["attendance_pct"]) == (13, 1, 7.69)
    assert get("EMP1003").json()["attendance_pct"] == 0.0
    assert get("EMP1004").json()["attendance_pct"] is None
    assert get("EMP0404").status_code == 404
    assert api.get("/analytics/employees/EMP1001/monthly", params={"month": "2026-13"}).status_code == 422


def test_department_summary(api):
    seed_analytics()
    r = api.get("/analytics/departments/summary", params={"month": "2026-07"}).json()
    assert r == {"month": "2026-07", "items": [
        {"department": "Engineering", "headcount": 3, "present_days": 3.5, "avg_work_hours": 6.9, "late_count": 2,
         "total_late_minutes": 35, "leave_count": 1, "on_duty_count": 1},
        # (8 + 8 + 4.49 + 8.97) / 4 = 7.365 -> 7.37 half-up (half-even would say 7.36)
        {"department": "Sales", "headcount": 3, "present_days": 3.5, "avg_work_hours": 7.37, "late_count": 4,
         "total_late_minutes": 70, "leave_count": 0, "on_duty_count": 0},
    ]}
    only = api.get("/analytics/departments/summary", params={"month": "2026-07", "department": "Sales"}).json()
    assert [i["department"] for i in only["items"]] == ["Sales"]
    june = api.get("/analytics/departments/summary", params={"month": "2026-06", "department": "Engineering"}).json()
    assert june["items"][0]["headcount"] == 2 and june["items"][0]["avg_work_hours"] == 8.0


def test_late_leaderboard_ties(api):
    seed_analytics()
    board = lambda **p: api.get("/analytics/leaderboard/late", params={"month": "2026-07", **p}).json()["items"]
    rows = board(limit=4)
    assert [(r["rank"], r["emp_code"], r["total_late_minutes"]) for r in rows] == [
        (1, "EMP1001", 35), (2, "EMP2001", 30), (2, "EMP2002", 30), (4, "EMP2003", 10)]
    assert rows[2] == {"rank": 2, "emp_code": "EMP2002", "name": "NEMP2002", "department": "Sales",
                       "total_late_minutes": 30, "late_count": 2}
    assert [r["emp_code"] for r in board(limit=2)] == ["EMP1001", "EMP2001", "EMP2002"]  # tie at the cutoff
    assert [r["emp_code"] for r in board(limit=3)] == ["EMP1001", "EMP2001", "EMP2002"]
    assert [r["emp_code"] for r in board(limit=1)] == ["EMP1001"]
    assert [(r["rank"], r["emp_code"]) for r in board(limit=1, department="Sales")] == [(1, "EMP2001"), (1, "EMP2002")]
    assert api.get("/analytics/leaderboard/late", params={"month": "2026-07", "limit": 51}).status_code == 422


def test_department_trend(api):
    seed_analytics()
    r = api.get("/analytics/departments/Engineering/trend", params={"from": "2026-07-12", "to": "2026-07-21"})
    assert r.status_code == 200, r.text
    rows = r.json()["items"]
    got = [(x["date"], x["is_working_day"], x["headcount"], x["present_count"], x["attendance_rate"], x["moving_avg_7d"])
           for x in rows]
    assert got == [
        ("2026-07-12", False, 2, 0, None, None),
        ("2026-07-13", True, 2, 1, 0.5, 0.5),
        ("2026-07-14", True, 2, 0, 0.0, 0.25),
        ("2026-07-15", True, 3, 0, 0.0, 0.1667),
        ("2026-07-16", True, 3, 0, 0.0, 0.125),
        ("2026-07-17", True, 3, 0, 0.0, 0.1),
        ("2026-07-18", False, 3, 0, None, 0.1),
        ("2026-07-19", False, 3, 0, None, 0.1),
        ("2026-07-20", True, 3, 1, 0.3333, 0.0667),
        ("2026-07-21", True, 3, 0, 0.0, 0.0667),
    ]
    sales = api.get("/analytics/departments/Sales/trend", params={"from": "2026-07-06", "to": "2026-07-08"}).json()["items"]
    assert [(x["present_count"], x["late_count"]) for x in sales] == [(1, 1), (1, 1), (0.5, 1)]


def test_department_trend_errors(api):
    seed_analytics()
    url = "/analytics/departments/{}/trend"
    assert api.get(url.format("Nope"), params={"from": "2026-07-01", "to": "2026-07-02"}).status_code == 404
    assert api.get(url.format("Sales"), params={"from": "2026-07-02", "to": "2026-07-01"}).status_code == 422
    assert api.get(url.format("Sales"), params={"from": "2026-07-01", "to": "2026-09-30"}).status_code == 200  # 92 days
    assert api.get(url.format("Sales"), params={"from": "2026-07-01", "to": "2026-10-01"}).status_code == 422
    future = api.get(url.format("Future"), params={"from": "2026-07-01", "to": "2026-07-03"}).json()["items"]
    assert [(x["headcount"], x["attendance_rate"]) for x in future] == [(0, None)] * 3


# --------------------------------------------------------------------------- explain
@pytest.mark.parametrize("endpoint,params", [
    ("attendance_list", {}),
    ("attendance_list", {"emp_code": "EMP1001", "date_from": "2026-07-01"}),
    ("attendance_list", {"status": "LEAVE"}),
    ("employee_monthly", {"emp_code": "EMP1001", "month": "2026-07"}),
    ("department_summary", {"month": "2026-07"}),
    ("department_summary", {"month": "2026-07", "department": "Sales"}),
    ("late_leaderboard", {"month": "2026-07"}),
    ("department_trend", {"department": "Engineering", "from": "2026-07-01", "to": "2026-07-31"}),
])
def test_explain_uses_indexes(api, endpoint, params):
    seed_analytics()
    r = api.get(f"/admin/explain/{endpoint}", params=params)
    assert r.status_code == 200, r.text
    text = str(r.json()["explain"])
    assert "IXSCAN" in text and "COLLSCAN" not in text


def test_explain_validation(api):
    assert api.get("/admin/explain/employee_monthly", params={"month": "2026-07"}).status_code == 422
    assert api.get("/admin/explain/department_trend", params={"department": "X"}).status_code == 422
    assert api.get("/admin/explain/bogus").status_code == 422
