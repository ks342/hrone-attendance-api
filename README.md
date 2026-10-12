# Employee Attendance & Analytics API

[![tests](https://github.com/ks342/hrone-attendance-api/actions/workflows/tests.yml/badge.svg)](https://github.com/ks342/hrone-attendance-api/actions/workflows/tests.yml)

FastAPI + MongoDB backend for the HROne assignment: punch-in/out, regularization with an audit trail, and four analytics endpoints built as aggregation pipelines. Everything the app runs is in [`app/main.py`](app/main.py).

## Run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # set MONGO_URI and MONGO_DB
uvicorn app.main:app --port 8000
```

`MONGO_URI` and `MONGO_DB` come from the environment. A `.env` only fills in what isn't already set. The app needs MongoDB 6.0+ (it uses `$densify`, `$setWindowFields` and `$documents`). Indexes are created at startup and doing so is idempotent. If MongoDB isn't reachable yet, `/health` returns 503 and creates them once it is.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q                                   # needs MONGO_URI; uses and wipes MONGO_DB_TEST (default attendance_test)
python scripts/generate_data.py             # ~100k realistic records into MONGO_DB, for explain/timing checks
```

The tests hit a real MongoDB through the API. They cover R1-R5 at the edges (grace period, overnight shifts, half-up rounding, the 30-minute overtime threshold), the concurrency guarantees, PATCH history, and the analytics on a hand-built dataset. That dataset includes employees with no logs, mid-month and future joiners, weekend records, ties at the leaderboard cutoff, legacy documents and orphan logs, with the expected numbers worked out by hand. They also check that every explain plan uses `IXSCAN` and never `COLLSCAN`.

## Notes

* Rounding (R4, R8) is half-up on decimals, never float `round()`. Inside pipelines it's done on `Decimal128`, because MongoDB's `$round` rounds half to even.
* Each analytics endpoint and `/admin/explain` call the same `build_*` function, so the explained plan is exactly the one the endpoint runs.
* A PATCH that loses a race with another PATCH on the same record gets a 409. History entries are never lost or overwritten.
* Interpretations where the contract leaves room: the trend range limit is 92 days counted inclusively (a full quarter is allowed). `moving_avg_7d` averages the rounded daily rates. "A request that changes nothing" means none of the stored fields, derived ones included, would change.

See [REVIEW.md](REVIEW.md) for the defects found in the starter, and [DECISIONS.md](DECISIONS.md) for the design questions.
