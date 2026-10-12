# DECISIONS.md

1. **Indexes.** On `attendance_logs`: unique `(emp_code, date)` makes duplicate records impossible and serves every per-employee lookup and `$lookup`. `(date -1, emp_code 1)` matches the list sort and every month/range scan. `(status, date -1, emp_code 1)` keeps a status-filtered list sorted from the index. `(emp_code, punch_in -1)` finds the record to punch out. On `employees`: unique `emp_code`, `(department, emp_code)` for the list, and `(joined_on, department)` for headcount. I rejected copying `department` into the logs: it is faster, but it goes stale.

2. **Punch-in race.** Both requests validate and compute the same document. Both call `insert_one`, and the unique index lets exactly one write through. The other gets `DuplicateKeyError`, which becomes a 409. There is no read-then-write window.

3. **Ties.** `$rank` gives competition ranks (1, 2, 2, 4), and I filter `rank <= limit` after ranking. With `limit=2` and two people tied at rank 2, both come back (three rows), ordered by `emp_code`.

4. **Headcount.** The summary starts from `employees` (`joined_on` <= month end), not from logs, and `$lookup`s each person's logs. Someone with no logs gets an empty array but still counts once.

5. **100x data.** I'd precompute a daily per-department rollup at write time, so the analytics read about 30 rows instead of scanning millions of logs.
