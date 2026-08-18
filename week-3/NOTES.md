# NOTES.md — Week 3: ETL and Data Validation

**Student ID used with `generate_for_student.py`:**
142301024


## Quarantine count vs. the 7 known injected problems

**Quarantined rows:** 6
**Total violations found:** 8

The 7 known injected problems are:
1. Null amount (row 370)
2. Null amount (row 534)
3. Negative amount (row 11)
4. Invalid merchant_category "crypto_kiosk" (row 140)
5. Invalid country "ZZ" (row 266) - Note: country column is not validated in the expectation suite
6. Null card_id (row 388)
7. Duplicate transaction_id (rows 175 and 275)

**Discrepancy explanation:**
- 6 distinct rows were quarantined, not 7, because the country "ZZ" violation is not caught by any expectation in the suite (the suite validates: amount not null, card_id not null, amount positive, merchant_category in set, transaction_id unique - but NOT country codes).
- 8 total violations were found across 6 rows because 2 rows have multiple violations:
  - Row 370: null amount (expect_column_not_null) + not positive amount (expect_column_positive)
  - Row 534: null amount (expect_column_not_null) + not positive amount (expect_column_positive)
- The duplicate transaction_id only flags the second occurrence (row 275) as a violation, not the first (row 175), which is correct behavior.

So: 7 injected problem types, but only 6 rows actually caught by the validation suite (one problem type - invalid country - has no corresponding expectation), and 2 rows have multiple violations, yielding 8 total violations.