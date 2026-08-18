"""
A minimal, from-scratch expectations framework in the spirit of Great
Expectations / data contracts (this week's lecture). You are implementing
the checking logic yourself rather than importing a library — the goal is
to understand what these tools actually do under the hood.

Fill in the four functions marked # TODO. Do not change the Violation
dataclass or any function signature.
"""
from dataclasses import dataclass


@dataclass
class Violation:
    expectation: str      # name of the check, e.g. "expect_column_not_null"
    column: str            # which column it was checking
    row_index: int          # index into the rows list where it failed
    detail: str              # short human-readable reason


def _is_null(value):
    return value is None or (isinstance(value, str) and value.strip() == "")


def expect_column_not_null(rows, column):
    """Return a Violation for every row where rows[i][column] is null/empty."""
    violations = []
    for i, row in enumerate(rows):
        if _is_null(row.get(column)):
            violations.append(Violation(
                expectation="expect_column_not_null",
                column=column,
                row_index=i,
                detail=f"{column} is null or empty"
            ))
    return violations


def expect_column_positive(rows, column):
    """Return a Violation for every row where rows[i][column], cast to float,
    is not strictly greater than 0. If the value can't be cast to float at
    all, that also counts as a violation (detail should say so).
    """
    violations = []
    for i, row in enumerate(rows):
        value = row.get(column)
        if _is_null(value):
            violations.append(Violation(
                expectation="expect_column_positive",
                column=column,
                row_index=i,
                detail=f"{column} is null or empty, cannot be positive"
            ))
        else:
            try:
                num = float(value)
                if num <= 0:
                    violations.append(Violation(
                        expectation="expect_column_positive",
                        column=column,
                        row_index=i,
                        detail=f"{column} is not positive: {num}"
                    ))
            except (ValueError, TypeError):
                violations.append(Violation(
                    expectation="expect_column_positive",
                    column=column,
                    row_index=i,
                    detail=f"{column} cannot be cast to float: {value!r}"
                ))
    return violations


def expect_column_in_set(rows, column, allowed_values):
    """Return a Violation for every row where rows[i][column] is not a member
    of allowed_values (a set or list you're given).
    """
    violations = []
    allowed_set = set(allowed_values)
    for i, row in enumerate(rows):
        value = row.get(column)
        if _is_null(value) or value not in allowed_set:
            violations.append(Violation(
                expectation="expect_column_in_set",
                column=column,
                row_index=i,
                detail=f"{column} value {value!r} not in allowed set"
            ))
    return violations


def expect_column_unique(rows, column):
    """Return a Violation for every row AFTER THE FIRST that repeats a value
    already seen in `column`. (i.e. if three rows share a value, rows 2 and 3
    are violations; row 1 is not.)
    """
    violations = []
    seen = {}
    for i, row in enumerate(rows):
        value = row.get(column)
        if value in seen:
            violations.append(Violation(
                expectation="expect_column_unique",
                column=column,
                row_index=i,
                detail=f"{column} value {value!r} already seen at row {seen[value]}"
            ))
        else:
            seen[value] = i
    return violations
