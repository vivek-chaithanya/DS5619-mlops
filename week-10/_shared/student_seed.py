"""
Shared helper — turns a student ID (roll number, email, whatever the course
uses) into a deterministic 32-bit random seed, so every week's data
generator can produce a DIFFERENT dataset per student from the SAME script,
with the SAME size/shape (so grading logic and smoke tests still apply
uniformly), but different actual values.

Why this exists: it's not primarily about hiding the assignment from an
LLM — an LLM can still write correct code against personalized data just as
well as against shared data. It's about making a shared or copy-pasted
ANSWER (a filled-in JSON report, a hardcoded row index, someone else's
completed script run against THEIR data) fail when checked against a
specific student's own data. If two students submit byte-identical
`*_report.json` output, that's now a strong, checkable signal, because two
different --student-id values essentially never produce the same seed.

Usage in any generate_for_student.py:

    from student_seed import seed_from_student_id
    seed = seed_from_student_id(args.student_id, salt="week03")

The `salt` argument means the same student gets a DIFFERENT seed in each
week (so Week 3's dataset and Week 10's dataset for the same student aren't
correlated) — pass the week's own name/number as the salt.
"""
import hashlib


def seed_from_student_id(student_id: str, salt: str = "") -> int:
    """Deterministically derive a 32-bit unsigned int seed from a student
    ID string (e.g. roll number or institute email) and an optional salt
    (e.g. the week name, so seeds don't repeat across weeks for the same
    student). Same inputs always produce the same seed; different
    student_id values produce effectively unrelated seeds.
    """
    if not student_id or not student_id.strip():
        raise ValueError(
            "student_id is required and can't be blank — use your roll "
            "number or institute email, e.g. --student-id 21CS10042"
        )
    normalized = student_id.strip().lower()
    digest = hashlib.sha256(f"{salt}:{normalized}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)  # first 32 bits of the hash, as an int
