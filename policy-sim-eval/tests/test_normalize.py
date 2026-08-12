"""Boundary-aware parsing test cases (EVAL_GUIDELINE.md sec 3.5 test table)."""
from src.io.normalize import extract_quantities, satisfies, Quantity


def _find(text, kind, bound=None):
    qs = extract_quantities(text)
    for q in qs:
        if q.kind == kind and (bound is None or q.bound == bound):
            return q
    raise AssertionError(f"no {kind}/{bound} quantity found in {qs!r} for text={text!r}")


def test_up_to_sgd():
    q = _find("provides co-funding of up to S$330,000 per project", "currency", "at_most")
    assert q.value == 330000
    assert q.currency == "SGD"


def test_more_than_sgd_million():
    q = _find("committed more than S$500 million through AI Singapore", "currency", "at_least")
    assert q.value == 5e8
    assert q.currency == "SGD"


def test_sgd_million_exact():
    q = _find("a fund of S$0.33 million was set aside", "currency", "exact")
    assert q.value == 330000
    assert q.currency == "SGD"


def test_krw_eok_exact():
    q = _find("연간 예산 2.6억원 지원", "currency", "exact")
    assert q.value == 2.6e8
    assert q.currency == "KRW"


def test_krw_plain_exact():
    q = _find("총 260,000,000원 규모", "currency", "exact")
    assert q.value == 2.6e8
    assert q.currency == "KRW"


def test_years_exact():
    q = _find("4단계 BK21 사업 7년간 지원", "years", "exact")
    assert q.value == 7


def test_range_months():
    q = _find("typically 6-12 months to complete", "months", "range")
    assert q.value == 6
    assert q.range_hi == 12


def test_days_with_calendar_infix():
    q = _find("not later than 30 calendar days after", "days", "exact")
    assert q.value == 30


def test_days_with_business_infix():
    q = _find("no later than 60 business days after the complaint", "days", "exact")
    assert q.value == 60


def test_korean_day_of_month_not_misparsed_as_duration():
    qs = extract_quantities("5월 1일부터 K-패스 사업을 시행한다")
    assert not any(q.kind == "days" for q in qs)


def test_percentage_point_exact():
    q = _find("an increase of 8.5 percentage points", "percentage_point", "exact")
    assert q.value == 8.5


def test_percent_exact():
    q = _find("a rate of 8.5%", "percent", "exact")
    assert q.value == 8.5


def test_approx_count():
    q = _find("about 2,500 participants", "count", "exact")
    assert q.value == 2500
    assert q.approx is True


def test_satisfies_boundary_semantics():
    ref_at_most = Quantity(330000, "currency", "SGD", "at_most", None, False, "", (0, 0))
    observed_ok = Quantity(300000, "currency", "SGD", "exact", None, False, "", (0, 0))
    observed_bad = Quantity(400000, "currency", "SGD", "exact", None, False, "", (0, 0))
    assert satisfies(observed_ok, ref_at_most)
    assert not satisfies(observed_bad, ref_at_most)

    ref_at_least = Quantity(5e8, "currency", "SGD", "at_least", None, False, "", (0, 0))
    observed_550m = Quantity(5.5e8, "currency", "SGD", "exact", None, False, "", (0, 0))
    assert satisfies(observed_550m, ref_at_least)

    ref_range = Quantity(6, "months", None, "range", 12, False, "", (0, 0))
    assert satisfies(Quantity(9, "months", None, "exact", None, False, "", (0, 0)), ref_range)
    assert not satisfies(Quantity(15, "months", None, "exact", None, False, "", (0, 0)), ref_range)

    ref_currency_mismatch = Quantity(1000, "currency", "SGD", "exact", None, False, "", (0, 0))
    observed_wrong_currency = Quantity(1000, "currency", "USD", "exact", None, False, "", (0, 0))
    assert not satisfies(observed_wrong_currency, ref_currency_mismatch)
