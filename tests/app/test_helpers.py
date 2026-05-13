from app.helpers import compute_em, strip_citations


def test_strip_citations_removes_bracket_ids():
    assert strip_citations("Paris [chunk-0000]") == "Paris"


def test_strip_citations_multiple():
    assert strip_citations("Yes [a-0000] [b-0001]") == "Yes"


def test_strip_citations_no_citations():
    assert strip_citations("Paris") == "Paris"


def test_compute_em_exact_match():
    assert compute_em("Paris", "Paris") == 1.0


def test_compute_em_case_insensitive():
    assert compute_em("paris", "Paris") == 1.0


def test_compute_em_strips_whitespace():
    assert compute_em("  Paris  ", "Paris") == 1.0


def test_compute_em_no_match():
    assert compute_em("London", "Paris") == 0.0


def test_compute_em_strips_citations_before_compare():
    assert compute_em("Paris [chunk-0000]", "Paris") == 1.0
