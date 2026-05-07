from mergerag.eval.scorer import normalize, exact_match, f1


def test_normalize_lowercases():
    assert normalize("Hello World") == "hello world"


def test_normalize_strips_articles():
    assert normalize("The cat sat on a mat") == "cat sat on mat"


def test_normalize_strips_punctuation():
    assert normalize("Hello, world!") == "hello world"


def test_normalize_collapses_whitespace():
    assert normalize("  hello   world  ") == "hello world"


def test_exact_match_identical():
    assert exact_match("Paris", "Paris") == 1.0


def test_exact_match_case_insensitive():
    assert exact_match("paris", "Paris") == 1.0


def test_exact_match_strips_articles():
    assert exact_match("The cat", "cat") == 1.0


def test_exact_match_mismatch():
    assert exact_match("London", "Paris") == 0.0


def test_exact_match_punctuation_ignored():
    assert exact_match("Paris.", "Paris") == 1.0


def test_f1_exact():
    assert f1("Paris is the capital", "Paris is the capital") == 1.0


def test_f1_partial_overlap():
    result = f1("Paris capital", "Paris is the capital of France")
    assert 0.0 < result < 1.0


def test_f1_no_overlap():
    assert f1("London", "Paris") == 0.0


def test_f1_both_empty():
    assert f1("", "") == 1.0


def test_f1_pred_empty():
    assert f1("", "Paris") == 0.0


def test_f1_gold_empty():
    assert f1("Paris", "") == 0.0
