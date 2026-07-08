"""
Rule-based MATH-style answer verification.

Used in two places:
  1. As the GRPO reward function (grpo_train.py) - cheap, no PRM needed.
  2. As the correctness filter during rejection sampling (data_pipeline.py).

Extraction + normalization follows the common \\boxed{} convention used in the
Hendrycks MATH dataset. Falls back to sympy for symbolic/numeric equivalence
when plain string normalization doesn't match (e.g. "1/2" vs "0.5", "2x" vs "x*2").
"""

import re
from typing import Optional

try:
    from sympy import simplify, sympify, Rational
    from sympy.parsing.latex import parse_latex
    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False


def extract_boxed(text: str) -> Optional[str]:
    """Extract the content of the LAST \\boxed{...} in text, handling nested braces."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        idx = text.rfind("\\fbox")
        if idx == -1:
            return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    # strip common latex wrappers
    for pat in [r"\\text\{(.*?)\}", r"\\mathrm\{(.*?)\}", r"\\!", r"\\,", r"\\ "]:
        s = re.sub(pat, r"\1" if "(.*?)" in pat else "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("$", "")
    s = s.replace(" ", "")
    s = s.rstrip(".")
    # normalize \dfrac / \tfrac to \frac
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # normalize \\!\\ variants already stripped above
    return s


def _sympy_equal(a: str, b: str) -> bool:
    if not _SYMPY_AVAILABLE:
        return False
    for parser in (sympify, ):
        try:
            va = parser(a.replace("\\frac", "").replace("^", "**") if "\\" not in a else None)
        except Exception:
            va = None
    # Try direct sympify on cleaned strings (works for plain fractions/decimals/ints)
    def clean_for_sympy(x: str) -> str:
        x = x.replace("^", "**")
        x = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", x)
        x = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", x)
        x = x.replace("\\pi", "pi")
        return x

    try:
        ca, cb = clean_for_sympy(a), clean_for_sympy(b)
        va, vb = sympify(ca), sympify(cb)
        return bool(simplify(va - vb) == 0)
    except Exception:
        pass

    # last resort: try LaTeX parser if available
    try:
        va, vb = parse_latex(a), parse_latex(b)
        return bool(simplify(va - vb) == 0)
    except Exception:
        return False


def answers_match(pred: str, gold: str) -> bool:
    """True if predicted boxed answer matches gold boxed answer."""
    np_, ng_ = normalize(pred), normalize(gold)
    if np_ == ng_:
        return True
    if np_.lower() == ng_.lower():
        return True
    return _sympy_equal(np_, ng_)


def score_completion(completion_text: str, gold_boxed_or_answer: str) -> float:
    """
    Reward function for GRPO / TRL-style trainers.
    Returns 1.0 for a correct boxed answer, 0.0 otherwise.
    Also gives small partial credit (0.1) for producing a syntactically valid
    \\boxed{} even if wrong, to avoid a totally flat reward landscape early in RL.
    """
    pred = extract_boxed(completion_text)
    if pred is None:
        return 0.0
    gold = extract_boxed(gold_boxed_or_answer) or gold_boxed_or_answer
    if answers_match(pred, gold):
        return 1.0
    return 0.1


if __name__ == "__main__":
    # quick self-test
    tests = [
        ("The answer is \\boxed{1/2}", "\\boxed{0.5}", True),
        ("\\boxed{3}", "\\boxed{3}", True),
        ("\\boxed{x+1}", "\\boxed{1+x}", True),
        ("\\boxed{4}", "\\boxed{5}", False),
        ("no boxed answer here", "\\boxed{5}", False),
    ]
    for pred, gold, expected in tests:
        got = answers_match(extract_boxed(pred), extract_boxed(gold))
        status = "OK" if got == expected else "FAIL"
        print(f"[{status}] pred={pred!r} gold={gold!r} -> {got} (expected {expected})")
