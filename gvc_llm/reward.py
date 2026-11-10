"""SimpleRL's binary math correctness reward, without a format penalty.

Use verl's Math-Verify integration to avoid vendoring SimpleRL's old parser.
The same function is used by GRPO and GVC.
"""
from math_verify import parse, verify, StringExtractionConfig
from math_verify.errors import TimeoutException
from verl.utils.reward_score.math_verify import compute_score as math_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # TimeoutException subclasses BaseException, so `except Exception` (including the
    # one inside verl's wrapper) does not catch it; treat timeouts as incorrect.
    try:
        # DeepMath also contains verbal binary answers, which the math-only parser misses.
        if ground_truth.strip().lower() in ("yes", "no"):
            config = [StringExtractionConfig(strings=("Yes", "No", "yes", "no"))]
            return float(verify(parse(ground_truth, extraction_config=config),
                                parse(solution_str, extraction_config=config)))
        return float(math_score(solution_str, ground_truth))
    except TimeoutException:
        return 0.0
