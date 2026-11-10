"""Log-space KL controller, updated once after each critic training step."""

import math


class AdaptiveBeta:
    def __init__(self, beta=0.01, kl_target=1.0, kl_range=0.1, delta_log_beta=0.1,
                 min_beta=0.0001, max_beta=100.0):
        if not (math.isfinite(min_beta) and math.isfinite(max_beta)
                and 0 < min_beta <= max_beta):
            raise ValueError("beta bounds must be finite and satisfy 0 < min_beta <= max_beta")
        if math.isnan(beta) or beta <= 0:
            raise ValueError("beta must be positive")
        if math.isfinite(beta) and not min_beta <= beta <= max_beta:
            raise ValueError("initial beta must be within [min_beta, max_beta]")
        if not math.isfinite(kl_target) or kl_target < 0:
            raise ValueError("kl_target must be finite and nonnegative")
        if not math.isfinite(kl_range) or not 0 <= kl_range <= kl_target:
            raise ValueError("kl_range must be finite and between zero and kl_target")
        if not math.isfinite(delta_log_beta) or delta_log_beta <= 0:
            raise ValueError("delta_log_beta must be finite and positive")
        self.log_beta = math.log2(beta)
        self.min_beta, self.max_beta = min_beta, max_beta
        self.min_log_beta, self.max_log_beta = math.log2(min_beta), math.log2(max_beta)
        self.kl_target = kl_target
        self.kl_range = kl_range
        self.delta_log_beta = delta_log_beta

    @property
    def beta(self):
        if math.isinf(self.log_beta):
            return float('inf')
        # Guard against floating-point roundoff at the endpoints as well.
        return min(self.max_beta, max(self.min_beta, 2.0 ** self.log_beta))

    def update(self, kl):
        if not math.isfinite(kl):
            raise ValueError("KL must be finite")
        # Infinity explicitly selects the unguided policy and stays fixed.
        adaptive = math.isfinite(self.log_beta)
        increment = int(adaptive and kl > self.kl_target + self.kl_range)
        decrement = int(adaptive and kl < self.kl_target - self.kl_range)
        previous = self.log_beta
        if adaptive:
            self.log_beta = min(self.max_log_beta, max(
                self.min_log_beta, self.log_beta + self.delta_log_beta * (increment - decrement)))
        return {
            "train/beta": self.beta,
            "train/log_beta": self.log_beta,
            "train/beta_increment": int(self.log_beta > previous),
            "train/beta_decrement": int(self.log_beta < previous),
        }
