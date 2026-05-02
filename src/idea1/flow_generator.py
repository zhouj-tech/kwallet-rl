#3/25 idea1
import random
from typing import List, Optional


# =========================
# Basic transaction samplers
# =========================

def sample_small(rng: random.Random) -> int:
    """Sample a small transaction."""
    return rng.randint(1, 100)


def sample_medium(rng: random.Random) -> int:
    """Sample a medium transaction."""
    return rng.randint(101, 400)


def sample_large(rng: random.Random) -> int:
    """Sample a large transaction."""
    return rng.randint(401, 1000)


def sample_by_category(rng: random.Random, category: str) -> int:
    """Sample one transaction value based on category name."""
    if category == "small":
        return sample_small(rng)
    if category == "medium":
        return sample_medium(rng)
    if category == "large":
        return sample_large(rng)
    raise ValueError(f"Unknown category: {category}")


# =========================
# Core helper
# =========================

def generate_segment(
    length: int,
    small_prob: float,
    medium_prob: float,
    large_prob: float,
    rng: random.Random,
) -> List[int]:
    """
    Generate one segment of transactions using category probabilities.

    The probabilities should sum to 1.0.
    """
    total = small_prob + medium_prob + large_prob
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Probabilities must sum to 1.0, but got {total:.6f}"
        )

    flow = []
    for _ in range(length):
        u = rng.random()
        if u < small_prob:
            flow.append(sample_small(rng))
        elif u < small_prob + medium_prob:
            flow.append(sample_medium(rng))
        else:
            flow.append(sample_large(rng))
    return flow


# =========================
# Regime A: Regular
# - small: 70%
# - medium: 30%
# - large: 0%
# - stable
# =========================

def generate_regular_flow(
    length: int,
    seed: Optional[int] = None,
) -> List[int]:
    rng = random.Random(seed)
    return generate_segment(
        length=length,
        small_prob=0.70,
        medium_prob=0.30,
        large_prob=0.00,
        rng=rng,
    )


# =========================
# Regime B: Heavy-Tail
# - small: 75%
# - medium: 20%
# - large: 5%
# - stable, large appears sparsely
# =========================

def generate_heavytail_flow(
    length: int,
    seed: Optional[int] = None,
) -> List[int]:
    rng = random.Random(seed)
    return generate_segment(
        length=length,
        small_prob=0.75,
        medium_prob=0.20,
        large_prob=0.05,
        rng=rng,
    )


# =========================
# Regime C: Burst
# - normal phase: small 70%, medium 30%, large 0%
# - burst phase:  small 30%, medium 20%, large 50%
# - one burst window in the middle
# =========================

def generate_burst_flow(
    length: int,
    seed: Optional[int] = None,
    burst_start_ratio: float = 0.30,
    burst_end_ratio: float = 0.60,
) -> List[int]:
    """
    Generate a burst regime flow.

    Default burst window:
    - first 30%: normal
    - middle 30%: burst
    - last 40%: normal
    """
    if not (0.0 <= burst_start_ratio < burst_end_ratio <= 1.0):
        raise ValueError("burst_start_ratio and burst_end_ratio must satisfy 0 <= start < end <= 1")

    rng = random.Random(seed)

    burst_start = int(length * burst_start_ratio)
    burst_end = int(length * burst_end_ratio)

    pre_len = burst_start
    burst_len = burst_end - burst_start
    post_len = length - burst_end

    pre_flow = generate_segment(
        length=pre_len,
        small_prob=0.70,
        medium_prob=0.30,
        large_prob=0.00,
        rng=rng,
    )

    burst_flow = generate_segment(
        length=burst_len,
        small_prob=0.30,
        medium_prob=0.20,
        large_prob=0.50,
        rng=rng,
    )

    post_flow = generate_segment(
        length=post_len,
        small_prob=0.70,
        medium_prob=0.30,
        large_prob=0.00,
        rng=rng,
    )

    return pre_flow + burst_flow + post_flow


# =========================
# Regime D: Switching
# - first half: same as Regular
# - second half: same as Burst
# - regime change at midpoint
# =========================

def generate_switching_flow(
    length: int,
    seed: Optional[int] = None,
    second_half_burst_start_ratio: float = 0.30,
    second_half_burst_end_ratio: float = 0.60,
) -> List[int]:
    """
    Generate a switching regime flow:
    - first half = Regular
    - second half = Burst
    """
    rng = random.Random(seed)

    first_half_len = length // 2
    second_half_len = length - first_half_len

    # First half: Regular
    first_half = generate_segment(
        length=first_half_len,
        small_prob=0.70,
        medium_prob=0.30,
        large_prob=0.00,
        rng=rng,
    )

    # Second half: Burst
    if not (0.0 <= second_half_burst_start_ratio < second_half_burst_end_ratio <= 1.0):
        raise ValueError("second_half_burst_start_ratio and second_half_burst_end_ratio must satisfy 0 <= start < end <= 1")

    burst_start = int(second_half_len * second_half_burst_start_ratio)
    burst_end = int(second_half_len * second_half_burst_end_ratio)

    pre_len = burst_start
    burst_len = burst_end - burst_start
    post_len = second_half_len - burst_end

    second_pre = generate_segment(
        length=pre_len,
        small_prob=0.70,
        medium_prob=0.30,
        large_prob=0.00,
        rng=rng,
    )

    second_burst = generate_segment(
        length=burst_len,
        small_prob=0.30,
        medium_prob=0.20,
        large_prob=0.50,
        rng=rng,
    )

    second_post = generate_segment(
        length=post_len,
        small_prob=0.70,
        medium_prob=0.30,
        large_prob=0.00,
        rng=rng,
    )

    second_half = second_pre + second_burst + second_post

    return first_half + second_half


# =========================
# Optional dispatcher
# =========================

def generate_flow(
    regime_name: str,
    length: int,
    seed: Optional[int] = None,
) -> List[int]:
    regime_name = regime_name.lower()

    if regime_name in {"a", "regular"}:
        return generate_regular_flow(length=length, seed=seed)

    if regime_name in {"b", "heavytail", "heavy_tail", "heavy-tail"}:
        return generate_heavytail_flow(length=length, seed=seed)

    if regime_name in {"c", "burst"}:
        return generate_burst_flow(length=length, seed=seed)

    if regime_name in {"d", "switching"}:
        return generate_switching_flow(length=length, seed=seed)

    raise ValueError(f"Unknown regime_name: {regime_name}")


# =========================
# Simple test
# =========================

if __name__ == "__main__":
    n = 20
    seed = 42

    flow_a = generate_regular_flow(n, seed=seed)
    flow_b = generate_heavytail_flow(n, seed=seed)
    flow_c = generate_burst_flow(n, seed=seed)
    flow_d = generate_switching_flow(n, seed=seed)

    print("Regime A (Regular):")
    print(flow_a)
    print()

    print("Regime B (Heavy-Tail):")
    print(flow_b)
    print()

    print("Regime C (Burst):")
    print(flow_c)
    print()

    print("Regime D (Switching):")
    print(flow_d)
    print()