"""Reproducible ordering with an early round across each probe category."""

from __future__ import annotations

import random
import secrets
from collections import defaultdict
from typing import Any


def sample_order(cases: list[dict[str, Any]], *, strategy: str = "stratified", seed: int | None = None) -> tuple[list[dict[str, Any]], int]:
    if strategy not in {"stratified", "ordered"}:
        raise ValueError("sampling must be stratified or ordered")
    if seed is not None and type(seed) is not int:
        raise ValueError("sample_seed must be an integer")
    actual_seed = secrets.randbits(63) if seed is None else seed
    if strategy == "ordered":
        return list(cases), actual_seed
    rng = random.Random(actual_seed)
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        categories[case["category"]].append(case)
    for group in categories.values():
        rng.shuffle(group)
    ordered: list[dict[str, Any]] = []
    while categories:
        active = sorted(categories)
        rng.shuffle(active)
        for category in active:
            ordered.append(categories[category].pop())
            if not categories[category]:
                del categories[category]
    return ordered, actual_seed
