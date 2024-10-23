import logging

from splatnlp.preprocessing.constants import (
    BUCKET_THRESHOLDS,
    MAIN_ONLY_ABILITIES,
    NULL,
    STANDARD_ABILITIES,
)

logger = logging.getLogger(__name__)


def tokenize_build(build: dict[str, int]) -> list[str]:
    if not build:
        return [NULL]

    total_ap = sum(build.values())
    if total_ap > 57:
        raise ValueError("Total ability points cannot exceed 57")

    if any(isinstance(value, float) for value in build.values()):
        raise ValueError("Ability points must be integers")

    tokens = []
    for ability in MAIN_ONLY_ABILITIES:
        if ability not in build:
            continue
        tokens.append(ability)

    for ability in STANDARD_ABILITIES:
        if ability not in build:
            continue
        for threshold in BUCKET_THRESHOLDS:
            if build[ability] >= threshold:
                tokens.append(f"{ability}_{threshold}")

    return tokens
