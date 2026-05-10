from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


def log_duration(name: str, duration: float) -> None:
    log.info("TIMING %s: %.2fs", name, duration)


class Timer:
    def __init__(self, name: str) -> None:
        self._name = name
        self._start: float | None = None

    def __enter__(self) -> Timer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *args) -> None:
        if self._start is not None:
            log_duration(self._name, time.monotonic() - self._start)
