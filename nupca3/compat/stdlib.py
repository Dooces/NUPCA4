"""Utility adapters exposing a narrow stdlib-like surface for kernel wiring."""
from __future__ import annotations

import collections
import hashlib
from typing import Callable, Generic, Iterable, Iterator, List, Optional, Sequence, TypeVar

T = TypeVar("T")


def hash64(data: bytes) -> int:
    """Deterministically hash `data` to a 64-bit unsigned integer."""
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def popcount64(value: int) -> int:
    """Return the number of set bits in a 64-bit quantity."""
    return value.bit_count()


def hamming64(a: int, b: int) -> int:
    """Hamming distance between two uint64 values."""
    return popcount64(a ^ b)


class RingBuffer(Generic[T]):
    """Simple deterministic ring buffer wrapper around collections.deque."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buffer: collections.deque[T] = collections.deque(maxlen=capacity)
        self.capacity = capacity

    def append(self, item: T) -> None:
        """Append a new element, evicting the oldest if at capacity."""
        self._buffer.append(item)

    def extend(self, items: Iterable[T]) -> None:
        """Append multiple items in order."""
        for item in items:
            self.append(item)

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[T]:
        return iter(self._buffer)

    def __bool__(self) -> bool:
        return bool(self._buffer)

    def clear(self) -> None:
        """Empty the buffer."""
        self._buffer.clear()

    def oldest(self) -> T:
        """Return the oldest element without removing it."""
        return self._buffer[0]

    def newest(self) -> T:
        """Return the newest element without removing it."""
        return self._buffer[-1]


def deterministic_truncate(
    items: Sequence[T],
    limit: int,
    key: Optional[Callable[[T], object]] = None,
) -> List[T]:
    """
    Return a deterministic slice of `items` respecting `limit`.

    If `key` is provided, items are ordered by (key(item), insertion_index) so the
    truncation behavior is stable even when `key` values are identical.
    """
    if limit is None or limit >= len(items):
        return list(items)

    if key is None:
        return list(items[:limit])

    decorated = [(key(item), idx, item) for idx, item in enumerate(items)]
    decorated.sort()
    return [item for _, _, item in decorated[:limit]]
