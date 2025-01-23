from typing import Iterable, Iterator, List, TypeVar


T = TypeVar("T")


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Yields batches of the given size from the given iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch


class FastReplacer:
    """Fast string replacement using a dictionary of patterns and replacements. This is faster than multiple re.sub calls for longer strings"""

    def __init__(self, replacements: dict):
        self.patterns = sorted(replacements.keys(), key=len, reverse=True)
        self.replacements = replacements

    def replace(self, text: str) -> str:
        if not text:
            return text
        parts = [text]
        for pattern in self.patterns:
            new_parts = []
            for part in parts:
                if len(part) < len(pattern):
                    new_parts.append(part)
                    continue
                splits = part.split(pattern)
                for i, split in enumerate(splits):
                    if i > 0:
                        new_parts.append(self.replacements[pattern])
                    if split:
                        new_parts.append(split)
            parts = new_parts
        return "".join(parts)
