from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence


@dataclass
class BlockSpec:
    block_id: int
    dims: List[int]
    cost: float
    level: int
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    kind: str = "leaf"


class BlockView:
    def __init__(self, specs: Sequence[BlockSpec]):
        self.specs: Dict[int, BlockSpec] = {int(s.block_id): s for s in specs}

    def legacy_blocks(self) -> List[List[int]]:
        return [spec.dims.copy() for spec in self.leaf_specs()]

    def leaf_specs(self) -> List[BlockSpec]:
        return [spec for spec in self.specs.values() if spec.kind == "leaf"]

    def block_cost(self, block_id: int) -> float:
        spec = self.specs.get(int(block_id))
        return float(spec.cost) if spec is not None else float("nan")

    def block_kind(self, block_id: int) -> str:
        spec = self.specs.get(int(block_id))
        return spec.kind if spec is not None else "unknown"


def build_block_specs(
    blocks: Iterable[Sequence[int]],
    *,
    cost_fn: Optional[Callable[[Sequence[int]], float]] = None,
) -> List[BlockSpec]:
    result: List[BlockSpec] = []
    for block_id, dims in enumerate(blocks):
        dims_list = list(int(d) for d in dims)
        cost = float(cost_fn(dims) if cost_fn is not None else len(dims_list))
        result.append(BlockSpec(block_id=int(block_id), dims=dims_list, cost=cost, level=0))
    return result
