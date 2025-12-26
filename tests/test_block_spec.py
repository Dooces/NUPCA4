import pytest

from nupca3.geometry.block_spec import BlockView, build_block_specs


def test_block_view_matches_legacy_blocks():
    blocks = [[0, 1, 2], [3, 4], [5]]
    specs = build_block_specs(blocks)
    view = BlockView(specs)
    assert view.legacy_blocks() == blocks
    for idx, block_dims in enumerate(blocks):
        assert view.block_cost(idx) == float(len(block_dims))
        assert view.block_kind(idx) == "leaf"
