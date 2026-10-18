from copy import deepcopy

import pytest
import torch

from physisserve.models.wan import WanDiT, _CachedTextProjection
from tests.test_models import tiny_transformer


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_diffusers_attention_preserves_full_pack_projection_and_precision(tmp_path, dtype):
    torch.manual_seed(17)
    model, _, channels, text_dim = tiny_transformer("wan21_1p3b")
    model.save_pretrained(tmp_path)
    dit = WanDiT(tmp_path)
    dit.load((0,), "cpu", dtype)
    attention = dit.model.blocks["0"].attn2
    reference = deepcopy(attention)
    shape = (1, channels, 1, 4, 4)
    with torch.inference_mode():
        condition = {"text": torch.randn(1, 5, text_dim)}
        dit.prepare(condition, shape, 8)
        projections = {name: getattr(attention, name) for name in ("to_k", "to_v", "norm_k")}
        calls = {name: [] for name in projections}
        for name, projection in projections.items():
            projection.module.register_forward_hook(
                lambda module, args, output, name=name: calls[name].append(tuple(args[0].shape)))
        texts = [torch.randn(1, 5, 16, dtype=dtype) for _ in range(4)]
        packs = [(texts[0],), (texts[0], texts[1]), (texts[1], texts[0]),
                 (texts[0], texts[0]), tuple(texts)]
        for index, pack in enumerate(packs, 1):
            text = dit._batch_text(pack)
            assert dit._batch_text(pack) is text
            torch.testing.assert_close(text, torch.cat(pack), rtol=0, atol=0)
            for _ in range(2):
                query = torch.randn(len(pack), 3, 16, dtype=dtype)
                expected = reference(query, torch.cat(pack))
                actual = attention(query, text)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert all(len(values) == index for values in calls.values())
            assert all(values[-1] == (len(pack), 5, 16) for values in calls.values())
        # A new request retains the installed modules but none of its cached inputs.
        dit.prepare(condition, shape, 8)
        assert not dit._text_batches
        for name, projection in projections.items():
            assert getattr(attention, name) is projection
            assert not projection.entries
        text = dit._batch_text(packs[0])
        query = torch.randn(1, 3, 16, dtype=dtype)
        torch.testing.assert_close(attention(query, text), reference(query, torch.cat(packs[0])),
                                   rtol=0, atol=0)
        assert all(len(values) == len(packs) + 1 for values in calls.values())


def test_text_pack_and_projection_memos_are_bounded_and_preserve_lru_order():
    dit = WanDiT.__new__(WanDiT)
    dit._text_batches = []
    module = torch.nn.Linear(4, 4).eval().requires_grad_(False)
    projection = _CachedTextProjection(module)
    calls = []
    module.register_forward_hook(lambda module, args, output: calls.append(args[0]))
    with torch.inference_mode():
        sources = [torch.randn(1, 3, 4) for _ in range(10)]
        packs = [dit._batch_text((source,)) for source in sources[:8]]
        results = [projection(pack) for pack in packs]
        assert dit._batch_text((sources[0],)) is packs[0]
        assert projection(packs[0]) is results[0]
        for source in sources[8:]:
            projection(dit._batch_text((source,)))
        assert len(dit._text_batches) == len(projection.entries) == 8
        indices = {id(source): index for index, source in enumerate(sources)}
        assert [indices[id(pack[0])] for pack, _ in dit._text_batches] == [3, 4, 5, 6, 7, 0, 8, 9]
        assert all(source is not packs[1] and source is not packs[2]
                   for source, _ in projection.entries)
        assert projection(packs[0]) is results[0]
        assert len(calls) == 10
        evicted = dit._batch_text((sources[1],))
        assert evicted is not packs[1]
        torch.testing.assert_close(projection(evicted), module(evicted), rtol=0, atol=0)
