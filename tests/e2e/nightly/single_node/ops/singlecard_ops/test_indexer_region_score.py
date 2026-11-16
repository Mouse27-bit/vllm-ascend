# SPDX-License-Identifier: Apache-2.0
"""Precision test for the Step5 indexer region score operator.

Calls ``torch.ops._C_ascend.npu_indexer_region_score.ids_out`` directly
(the exact call ``vllm_ascend/models/step5/ops/indexer_region_score.py``
makes for the Step5 DSA indexer: fused region scoring + causal top-k +
ascending logical-id sort, ids-only path with a preallocated
``selected_region_ids`` buffer, ``cache_mode=1, mask_mode=3, sort_mode=1``).

Reference semantics (csrc/attention/indexer_region_score/README.md): for
query token ``j`` of request ``n``, group ``g``, the visible region count is
``clamp((actual_kv_lengths[n] - q_len + j) // region_size, 0, max_region_len)``
and each visible region ``r`` scores
``sum_h(weights[t, g, h] * head_scale * relu(dot(query[t, g, h], key[n, r])))``
where ``key[n, r] = mean_cache[block_table[n, r // S] , r % S, 0, :]``. IDs
are selected by descending score (ties -> smaller logical ID) and emitted in
ascending order with ``-1`` tail padding.

All inputs use binary fractions that are exact in FP8 E4M3 / BF16 / FP32,
so the FP32 torch oracle is bit-exact and the INT32 ids are compared exactly
(integer/index output -> exact comparison).
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import enable_custom_op

HEAD_DIM = 256
REGION_SIZE = 8


def make_inputs(query_lengths, regions, page_size, histories, *, groups=1, heads, dtype, affine_pages):
    """Binary-fraction inputs (exact through the FP8 quantize) and a poisoned
    3-layer cache backing whose layer-1 view is the operator input."""
    generator = torch.Generator().manual_seed(431)
    tokens = sum(query_lengths)
    columns = max(1, (regions + page_size - 1) // page_size)
    pages = len(query_lengths) * columns + 3
    query = (torch.randint(-2, 3, (tokens, groups, heads, HEAD_DIM), generator=generator).float() / 4).to(dtype)
    weights = (torch.randint(-4, 5, (tokens, groups, heads), generator=generator).float() / 8).to(torch.float32)
    key = (torch.randint(-2, 3, (pages, page_size, 1, HEAD_DIM), generator=generator).float() / 4).to(
        torch.float8_e4m3fn
    )
    # Poison unused layers so a wrong layer offset or page stride shows up.
    backing = torch.full((pages, 3, page_size, 1, HEAD_DIM), 31.0).to(torch.float8_e4m3fn)
    backing[:, 1] = key
    if affine_pages:
        table = torch.arange(len(query_lengths) * columns, dtype=torch.int32).view(len(query_lengths), columns)
    else:
        table = torch.randint(0, pages, (len(query_lengths), columns), generator=generator, dtype=torch.int32)
        if columns > 2:
            table[:, 1] = -1
            table[:, 2] = pages  # out of range on purpose; never visible
    cu = torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()], dtype=torch.int32)
    lengths = torch.tensor(
        [q + history * REGION_SIZE for q, history in zip(query_lengths, histories)], dtype=torch.int32
    )
    return query, weights, backing, lengths, table, cu


def reference_ids(query, weights, backing, lengths, table, cu, *, regions, top_k, head_scale):
    """FP32 oracle; dyadic inputs keep every score exact, including ties."""
    page_size = backing.shape[2]
    groups = query.shape[1]
    ids = torch.full((groups, query.shape[0], top_k), -1, dtype=torch.int32)
    logical = torch.arange(regions)
    for request in range(table.shape[0]):
        start, end = int(cu[request]), int(cu[request + 1])
        physical = table[request, logical // page_size].long()
        valid = (physical >= 0) & (physical < backing.shape[0])
        keys = backing[physical.clamp(0, backing.shape[0] - 1), 1, logical % page_size, 0].float()
        keys[~valid] = 0
        # [seg, G, H, R] -> weighted per-head ReLU scores summed over H.
        dots = torch.matmul(query[start:end].float(), keys.T).relu()
        values = (dots * weights[start:end, :, :, None] * head_scale).sum(dim=2)
        for token in range(start, end):
            pos = int(lengths[request]) - (end - start) + (token - start)
            visible = max(0, min(regions, pos // REGION_SIZE))
            if visible == 0:
                continue
            selected = torch.argsort(values[token - start, :, :visible], descending=True, stable=True)[:, :top_k]
            ids[:, token, : selected.shape[1]] = selected.sort().values.to(torch.int32)
    return ids


def check_case(
    query_lengths, regions, page_size, top_k, histories, *, groups, heads, dtype, head_scale=1.0, affine_pages=False
):
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("IndexerRegionScore requires Ascend 950 (A5)")
    enable_custom_op()
    query, weights, backing, lengths, table, cu = make_inputs(
        query_lengths, regions, page_size, histories, groups=groups, heads=heads, dtype=dtype, affine_pages=affine_pages
    )
    expected = reference_ids(
        query, weights, backing, lengths, table, cu, regions=regions, top_k=top_k, head_scale=head_scale
    )

    tokens = sum(query_lengths)
    # Mirror the production wrapper: the query is quantized to FP8 E4M3
    # before the call (values here are exactly representable, so the oracle
    # stays exact); the preallocated [G, T, top_k] ids buffer is the
    # selected_region_ids output of the ids_out overload.
    query_npu = query.to(torch.float8_e4m3fn).npu()
    ids = torch.empty((groups, tokens, top_k), dtype=torch.int32, device="npu")
    torch.ops._C_ascend.npu_indexer_region_score.ids_out(
        query_npu,
        weights.npu(),
        backing.npu()[:, 1],
        lengths.npu(),
        table.npu(),
        cu.npu(),
        max_region_len=regions,
        top_k=top_k,
        region_size=REGION_SIZE,
        head_scale=head_scale,
        cache_mode=1,
        mask_mode=3,
        sort_mode=1,
        selected_region_ids=ids,
    )

    torch.npu.synchronize()
    assert ids.shape == (groups, tokens, top_k)
    assert ids.dtype == torch.int32
    torch.testing.assert_close(
        ids.cpu(), expected, rtol=0, atol=0, msg="region ids mismatch"
    )
    torch.npu.empty_cache()


def test_decode_real_shape_under_page():
    """Real Step5 indexer shape (G=1, H=16, FP8) on a decode batch.

    Histories keep every visible region inside the first (random) page; the
    poisoned table columns must never be read. top_k > visible rows leave
    -1 padding."""
    check_case((1, 1, 1, 1), 37, 17, 64, (0, 0, 16, 1), groups=1, heads=16, dtype=torch.float8_e4m3fn)


def test_prefill_chunk_topk_512():
    """Real top_k=512 prefill chunk crossing many pages (affine table): a
    64-token request with a long history plus a short one, -1 padding below
    512 visible candidates."""
    check_case(
        (64, 37), 900, 16, 512, (100, 3), groups=1, heads=16, dtype=torch.float8_e4m3fn, affine_pages=True
    )


def test_multi_group_full_history_head_scale():
    """G=4 (README group constraint), full-history selection saturating
    top_k=17, and a non-default head_scale."""
    check_case(
        (4,), 65, 16, 17, (65,), groups=4, heads=4, dtype=torch.float8_e4m3fn, head_scale=0.5, affine_pages=True
    )


def test_bf16_fallback_input_is_quantized():
    """The non-fused fallback hands BF16 queries in; the wrapper quantizes
    them to FP8 (values chosen exactly representable, so the oracle stays
    exact)."""
    check_case((9,), 130, 16, 32, (12,), groups=1, heads=8, dtype=torch.bfloat16, affine_pages=True)


def test_first_chunk_visible_zero():
    """A request's first-ever chunk: every query position is below one
    region, so all id rows must be entirely -1."""
    check_case((3,), 40, 16, 8, (0,), groups=1, heads=4, dtype=torch.float8_e4m3fn)
