"""Fused inference kernels for sparse convex token planning."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the normal runtime ships Triton
    triton = None
    tl = None


def triton_acceleration_available() -> bool:
    return triton is not None


if triton is not None:
    @triton.jit
    def _streaming_top4_kernel(
        query_ptr, basis_t_ptr, value_ptr, index_ptr,
        rows, basis_count, channels: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        block_m = tl.program_id(0)
        query_row = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        valid_query = query_row < rows
        top0 = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        top1 = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        top2 = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        top3 = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        id0 = tl.full((BLOCK_M,), -1, tl.int32)
        id1 = tl.full((BLOCK_M,), -1, tl.int32)
        id2 = tl.full((BLOCK_M,), -1, tl.int32)
        id3 = tl.full((BLOCK_M,), -1, tl.int32)
        for basis_begin in range(0, basis_count, BLOCK_N):
            basis_column = basis_begin + tl.arange(0, BLOCK_N)
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for channel_begin in tl.static_range(0, channels, BLOCK_K):
                channel = channel_begin + tl.arange(0, BLOCK_K)
                query = tl.load(
                    query_ptr + query_row[:, None] * channels + channel[None, :],
                    mask=valid_query[:, None] & (channel[None, :] < channels), other=0.0,
                )
                basis = tl.load(
                    basis_t_ptr + channel[:, None] * basis_count + basis_column[None, :],
                    mask=(channel[:, None] < channels) & (basis_column[None, :] < basis_count),
                    other=0.0,
                )
                accumulator += tl.dot(query, basis)
            # The released retrieval path materializes a BF16 GEMM result
            # before Top-k.  Round once here as well so rank decisions keep
            # the same precision contract while the matrix stays streaming.
            accumulator = accumulator.to(tl.bfloat16).to(tl.float32)
            accumulator = tl.where(
                valid_query[:, None] & (basis_column[None, :] < basis_count),
                accumulator, -float("inf"),
            )
            for _ in tl.static_range(0, 4):
                tile_value = tl.max(accumulator, axis=1)
                tile_local = tl.argmax(accumulator, axis=1)
                tile_id = basis_begin + tile_local
                accumulator = tl.where(
                    tl.arange(0, BLOCK_N)[None, :] == tile_local[:, None],
                    -float("inf"), accumulator,
                )
                better = tile_value > top0
                carry_value = tl.where(better, top0, tile_value)
                carry_id = tl.where(better, id0, tile_id)
                top0 = tl.where(better, tile_value, top0)
                id0 = tl.where(better, tile_id, id0)
                better = carry_value > top1
                next_value = tl.where(better, top1, carry_value)
                next_id = tl.where(better, id1, carry_id)
                top1 = tl.where(better, carry_value, top1)
                id1 = tl.where(better, carry_id, id1)
                better = next_value > top2
                carry_value = tl.where(better, top2, next_value)
                carry_id = tl.where(better, id2, next_id)
                top2 = tl.where(better, next_value, top2)
                id2 = tl.where(better, next_id, id2)
                better = carry_value > top3
                top3 = tl.where(better, carry_value, top3)
                id3 = tl.where(better, carry_id, id3)
        output = query_row * 4
        tl.store(value_ptr + output, top0, mask=valid_query)
        tl.store(value_ptr + output + 1, top1, mask=valid_query)
        tl.store(value_ptr + output + 2, top2, mask=valid_query)
        tl.store(value_ptr + output + 3, top3, mask=valid_query)
        tl.store(index_ptr + output, id0, mask=valid_query)
        tl.store(index_ptr + output + 1, id1, mask=valid_query)
        tl.store(index_ptr + output + 2, id2, mask=valid_query)
        tl.store(index_ptr + output + 3, id3, mask=valid_query)


    @triton.jit
    def _gram_entry(gram_ptr, row, first, second, ridge: tl.constexpr):
        value = tl.load(gram_ptr + row * 16 + first * 4 + second).to(tl.float32)
        return value + tl.where(first == second, ridge, 0.0)


    @triton.jit
    def _four_support_active_candidates_kernel(
        gram_ptr, rhs_ptr, active_ids_ptr, active_count_ptr,
        candidate_ptr, objective_ptr, rows, ridge: tl.constexpr,
    ):
        program = tl.program_id(0)
        row = program // 15
        active_set = program - row * 15
        valid_row = row < rows
        count = tl.load(active_count_ptr + active_set, mask=valid_row, other=1).to(tl.int32)
        base = active_set * 4
        i0 = tl.load(active_ids_ptr + base).to(tl.int32)
        i1 = tl.load(active_ids_ptr + base + 1).to(tl.int32)
        i2 = tl.load(active_ids_ptr + base + 2).to(tl.int32)
        p = tl.load(active_ids_ptr + base + count - 1).to(tl.int32)
        hp = _gram_entry(gram_ptr, row, p, p, ridge)
        rp = tl.load(rhs_ptr + row * 4 + p, mask=valid_row, other=0.0).to(tl.float32)

        h00 = _gram_entry(gram_ptr, row, i0, i0, ridge) - 2.0 * _gram_entry(gram_ptr, row, i0, p, ridge) + hp
        h01 = _gram_entry(gram_ptr, row, i0, i1, ridge) - _gram_entry(gram_ptr, row, i0, p, ridge) - _gram_entry(gram_ptr, row, p, i1, ridge) + hp
        h02 = _gram_entry(gram_ptr, row, i0, i2, ridge) - _gram_entry(gram_ptr, row, i0, p, ridge) - _gram_entry(gram_ptr, row, p, i2, ridge) + hp
        h10 = _gram_entry(gram_ptr, row, i1, i0, ridge) - _gram_entry(gram_ptr, row, i1, p, ridge) - _gram_entry(gram_ptr, row, p, i0, ridge) + hp
        h11 = _gram_entry(gram_ptr, row, i1, i1, ridge) - 2.0 * _gram_entry(gram_ptr, row, i1, p, ridge) + hp
        h12 = _gram_entry(gram_ptr, row, i1, i2, ridge) - _gram_entry(gram_ptr, row, i1, p, ridge) - _gram_entry(gram_ptr, row, p, i2, ridge) + hp
        h20 = _gram_entry(gram_ptr, row, i2, i0, ridge) - _gram_entry(gram_ptr, row, i2, p, ridge) - _gram_entry(gram_ptr, row, p, i0, ridge) + hp
        h21 = _gram_entry(gram_ptr, row, i2, i1, ridge) - _gram_entry(gram_ptr, row, i2, p, ridge) - _gram_entry(gram_ptr, row, p, i1, ridge) + hp
        h22 = _gram_entry(gram_ptr, row, i2, i2, ridge) - 2.0 * _gram_entry(gram_ptr, row, i2, p, ridge) + hp
        c0 = tl.load(rhs_ptr + row * 4 + i0, mask=valid_row, other=0.0).to(tl.float32) - rp - _gram_entry(gram_ptr, row, i0, p, ridge) + hp
        c1 = tl.load(rhs_ptr + row * 4 + i1, mask=valid_row, other=0.0).to(tl.float32) - rp - _gram_entry(gram_ptr, row, i1, p, ridge) + hp
        c2 = tl.load(rhs_ptr + row * 4 + i2, mask=valid_row, other=0.0).to(tl.float32) - rp - _gram_entry(gram_ptr, row, i2, p, ridge) + hp

        safe_h00 = tl.where(tl.abs(h00) > 1.0e-12, h00, 1.0)
        one_x = c0 / safe_h00
        det2 = h00 * h11 - h01 * h10
        safe_det2 = tl.where(tl.abs(det2) > 1.0e-12, det2, 1.0)
        two_x = (c0 * h11 - h01 * c1) / safe_det2
        two_y = (h00 * c1 - h10 * c0) / safe_det2
        det3 = h00 * (h11 * h22 - h12 * h21) - h01 * (h10 * h22 - h12 * h20) + h02 * (h10 * h21 - h11 * h20)
        safe_det3 = tl.where(tl.abs(det3) > 1.0e-12, det3, 1.0)
        three_x = (c0 * (h11 * h22 - h12 * h21) - h01 * (c1 * h22 - h12 * c2) + h02 * (c1 * h21 - h11 * c2)) / safe_det3
        three_y = (h00 * (c1 * h22 - h12 * c2) - c0 * (h10 * h22 - h12 * h20) + h02 * (h10 * c2 - c1 * h20)) / safe_det3
        three_z = (h00 * (h11 * c2 - c1 * h21) - h01 * (h10 * c2 - c1 * h20) + c0 * (h10 * h21 - h11 * h20)) / safe_det3
        dimension = count - 1
        b0 = tl.where(dimension == 1, one_x, tl.where(dimension == 2, two_x, tl.where(dimension == 3, three_x, 0.0)))
        b1 = tl.where(dimension == 2, two_y, tl.where(dimension == 3, three_y, 0.0))
        b2 = tl.where(dimension == 3, three_z, 0.0)
        ap = 1.0 - b0 - b1 - b2
        a0 = tl.where(i0 == 0, b0, 0.0) + tl.where(i1 == 0, b1, 0.0) + tl.where(i2 == 0, b2, 0.0) + tl.where(p == 0, ap, 0.0)
        a1 = tl.where(i0 == 1, b0, 0.0) + tl.where(i1 == 1, b1, 0.0) + tl.where(i2 == 1, b2, 0.0) + tl.where(p == 1, ap, 0.0)
        a2 = tl.where(i0 == 2, b0, 0.0) + tl.where(i1 == 2, b1, 0.0) + tl.where(i2 == 2, b2, 0.0) + tl.where(p == 2, ap, 0.0)
        a3 = tl.where(i0 == 3, b0, 0.0) + tl.where(i1 == 3, b1, 0.0) + tl.where(i2 == 3, b2, 0.0) + tl.where(p == 3, ap, 0.0)
        # For singleton sets p == i0.  b0 is zero, so no double contribution occurs.
        stable = (count == 1) | ((dimension == 1) & (tl.abs(h00) > 1.0e-12)) | ((dimension == 2) & (tl.abs(det2) > 1.0e-12)) | ((dimension == 3) & (tl.abs(det3) > 1.0e-12))
        feasible = valid_row & stable & (a0 >= -1.0e-5) & (a1 >= -1.0e-5) & (a2 >= -1.0e-5) & (a3 >= -1.0e-5)
        a0 = tl.maximum(a0, 0.0)
        a1 = tl.maximum(a1, 0.0)
        a2 = tl.maximum(a2, 0.0)
        a3 = tl.maximum(a3, 0.0)
        alpha_sum = tl.maximum(a0 + a1 + a2 + a3, 1.0e-12)
        a0 = a0 / alpha_sum
        a1 = a1 / alpha_sum
        a2 = a2 / alpha_sum
        a3 = a3 / alpha_sum
        r0 = tl.load(rhs_ptr + row * 4, mask=valid_row, other=0.0).to(tl.float32)
        r1 = tl.load(rhs_ptr + row * 4 + 1, mask=valid_row, other=0.0).to(tl.float32)
        r2 = tl.load(rhs_ptr + row * 4 + 2, mask=valid_row, other=0.0).to(tl.float32)
        r3 = tl.load(rhs_ptr + row * 4 + 3, mask=valid_row, other=0.0).to(tl.float32)
        q0 = tl.load(gram_ptr + row * 16) * a0 + tl.load(gram_ptr + row * 16 + 1) * a1 + tl.load(gram_ptr + row * 16 + 2) * a2 + tl.load(gram_ptr + row * 16 + 3) * a3
        q1 = tl.load(gram_ptr + row * 16 + 4) * a0 + tl.load(gram_ptr + row * 16 + 5) * a1 + tl.load(gram_ptr + row * 16 + 6) * a2 + tl.load(gram_ptr + row * 16 + 7) * a3
        q2 = tl.load(gram_ptr + row * 16 + 8) * a0 + tl.load(gram_ptr + row * 16 + 9) * a1 + tl.load(gram_ptr + row * 16 + 10) * a2 + tl.load(gram_ptr + row * 16 + 11) * a3
        q3 = tl.load(gram_ptr + row * 16 + 12) * a0 + tl.load(gram_ptr + row * 16 + 13) * a1 + tl.load(gram_ptr + row * 16 + 14) * a2 + tl.load(gram_ptr + row * 16 + 15) * a3
        objective = a0 * q0 + a1 * q1 + a2 * q2 + a3 * q3 - 2.0 * (a0 * r0 + a1 * r1 + a2 * r2 + a3 * r3)
        objective = tl.where(feasible, objective, float("inf"))
        output = (row * 15 + active_set) * 4
        tl.store(candidate_ptr + output, a0, mask=valid_row)
        tl.store(candidate_ptr + output + 1, a1, mask=valid_row)
        tl.store(candidate_ptr + output + 2, a2, mask=valid_row)
        tl.store(candidate_ptr + output + 3, a3, mask=valid_row)
        tl.store(objective_ptr + row * 15 + active_set, objective, mask=valid_row)


    @triton.jit
    def _select_active_candidate_kernel(
        candidate_ptr, objective_ptr, gram_ptr, rhs_ptr, output_ptr, rows,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, 16)
        values = tl.load(
            objective_ptr + row * 15 + offsets,
            mask=(row < rows) & (offsets < 15), other=float("inf"),
        )
        best = tl.argmin(values, axis=0)
        source = (row * 15 + best) * 4
        alpha_offset = tl.arange(0, 4)
        alpha = tl.load(candidate_ptr + source + alpha_offset, mask=row < rows, other=0.0)
        alpha = alpha / tl.maximum(tl.sum(alpha, axis=0), 1.0e-12)
        rhs = tl.load(rhs_ptr + row * 4 + alpha_offset, mask=row < rows, other=0.0).to(tl.float32)
        gram_row = alpha_offset[:, None]
        gram_column = alpha_offset[None, :]
        gram = tl.load(
            gram_ptr + row * 16 + gram_row * 4 + gram_column,
            mask=row < rows, other=0.0,
        ).to(tl.float32)
        quadratic = tl.sum(gram * alpha[None, :], axis=1)
        numerator = tl.sum(alpha * rhs, axis=0)
        squared_norm = tl.sum(alpha * quadratic, axis=0)
        cosine = numerator * tl.rsqrt(tl.maximum(squared_norm, 1.0e-12))
        single = tl.load(rhs_ptr + row * 4, mask=row < rows, other=0.0).to(tl.float32)
        use_single = single > cosine
        alpha = tl.where(use_single, tl.where(alpha_offset == 0, 1.0, 0.0), alpha)
        tl.store(output_ptr + row * 4 + alpha_offset, alpha, mask=row < rows)

    @triton.jit
    def _four_support_projected_gradient_kernel(
        gram_ptr, rhs_ptr, output_ptr, rows: tl.constexpr, steps: tl.constexpr,
    ):
        row = tl.program_id(0)
        valid = row < rows
        gram_offset = row * 16
        rhs_offset = row * 4

        g00 = tl.load(gram_ptr + gram_offset + 0, mask=valid, other=0.0).to(tl.float32)
        g01 = tl.load(gram_ptr + gram_offset + 1, mask=valid, other=0.0).to(tl.float32)
        g02 = tl.load(gram_ptr + gram_offset + 2, mask=valid, other=0.0).to(tl.float32)
        g03 = tl.load(gram_ptr + gram_offset + 3, mask=valid, other=0.0).to(tl.float32)
        g10 = tl.load(gram_ptr + gram_offset + 4, mask=valid, other=0.0).to(tl.float32)
        g11 = tl.load(gram_ptr + gram_offset + 5, mask=valid, other=0.0).to(tl.float32)
        g12 = tl.load(gram_ptr + gram_offset + 6, mask=valid, other=0.0).to(tl.float32)
        g13 = tl.load(gram_ptr + gram_offset + 7, mask=valid, other=0.0).to(tl.float32)
        g20 = tl.load(gram_ptr + gram_offset + 8, mask=valid, other=0.0).to(tl.float32)
        g21 = tl.load(gram_ptr + gram_offset + 9, mask=valid, other=0.0).to(tl.float32)
        g22 = tl.load(gram_ptr + gram_offset + 10, mask=valid, other=0.0).to(tl.float32)
        g23 = tl.load(gram_ptr + gram_offset + 11, mask=valid, other=0.0).to(tl.float32)
        g30 = tl.load(gram_ptr + gram_offset + 12, mask=valid, other=0.0).to(tl.float32)
        g31 = tl.load(gram_ptr + gram_offset + 13, mask=valid, other=0.0).to(tl.float32)
        g32 = tl.load(gram_ptr + gram_offset + 14, mask=valid, other=0.0).to(tl.float32)
        g33 = tl.load(gram_ptr + gram_offset + 15, mask=valid, other=0.0).to(tl.float32)
        r0 = tl.load(rhs_ptr + rhs_offset + 0, mask=valid, other=0.0).to(tl.float32)
        r1 = tl.load(rhs_ptr + rhs_offset + 1, mask=valid, other=0.0).to(tl.float32)
        r2 = tl.load(rhs_ptr + rhs_offset + 2, mask=valid, other=0.0).to(tl.float32)
        r3 = tl.load(rhs_ptr + rhs_offset + 3, mask=valid, other=0.0).to(tl.float32)

        norm0 = tl.abs(g00) + tl.abs(g01) + tl.abs(g02) + tl.abs(g03)
        norm1 = tl.abs(g10) + tl.abs(g11) + tl.abs(g12) + tl.abs(g13)
        norm2 = tl.abs(g20) + tl.abs(g21) + tl.abs(g22) + tl.abs(g23)
        norm3 = tl.abs(g30) + tl.abs(g31) + tl.abs(g32) + tl.abs(g33)
        lipschitz = tl.maximum(tl.maximum(norm0, norm1), tl.maximum(norm2, norm3))
        step = 0.95 / tl.maximum(lipschitz, 1.0e-4)

        a0, a1, a2, a3 = 1.0, 0.0, 0.0, 0.0
        for _ in range(steps):
            p0 = a0 - step * (g00 * a0 + g01 * a1 + g02 * a2 + g03 * a3 - r0)
            p1 = a1 - step * (g10 * a0 + g11 * a1 + g12 * a2 + g13 * a3 - r1)
            p2 = a2 - step * (g20 * a0 + g21 * a1 + g22 * a2 + g23 * a3 - r2)
            p3 = a3 - step * (g30 * a0 + g31 * a1 + g32 * a2 + g33 * a3 - r3)

            # Four-value descending sorting network. Only the threshold is
            # derived from the sorted values; weights retain support order.
            s0, s1 = tl.maximum(p0, p1), tl.minimum(p0, p1)
            s2, s3 = tl.maximum(p2, p3), tl.minimum(p2, p3)
            s0, s2 = tl.maximum(s0, s2), tl.minimum(s0, s2)
            s1, s3 = tl.maximum(s1, s3), tl.minimum(s1, s3)
            s1, s2 = tl.maximum(s1, s2), tl.minimum(s1, s2)
            theta = s0 - 1.0
            theta2 = (s0 + s1 - 1.0) * 0.5
            theta = tl.where(s1 > theta2, theta2, theta)
            theta3 = (s0 + s1 + s2 - 1.0) / 3.0
            theta = tl.where(s2 > theta3, theta3, theta)
            theta4 = (s0 + s1 + s2 + s3 - 1.0) * 0.25
            theta = tl.where(s3 > theta4, theta4, theta)
            a0 = tl.maximum(p0 - theta, 0.0)
            a1 = tl.maximum(p1 - theta, 0.0)
            a2 = tl.maximum(p2 - theta, 0.0)
            a3 = tl.maximum(p3 - theta, 0.0)

        tl.store(output_ptr + rhs_offset + 0, a0, mask=valid)
        tl.store(output_ptr + rhs_offset + 1, a1, mask=valid)
        tl.store(output_ptr + rhs_offset + 2, a2, mask=valid)
        tl.store(output_ptr + rhs_offset + 3, a3, mask=valid)


def four_support_projected_gradient(
    gram: torch.Tensor, rhs: torch.Tensor, *, steps: int = 32,
) -> torch.Tensor:
    """Run the existing four-way simplex iterations in one Triton kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for the fused convex solver")
    if gram.ndim != 3 or gram.shape[1:] != (4, 4) or rhs.shape != (len(gram), 4):
        raise ValueError("the fused convex solver requires [N,4,4] Gram and [N,4] rhs")
    gram, rhs = gram.contiguous().float(), rhs.contiguous().float()
    output = torch.empty_like(rhs)
    _four_support_projected_gradient_kernel[(len(gram),)](
        gram, rhs, output, rows=len(gram), steps=steps, num_warps=1,
    )
    return output


_ACTIVE_IDS = torch.tensor([
    [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3],
    [0, 1, 1, 1], [0, 2, 2, 2], [0, 3, 3, 3],
    [1, 2, 2, 2], [1, 3, 3, 3], [2, 3, 3, 3],
    [0, 1, 2, 2], [0, 1, 3, 3], [0, 2, 3, 3], [1, 2, 3, 3],
    [0, 1, 2, 3],
], dtype=torch.int32)
_ACTIVE_COUNT = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4], dtype=torch.int32)


def streaming_top4(query: torch.Tensor, basis_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-bounded BF16 Top-4 without materializing ``query @ basis``."""
    if triton is None or not query.is_cuda:
        raise RuntimeError("streaming_top4 requires CUDA Triton")
    if query.ndim != 2 or basis_t.ndim != 2 or query.shape[1] != basis_t.shape[0]:
        raise ValueError("query [M,D] and transposed basis [D,N] are required")
    rows, channels = query.shape
    basis_count = basis_t.shape[1]
    if basis_count < 4:
        raise ValueError("streaming Top-4 requires at least four bases")
    query = query.contiguous().to(torch.bfloat16)
    basis_t = basis_t.contiguous().to(torch.bfloat16)
    values = torch.empty((rows, 4), device=query.device, dtype=torch.float32)
    indices = torch.empty((rows, 4), device=query.device, dtype=torch.int32)
    block_m, block_n, block_k = 16, 32, 32
    _streaming_top4_kernel[(triton.cdiv(rows, block_m),)](
        query, basis_t, values, indices, rows, basis_count,
        channels=channels, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4, num_stages=1,
    )
    return values, indices.long()


def four_support_active_set(gram: torch.Tensor, rhs: torch.Tensor, *, ridge: float = 1e-4) -> torch.Tensor:
    """Exact 15-active-set simplex least-squares solve in two Triton launches."""
    if triton is None or not gram.is_cuda:
        raise RuntimeError("four_support_active_set requires CUDA Triton")
    if gram.ndim != 3 or gram.shape[1:] != (4, 4) or rhs.shape != (len(gram), 4):
        raise ValueError("active-set solver requires [N,4,4] Gram and [N,4] rhs")
    gram = gram.contiguous().float()
    rhs = rhs.contiguous().float()
    rows = len(gram)
    active_ids = _ACTIVE_IDS.to(device=gram.device)
    active_count = _ACTIVE_COUNT.to(device=gram.device)
    candidates = torch.empty((rows, 15, 4), device=gram.device, dtype=torch.float32)
    objectives = torch.empty((rows, 15), device=gram.device, dtype=torch.float32)
    _four_support_active_candidates_kernel[(rows * 15,)](
        gram, rhs, active_ids, active_count, candidates, objectives, rows,
        ridge=float(ridge), num_warps=1,
    )
    output = torch.empty((rows, 4), device=gram.device, dtype=torch.float32)
    _select_active_candidate_kernel[(rows,)](
        candidates, objectives, gram, rhs, output, rows, num_warps=1,
    )
    return output
