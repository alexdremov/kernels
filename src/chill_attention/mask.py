import torch

import inspect
import types
from typing import get_type_hints

import triton
import triton.language as tl

from abc import ABC
from collections import defaultdict


class ChillMask(ABC):
    def __init__(self, constexprs=None, max_pos=10):
        constexprs = constexprs or dict

        self.mask = triton.heuristics(constexprs)(self.mask)

        if max_pos is not None:
            self.verify(max_pos)

    def _make_meta_jit_function(self, name, args, constexprs):
        args_string = ', '.join(args)
        constexprs_string = ""
        constexprs_assignments = ""
        if len(constexprs) > 0:
            constexprs_string = ", "
            constexprs_string += ", ".join(
                f"{k}: tl.constexpr"
                for k in constexprs
            )
            constexprs_assignments = ", "
            constexprs_assignments += ", ".join(
                f"{k}={k}"
                for k in constexprs
            )
        class_name = type(self).__name__
        result = (
            "@triton.jit\n"
            f"def _{name}({args_string}{constexprs_string}):"
            f"    return {class_name}.{name}({args_string}{constexprs_assignments})"
        )

        namespace = {}
        method = getattr(self, name)
        exec(result, method.__globals__, namespace)
        meta_method = namespace[f"_{name}"]
        meta_method.__kwdefaults__ = meta_method.__kwdefaults__ or dict()
        meta_method.__kwdefaults__ |= constexprs

        meta_method = staticmethod(meta_method)
        setattr(self, f"_{name}", meta_method)

    @staticmethod
    @triton.jit
    def mask(q: tl.tensor, k: tl.tensor) -> tl.tensor:
        return tl.full((q.shape[0], k.shape[0]), True, dtype=tl.int1)

    @staticmethod
    @triton.jit
    def q_range_for_k(k: int) -> tuple[tl.tensor, tl.tensor]:
        return 0, -1

    @staticmethod
    @triton.jit
    def k_range_for_q(q: int) -> tuple[tl.tensor, tl.tensor]:
        return 0, -1

    @staticmethod
    @triton.jit
    def _mask_infer(
        CHUNK_SIZE: tl.constexpr,
        result: torch.Tensor,
        result_stride_q: int,
        result_stride_k: int,
        mask,
    ):
        q_start, k_start = (
            tl.program_id(axis=0) * CHUNK_SIZE,
            tl.program_id(axis=1) * CHUNK_SIZE,
        )

        q_pos = q_start + tl.arange(0, CHUNK_SIZE)
        k_pos = k_start + tl.arange(0, CHUNK_SIZE)

        mask_result = mask(q_pos, k_pos)

        assert mask_result.dtype == tl.int1
        assert mask_result.shape[0] == mask_result.shape[1]
        assert mask_result.shape[0] == CHUNK_SIZE

        chunks_count = tl.num_programs(0)
        result_ptr = tl.make_block_ptr(
            base=result,
            shape=(chunks_count * CHUNK_SIZE, chunks_count * CHUNK_SIZE),
            strides=(result_stride_q, result_stride_k),
            offsets=(q_start, k_start),
            block_shape=(CHUNK_SIZE, CHUNK_SIZE),
            order=(1, 0)
        )
        tl.store(result_ptr, mask_result.to(tl.int8))


    @staticmethod
    @triton.jit
    def _limits_infer(
        CHUNK_SIZE: tl.constexpr,
        result: torch.Tensor,
        result_stride_i: int,
        result_stride_pos: int,
        rule,
    ):
        i_start = tl.program_id(axis=0) * CHUNK_SIZE

        lower_limits = tl.zeros((CHUNK_SIZE, ), dtype=tl.int32)
        upper_limits = tl.zeros((CHUNK_SIZE, ), dtype=tl.int32)

        for i in tl.range(0, CHUNK_SIZE):
            mask = tl.arange(0, CHUNK_SIZE) == i
            start, end = rule(i_start + i)

            lower_limits = tl.where(mask, start, lower_limits)
            upper_limits = tl.where(mask, end, upper_limits)

        limits = tl.join(lower_limits, upper_limits)

        chunks_count = tl.num_programs(0)
        result_ptr = tl.make_block_ptr(
            base=result,
            shape=(chunks_count * CHUNK_SIZE, chunks_count * CHUNK_SIZE),
            strides=(result_stride_i, result_stride_pos),
            offsets=(i_start, 0),
            block_shape=(CHUNK_SIZE, 2),
            order=(1, 0)
        )
        tl.store(result_ptr, limits.to(tl.int32))

    def make_mask(self, max_pos):
        chunk_size = 32
        chunks = triton.cdiv(max_pos, chunk_size)
        total_pos = chunk_size * chunks

        mask = torch.empty(
            (total_pos, total_pos),
            device='cuda',
            dtype=torch.bool,
        )
        self._mask_infer[(chunks, chunks)](
            CHUNK_SIZE=chunk_size,
            result=mask,
            result_stride_q=mask.stride(0),
            result_stride_k=mask.stride(1),
            mask=self.mask
        )
        return mask

    def verify(self, max_pos):
        chunk_size = 32
        chunks = triton.cdiv(max_pos, chunk_size)
        total_pos = chunk_size * chunks

        mask = self.make_mask(total_pos).cpu()

        q_lims_for_k = torch.zeros(
            (total_pos, 2),
            device='cuda',
            dtype=torch.int32,
        )
        self._limits_infer[(chunks, 1)](
            CHUNK_SIZE=chunk_size,
            result=q_lims_for_k,
            result_stride_i=q_lims_for_k.stride(0),
            result_stride_pos=q_lims_for_k.stride(1),
            rule=self.q_range_for_k
        )
        q_lims_for_k = torch.where(
            q_lims_for_k == -1,
            total_pos - 1,
            q_lims_for_k,
        )
        q_lims_for_k = q_lims_for_k.cpu()

        k_lims_for_q = torch.zeros(
            (total_pos, 2),
            device='cuda',
            dtype=torch.int32,
        )
        self._limits_infer[(chunks, 1)](
            CHUNK_SIZE=chunk_size,
            result=k_lims_for_q,
            result_stride_i=k_lims_for_q.stride(0),
            result_stride_pos=k_lims_for_q.stride(1),
            rule=self.k_range_for_q
        )
        k_lims_for_q = torch.where(
            k_lims_for_q == -1,
            total_pos - 1,
            k_lims_for_q,
        )
        k_lims_for_q = k_lims_for_q.cpu()

        k_lims_for_q_real = defaultdict(lambda: (None, None))
        q_lims_for_k_real = defaultdict(lambda: (None, None))

        def update_lims(prev_lims, new_value):
            return (
                new_value if prev_lims[0] is None else min(prev_lims[0], new_value),
                new_value if prev_lims[1] is None else max(prev_lims[1], new_value)
            )

        for q in range(total_pos):
            for k in range(total_pos):
                if not mask[q, k]:
                    continue

                k_lims_for_q_real[q] = update_lims(k_lims_for_q_real[q], k)
                q_lims_for_k_real[k] = update_lims(q_lims_for_k_real[k], q)

        for q in k_lims_for_q_real:
            real = torch.tensor(k_lims_for_q_real[q])
            analytical = k_lims_for_q[q]
            match = (real == analytical).all().item()
            assert match, (
                "Mismatch real vs analytical k lims for q position. "
                f"{real = }, {analytical = }"
            )

        for k in q_lims_for_k_real:
            real = torch.tensor(q_lims_for_k_real[k])
            analytical = q_lims_for_k[k]
            match = (real == analytical).all().item()
            assert match, (
                "Mismatch real vs analytical q lims for k position. "
                f"{real = }, {analytical = }"
            )

class CausalChillMask(ChillMask):
    @staticmethod
    @triton.jit
    def mask(q: tl.tensor, k: tl.tensor) -> tl.tensor:
        return q[:, None] >= k[None, :]

    @staticmethod
    @triton.jit
    def q_range_for_k(k: int) -> tuple[tl.tensor, tl.tensor]:
        return k, -1

    @staticmethod
    @triton.jit
    def k_range_for_q(q: int) -> tuple[tl.tensor, tl.tensor]:
        return 0, q


class ChunkwiseChillMask(ChillMask):
    @staticmethod
    @triton.jit
    def mask(
        q: tl.tensor,
        k: tl.tensor,
        context_size: tl.constexpr,
        back_contexts: tl.constexpr,
    ) -> tl.tensor:
        return q[:, None] >= k[None, :]

    @staticmethod
    @triton.jit
    def q_range_for_k(
        k: int,
    ) -> tuple[tl.tensor, tl.tensor]:
        return k, -1

    @staticmethod
    @triton.jit
    def k_range_for_q(
        q: int,
    ) -> tuple[tl.tensor, tl.tensor]:
        return 0, q


if __name__ == "__main__":
    full_mask = ChunkwiseChillMask(10)
