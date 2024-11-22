import os
import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


def triton_no_tests_autotune(autotune, heuristics):
    def wrapper(fn):
        if "PYTEST_VERSION" in os.environ:
            return triton.heuristics(heuristics)(fn)
        return triton.autotune(**autotune)(fn)

    return wrapper


MAX_TILE_SIZE = 256
MIN_TILE_SIZE = 32


def strides(t):
    return [t.stride(i) for i in range(t.ndim)]


def fwd_configs_pruner(configs, nargs, CONTEXT_SIZE, HEAD_DIM, **kwargs):
    min_size = min(CONTEXT_SIZE, 64)
    max_size = CONTEXT_SIZE * 4
    min_pipeline, max_pipeline = 1, 3

    if HEAD_DIM == 64:
        min_pipeline = 2
    elif HEAD_DIM == 128:
        max_size = 128
        min_size = 32
        max_pipeline = 2
    elif HEAD_DIM == 256:
        max_size = 128
        min_size = 32
        max_pipeline = 1

    configs = [i for i in configs if min_size <= i.kwargs['TILE_K_SIZE'] <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs['TILE_Q_SIZE'] <= max_size]
    configs = [i for i in configs if min_pipeline <= i.kwargs['PIPELINING'] <= max_pipeline]
    print(f"Start benchmarking {len(configs) = }")
    return configs

@triton_no_tests_autotune(
    autotune=dict(
        configs=[
            triton.Config(
                dict(
                    PIPELINING=pipe,
                    TILE_Q_SIZE=tile_q,
                    TILE_K_SIZE=tile_k,
                ),
                num_warps=num_warps,
                num_stages=pipe,
            )
            for num_warps in [4, 8]
            for pipe in [1, 2, 3]
            for tile_q in [2 ** i for i in range(int(math.log2(MIN_TILE_SIZE) + 0.1), int(math.log2(MAX_TILE_SIZE) + 0.1) + 1)]
            for tile_k in [2 ** i for i in range(int(math.log2(MIN_TILE_SIZE) + 0.1), int(math.log2(MAX_TILE_SIZE) + 0.1) + 1)]
        ],
        key=["HEAD_DIM", "CONTEXT_SIZE", "CONTEXTS_BACK", "INPUT_PRECISION", "TIME_BUCKET", "DTYPE"],
        prune_configs_by=dict(
            early_config_prune=fwd_configs_pruner
        )
    ),
    heuristics=dict(
        PIPELINING=lambda _: 1,
        TILE_Q_SIZE=lambda args: min(
            64, max(MIN_TILE_SIZE, triton.next_power_of_2(args['CONTEXT_SIZE']))
        ),
        TILE_K_SIZE=lambda args: min(
            64, max(MIN_TILE_SIZE, triton.next_power_of_2(args['CONTEXT_SIZE']))
        ),
    ),
)
@triton.heuristics(
    dict(
        Q_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_Q_SIZE'] == 0,
        K_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_K_SIZE'] == 0,
        PERFECT_MATCHING=lambda args : args['TILE_K_SIZE'] == args['TILE_Q_SIZE'] and args['TILE_Q_SIZE'] == args['CONTEXT_SIZE'],
        RCP_LN2=lambda _: math.log2(math.e),
    )
)
@triton.jit
def _streaming_attn_fwd(
    Q: tl.tensor, Kt: tl.tensor, V: tl.tensor, L: tl.tensor, #
    LSE: tl.tensor, O: tl.tensor,  #
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,  #
    stride_kb: int, stride_kh: int, stride_kk: int, stride_kt: int,  #
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,  #
    stride_mb: int, stride_mh: int, stride_mt: int,  #
    stride_ob: int, stride_oh: int, stride_ot: int, stride_ok: int, #
    lens_stride: int,
    T: int,  #
    HEAD_DIM: tl.constexpr,  #
    CONTEXT_SIZE: tl.constexpr,  #
    CONTEXTS_BACK: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    DTYPE:  tl.constexpr,  #
    TIME_BUCKET:  tl.constexpr,  #
    PRESCALE_QK: tl.constexpr,  #
    OUTPUT_LOGSUMEXP: tl.constexpr,  #
    TILE_Q_SIZE: tl.constexpr,  #
    TILE_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
    Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    K_BLOCK_DIVISIBLE: tl.constexpr,  #
    PERFECT_MATCHING: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    q_tile_idx = tl.program_id(2)
    q_token_idx = q_tile_idx * TILE_Q_SIZE

    seq_len = tl.load(L + batch * lens_stride)
    seq_len = min(seq_len, T)
    if seq_len <= q_token_idx:
        return

    qbatch_head_offset = batch * stride_qb + head * stride_qh
    q_tile_ptr = tl.make_block_ptr(
        base=Q + qbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_qt, stride_qk),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(0, 1),
    )

    kbatch_head_offset = batch * stride_kb + head * stride_kh
    k_tile_ptr = tl.make_block_ptr(
        base=Kt + kbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_kk, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_K_SIZE),
        order=(1, 0),
    )

    vbatch_head_offset = batch * stride_vb + head * stride_vh
    v_tile_ptr = tl.make_block_ptr(
        base=V + vbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_vt, stride_vk),
        offsets=(0, 0),
        block_shape=(TILE_K_SIZE, HEAD_DIM),
        order=(0, 1),
    )

    m_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32)
    acc = tl.zeros([TILE_Q_SIZE, HEAD_DIM], dtype=tl.float32)
    if not PERFECT_MATCHING:
        q_attended = tl.zeros([TILE_Q_SIZE], dtype=tl.int1) > 0

    q_tile_min_context = q_token_idx // CONTEXT_SIZE
    kv_start_tile_idx = max(
        0, ((q_tile_min_context - CONTEXTS_BACK) * CONTEXT_SIZE)
    ) // TILE_K_SIZE

    q_tile_max_token = min(q_token_idx + TILE_Q_SIZE, seq_len)
    q_tile_max_context = (q_tile_max_token - 1) // CONTEXT_SIZE
    kv_end_tile_idx = tl.cdiv(
        min((q_tile_max_context + 1) * CONTEXT_SIZE, seq_len), TILE_K_SIZE
    )

    q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)

    if not PERFECT_MATCHING:
        q_context_indices = q_tile_indices // CONTEXT_SIZE

    if Q_BLOCK_DIVISIBLE:
        q_tile = tl.load(q_tile_ptr)
    else:
        q_tile = tl.load(
            q_tile_ptr,
            boundary_check=(0,),
        )

    # lg2(e) temperatire adjustment
    softmax_scale: tl.constexpr = tl.cast((HEAD_DIM**-0.5) * RCP_LN2, q_tile.dtype)

    for kv_tile_idx in tl.range(
        kv_start_tile_idx, kv_end_tile_idx, num_stages=PIPELINING
    ):
        kv_token_idx = kv_tile_idx * TILE_K_SIZE

        if K_BLOCK_DIVISIBLE:
            k_tile = tl.load(
                tl.advance(k_tile_ptr, (0, kv_token_idx)),
            )
            v_tile = tl.load(
                tl.advance(v_tile_ptr, (kv_token_idx, 0)),
            )
        else:
            k_tile = tl.load(
                tl.advance(k_tile_ptr, (0, kv_token_idx)),
                boundary_check=(1,),
            )
            v_tile = tl.load(
                tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                boundary_check=(0,),
            )

        if PRESCALE_QK:
            q_tile = q_tile * softmax_scale
        qk = tl.dot(
            q_tile, k_tile, input_precision=INPUT_PRECISION, out_dtype=tl.float32
        )

        kv_indices = kv_token_idx + tl.arange(0, TILE_K_SIZE)
        mask = (
            kv_indices[None, :] < seq_len
        ) & (
            q_tile_indices[:, None] < seq_len
        )
        if not PERFECT_MATCHING:
            kv_context_indices = kv_indices // CONTEXT_SIZE
            blocks_diff = q_context_indices[:, None] - kv_context_indices[None, :]
            streaming_mask = (blocks_diff >= 0) & (blocks_diff <= CONTEXTS_BACK)
            mask &= streaming_mask

            q_attended |= tl.max(mask, 1) > 0

        if not PRESCALE_QK:
            qk = qk * softmax_scale
        qk = tl.where(mask, qk, tl.cast(-float("inf"), qk.dtype))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        if not PERFECT_MATCHING:
            m_ij_safe = tl.where(q_attended, m_ij, tl.cast(0, m_ij.dtype))
        else:
            m_ij_safe = m_ij
        p = tl.math.exp2(qk - m_ij_safe[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij_safe)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        acc = tl.dot(
            p.to(v_tile.dtype),
            v_tile,
            acc,
            input_precision=INPUT_PRECISION,
            out_dtype=tl.float32,
        )
        m_i = m_ij

    if not PERFECT_MATCHING:
        l_i = tl.where(q_attended, l_i, 1)
        acc = acc / l_i[:, None]
    else:
        acc = acc / l_i[:, None]
        seq_mask = q_tile_indices < seq_len
        acc = tl.where(seq_mask[:, None], acc, 0.0)


    obatch_head_offset = batch * stride_ob + head * stride_oh
    o_tile_ptr = tl.make_block_ptr(
        base=O + obatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_ot, stride_ok),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    if Q_BLOCK_DIVISIBLE:
        tl.store(
            o_tile_ptr,
            acc.to(o_tile_ptr.type.element_ty),
        )
    else:
        tl.store(
            o_tile_ptr,
            acc.to(o_tile_ptr.type.element_ty),
            boundary_check=(0,),
        )

    if OUTPUT_LOGSUMEXP:
        m_i += tl.math.log2(l_i)

        mbatch_head_offset = batch * stride_mb + head * stride_mh
        m_tile_ptr = tl.make_block_ptr(
            base=LSE + mbatch_head_offset,
            shape=(T,),
            strides=(stride_mt,),
            offsets=(q_token_idx,),
            block_shape=(TILE_Q_SIZE,),
            order=(0,),
        )

        if Q_BLOCK_DIVISIBLE:
            tl.store(
                m_tile_ptr,
                m_i,
            )
        else:
            tl.store(
                m_tile_ptr,
                m_i,
                boundary_check=(0,),
            )


@triton.heuristics(
    dict(
        TILE_DQ_Q_SIZE=lambda _: 16,
        TILE_DQ_K_SIZE=lambda _: 16,
        TILE_DK_Q_SIZE=lambda _: 16,
        TILE_DK_K_SIZE=lambda _: 16,

    )
)
@triton.heuristics(
    dict(
        RCP_LN2=lambda _: math.log2(math.e),
        DQ_TILES_NUM=lambda args: triton.cdiv(args['T'], args["TILE_DQ_Q_SIZE"]),
    )
)
@triton.jit
def _streaming_attn_bwd(
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, L: tl.tensor, #
    DELTA: tl.tensor, LSE: tl.tensor, O: tl.tensor,  #
    DO: tl.tensor, DQ: tl.tensor, DK: tl.tensor, DV: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,  #
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: int,  #
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,  #
    stride_deltab: int, stride_deltah: int, stride_deltat: int,  #
    stride_mb: int, stride_mh: int, stride_mt: int,  #
    stride_ob: int, stride_oh: int, stride_ot: int, stride_ok: int,  #
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: int,  #
    stride_dqb: int, stride_dqh: int, stride_dqt: int, stride_dqk: int,  #
    stride_dkb: int, stride_dkh: int, stride_dkt: int, stride_dkk: int,  #
    stride_dvb: int, stride_dvh: int, stride_dvt: int, stride_dvk: int,  #
    lens_stride: int,
    T: int,  #
    BATCH: int,  #
    HEAD_DIM: tl.constexpr,  #
    CONTEXT_SIZE: tl.constexpr,  #
    CONTEXTS_BACK: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    TILE_DQ_Q_SIZE: tl.constexpr, TILE_DQ_K_SIZE: tl.constexpr,  #
    TILE_DK_Q_SIZE: tl.constexpr, TILE_DK_K_SIZE: tl.constexpr,  #
    DQ_TILES_NUM: tl.constexpr,  #
    PRESCALE_QK: tl.constexpr,  #
    DTYPE: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    dkv_worker = tl.program_id(2) >= DQ_TILES_NUM
    tile_id = tl.program_id(2) - (DQ_TILES_NUM * dkv_worker)

    seq_len = tl.load(L + batch * lens_stride)
    seq_len = min(seq_len, T)

    if dkv_worker:
        kv_tile_idx = tile_id
        kv_token_idx = kv_tile_idx * TILE_DK_K_SIZE

        qbatch_head_offset = batch * stride_qb + head * stride_qh
        qt_tile_ptr = tl.make_block_ptr(
            base=Q + qbatch_head_offset,
            shape=(HEAD_DIM, T),
            strides=(stride_qk, stride_qt),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, TILE_DK_Q_SIZE),
            order=(1, 0),
        )

        kbatch_head_offset = batch * stride_kb + head * stride_kh
        k_tile_ptr = tl.make_block_ptr(
            base=K + kbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_kt, stride_kk),
            offsets=(kv_token_idx, 0),
            block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
            order=(0, 1),
        )

        vbatch_head_offset = batch * stride_vb + head * stride_vh
        v_tile_ptr = tl.make_block_ptr(
            base=V + vbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_vt, stride_vk),
            offsets=(kv_token_idx, 0),
            block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
            order=(0, 1),
        )

        dobatch_head_offset = batch * stride_dob + head * stride_doh
        do_tile_ptr = tl.make_block_ptr(
            base=DO + dobatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_dot, stride_dok),
            offsets=(0, 0),
            block_shape=(TILE_DK_Q_SIZE, HEAD_DIM),
            order=(0, 1),
        )

        lsebatch_head_offset = batch * stride_mb + head * stride_mh
        lse_tile_ptr = tl.make_block_ptr(
            base=LSE + lsebatch_head_offset,
            shape=(T,),
            strides=(stride_mt,),
            offsets=(0,),
            block_shape=(TILE_DK_Q_SIZE,),
            order=(0,),
        )

        deltabatch_head_offset = batch * stride_deltab + head * stride_deltah
        delta_tile_ptr = tl.make_block_ptr(
            base=DELTA + deltabatch_head_offset,
            shape=(T,),
            strides=(stride_deltat,),
            offsets=(0,),
            block_shape=(TILE_DK_Q_SIZE,),
            order=(0,),
        )

        dv = tl.zeros([TILE_DK_K_SIZE, HEAD_DIM], dtype=tl.float32)
        dk = tl.zeros([TILE_DK_K_SIZE, HEAD_DIM], dtype=tl.float32)

        k = tl.load(
            k_tile_ptr,
            boundary_check=(0,),
            padding_option='zero'
        )
        v = tl.load(
            v_tile_ptr,
            boundary_check=(0,),
            padding_option='zero'
        )

        dk, dv = _streaming_attn_bwd_dkdv(
            dk, dv,
            qt_tile_ptr, do_tile_ptr, lse_tile_ptr, delta_tile_ptr,
            k, v,
            seq_len=seq_len,
            kv_token_idx=kv_token_idx,
            HEAD_DIM=HEAD_DIM,
            CONTEXT_SIZE=CONTEXT_SIZE,
            CONTEXTS_BACK=CONTEXTS_BACK,
            TILE_Q_SIZE=TILE_DK_Q_SIZE,
            TILE_K_SIZE=TILE_DK_K_SIZE,
            INPUT_PRECISION=INPUT_PRECISION,
            RCP_LN2=RCP_LN2,
        )

        dkbatch_head_offset = batch * stride_dkb + head * stride_dkh
        dk_tile_ptr = tl.make_block_ptr(
            base=DK + dkbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_dkt, stride_dkk),
            offsets=(kv_token_idx, 0),
            block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
            order=(0, 1),
        )
        tl.store(dk_tile_ptr, dk.to(dk_tile_ptr.type.element_ty), boundary_check=(0,))

        dvbatch_head_offset = batch * stride_dvb + head * stride_dvh
        dv_tile_ptr = tl.make_block_ptr(
            base=DV + dvbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_dvt, stride_dvk),
            offsets=(kv_token_idx, 0),
            block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
            order=(0, 1),
        )
        tl.store(dv_tile_ptr, dv.to(dv_tile_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def _streaming_attn_bwd_dkdv(
    dk, dv,
    qt_tile_ptr, do_tile_ptr, lse_tile_ptr, delta_tile_ptr,
    k, v,
    seq_len,
    kv_token_idx,
    HEAD_DIM: tl.constexpr,
    CONTEXT_SIZE: tl.constexpr,
    CONTEXTS_BACK: tl.constexpr,
    TILE_Q_SIZE: tl.constexpr,
    TILE_K_SIZE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    RCP_LN2,
):
    kv_tile_min_context = kv_token_idx // CONTEXT_SIZE
    q_start_tile_idx = (kv_tile_min_context * CONTEXT_SIZE) // TILE_Q_SIZE

    kv_tile_max_token = min(kv_token_idx + TILE_K_SIZE, seq_len) - 1
    kv_tile_max_context = kv_tile_max_token // CONTEXT_SIZE
    q_end_tile_idx = tl.cdiv(
        min((kv_tile_max_context + CONTEXTS_BACK + 1) * CONTEXT_SIZE, seq_len),
        TILE_Q_SIZE,
    )

    kv_indices = kv_token_idx + tl.arange(0, TILE_K_SIZE)
    kv_context_indices = kv_indices // CONTEXT_SIZE

    softmax_scale: tl.constexpr = tl.cast((HEAD_DIM**-0.5), dk.dtype)

    for q_tile_idx in tl.range(q_start_tile_idx, q_end_tile_idx):
        q_token_idx = q_tile_idx * TILE_Q_SIZE
        qT = tl.load(
            tl.advance(qt_tile_ptr, (0, q_token_idx)),
            boundary_check=(1,),
            padding_option='zero',
        )
        m = tl.load(
            tl.advance(lse_tile_ptr, (q_token_idx,)),
            boundary_check=(0,),
            padding_option='zero',
        )
        tl.static_assert(m.dtype == tl.float32)

        qkT = tl.dot(k, qT, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        pT = tl.math.exp2(qkT - m[None, :])

        q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)
        q_context_indices = q_tile_indices // CONTEXT_SIZE
        mask = (
            kv_indices[:, None] < seq_len
        ) & (
            q_tile_indices[None, :] < seq_len
        )
        blocks_diff = q_context_indices[None, :] - kv_context_indices[:, None]
        streaming_mask = (blocks_diff >= 0) & (blocks_diff <= CONTEXTS_BACK)
        mask &= streaming_mask
        pT = tl.where(mask, pT, 0.0)

        do = tl.load(
            tl.advance(do_tile_ptr, (q_token_idx, 0)),
            boundary_check=(0,),
            padding_option='zero',
        )

        dv = tl.dot(pT.to(do.dtype), do, dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(
            tl.advance(delta_tile_ptr, (q_token_idx,)),
            boundary_check=(0,),
            padding_option='zero',
        )
        tl.static_assert(Di.dtype == tl.float32)

        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dk = tl.dot(dsT.to(qT.dtype), tl.trans(qT), dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    dk *= softmax_scale
    return dk, dv


class StreamingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lens: torch.Tensor | None, context_size: int, back_contexts: int):
        batch, heads, T, HEAD_DIM = q.shape

        assert back_contexts >= 0 and context_size >= 1
        assert HEAD_DIM in {16, 32, 64, 128, 256}
        assert HEAD_DIM == k.shape[-1] and HEAD_DIM == v.shape[-1]
        assert T == k.shape[-2] and T == v.shape[-2]

        if lens is None:
            lens = torch.tensor([T], dtype=torch.int32, device=q.device).repeat(batch)
        else:
            lens = torch.as_tensor(lens, dtype=torch.int32)

        assert lens is None or (lens.dtype == torch.int32 and batch == len(lens) and lens.ndim == 1)

        O = torch.zeros_like(q, memory_format=torch.contiguous_format)
        LSE = torch.zeros(q.shape[:3], dtype=torch.float32, device=q.device)

        grid = lambda args: (
            batch,
            heads,
            triton.cdiv(T, args["TILE_Q_SIZE"]),
        )

        need_grad = any(i.requires_grad for i in (q, k, v))

        kt = k.transpose(-1, -2)  # just stride tricks, same data
        _streaming_attn_fwd[grid](
            q, kt, v, lens, #
            LSE, O,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            kt.stride(0), kt.stride(1), kt.stride(2), kt.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            LSE.stride(0), LSE.stride(1), LSE.stride(2),  #
            stride_ob=O.stride(0), stride_oh=O.stride(1), stride_ot=O.stride(2), stride_ok=O.stride(3),  #
            lens_stride=lens.stride(0),
            T=T,
            HEAD_DIM=HEAD_DIM,  #
            CONTEXT_SIZE=context_size,
            CONTEXTS_BACK=back_contexts,
            INPUT_PRECISION=(
                "tf32" if torch.get_float32_matmul_precision() != "highest" else "ieee"
            ),
            PRESCALE_QK=True,
            DTYPE=q.dtype,
            TIME_BUCKET=triton.next_power_of_2(T),
            OUTPUT_LOGSUMEXP=need_grad,
        )

        ctx.save_for_backward(q, k, v, O, LSE, lens)
        ctx.grid = grid
        ctx.context_size = context_size
        ctx.back_contexts = back_contexts
        ctx.HEAD_DIM = HEAD_DIM
        return O

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, lens = ctx.saved_tensors
        batch, heads, T, HEAD_DIM = q.shape

        delta = (o * do.float()).sum(-1)

        DQ = torch.zeros_like(q, memory_format=torch.contiguous_format)
        DK = torch.zeros_like(k, memory_format=torch.contiguous_format)
        DV = torch.zeros_like(v, memory_format=torch.contiguous_format)

        grid = lambda args: (
            batch,
            heads,
            triton.cdiv(T, args["TILE_DQ_Q_SIZE"]) + triton.cdiv(T, args["TILE_DK_K_SIZE"]),
        )

        _streaming_attn_bwd[grid](
            q, k, v, lens,
            delta, lse, o,
            do, DQ, DK, DV,
            *strides(q),
            *strides(k),
            *strides(v),
            *strides(delta),
            *strides(lse),
            *strides(o),
            *strides(do),
            *strides(DQ),
            *strides(DK),
            *strides(DV),
            *strides(lens),
            T=T,
            BATCH=batch,
            HEAD_DIM=HEAD_DIM,
            CONTEXT_SIZE=ctx.context_size,
            CONTEXTS_BACK=ctx.back_contexts,
            INPUT_PRECISION=(
                "tf32" if torch.get_float32_matmul_precision() != "highest" else "ieee"
            ),
            DTYPE=q.dtype,
            PRESCALE_QK=True,
        )

        return DQ, DK, DV, None, None, None


def streaming_attention_reference(q, k, v, context_size, back_contexts, lens):
    block_size = context_size
    left_context_blocks_count = back_contexts + 1
    T = q.shape[-2]

    block_idxes = torch.div(torch.arange(T), block_size, rounding_mode="floor")
    block_idxes_diff = block_idxes.unsqueeze(1) - block_idxes.unsqueeze(0)
    attn_mask = (block_idxes_diff >= 0) & (block_idxes_diff < left_context_blocks_count)
    attn_mask = attn_mask.cuda()

    if lens is not None:
        key_padding_mask = (
            torch.arange(T, device="cuda").unsqueeze(0) < lens.unsqueeze(-1)
        ).unsqueeze(-1)
        key_padding_mask_ref = key_padding_mask
        key_padding_mask = key_padding_mask & key_padding_mask.transpose(-1, -2)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) & key_padding_mask.unsqueeze(1)
        res_mask = key_padding_mask_ref.unsqueeze(1)
    else:
        res_mask = torch.tensor([True], device="cuda")

    sparsity_fraction = attn_mask.sum().item() / attn_mask.numel()
    return (
        F.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attn_mask
        ) * res_mask,
        res_mask,
        sparsity_fraction,
    )


streaming_attention = StreamingAttention.apply


if __name__ == "__main__":
    import sys

    sys.path.insert(
        0,
        f"{os.path.dirname(os.path.realpath(__file__))}/../../"
    )
    sys.path.insert(
        0,
        f"{os.path.dirname(os.path.realpath(__file__))}/../"
    )

    B, H, T, D = 7, 1, 1, 128
    context, back = 10, 9

    from tests.test_streaming_attention import test_op

    test_op(
        B, H, T, D, context, back, dtype=torch.float16, lens='none'
    )
