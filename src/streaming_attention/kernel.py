import logging
import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

MAX_TILE_SIZE = 256
MIN_TILE_SIZE = 32


logger = logging.getLogger(__name__)


def strides(t):
    return [t.stride(i) for i in range(t.ndim)]


def fwd_configs_pruner(configs, nargs, CONTEXT_SIZE, HEAD_DIM, **kwargs):
    min_size = min(CONTEXT_SIZE, 64)
    max_size = CONTEXT_SIZE * 4
    min_pipeline, max_pipeline = 1, 3
    min_warps, max_warps = 1, 8

    if HEAD_DIM == 64:
        min_pipeline = 2
    elif HEAD_DIM == 128:
        max_size = 128
        min_size = 32
        max_pipeline = 2
        max_warps = 4
    elif HEAD_DIM == 256:
        max_size = 128
        min_size = 32
        max_pipeline = 1
        max_warps = 4

    configs = [i for i in configs if min_size <= i.kwargs['TILE_K_SIZE'] <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs['TILE_Q_SIZE'] <= max_size]
    configs = [i for i in configs if min_pipeline <= i.kwargs['PIPELINING'] <= max_pipeline]
    configs = [i for i in configs if min_warps <= i.num_warps <= max_warps]
    logger.warning(f"Start benchmarking forward streaming_attention {len(configs) = }")
    return configs


def bwd_configs_pruner(configs, nargs, CONTEXT_SIZE, HEAD_DIM, **kwargs):
    min_size = min(CONTEXT_SIZE, 64)
    max_size = CONTEXT_SIZE * 4
    min_pipeline, max_pipeline = 1, 3
    min_warps, max_warps = 1, 8

    if HEAD_DIM == 32 or HEAD_DIM == 16:
        min_pipeline = 3
        max_size = 64
    if HEAD_DIM == 64:
        min_pipeline = 3
        max_size = 128
    elif HEAD_DIM == 128:
        max_size = 128
        min_size = 64
        max_pipeline = 2
        min_pipeline = 2
        max_warps = 4
    elif HEAD_DIM == 256:
        max_size = 64
        min_size = 32
        max_pipeline = 2
        min_pipeline = 2
        max_warps = 4

    configs = [i for i in configs if min_size <= i.kwargs['TILE_DQ_Q_SIZE'] <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs['TILE_DQ_K_SIZE'] <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs['TILE_DK_Q_SIZE'] <= max_size]
    configs = [i for i in configs if min_size <= i.kwargs['TILE_DK_K_SIZE'] <= max_size]
    configs = [i for i in configs if min_pipeline <= i.kwargs['PIPELINING'] <= max_pipeline]
    configs = [i for i in configs if min_warps <= i.num_warps <= max_warps]
    logger.warning(f"Start benchmarking backward streaming_attention {len(configs) = }")
    return configs


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
    SM_SCALE: tl.constexpr,  #
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
        order=(1, 0),
    )

    kbatch_head_offset = batch * stride_kb + head * stride_kh
    kt_tile_ptr = tl.make_block_ptr(
        base=Kt + kbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_kk, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_K_SIZE),
        order=(0, 1),
    )

    vbatch_head_offset = batch * stride_vb + head * stride_vh
    v_tile_ptr = tl.make_block_ptr(
        base=V + vbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_vt, stride_vk),
        offsets=(0, 0),
        block_shape=(TILE_K_SIZE, HEAD_DIM),
        order=(1, 0),
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
    q_lens_mask = (
        q_tile_indices[:, None] < seq_len
    )

    if not PERFECT_MATCHING:
        q_context_indices = q_tile_indices // CONTEXT_SIZE

    if Q_BLOCK_DIVISIBLE:
        q_tile = tl.load(q_tile_ptr)
    else:
        q_tile = tl.load(
            q_tile_ptr,
            boundary_check=(0,),
        )

    softmax_scale: tl.constexpr = tl.cast(SM_SCALE * RCP_LN2, q_tile.dtype)
    tile_k_arange = tl.arange(0, TILE_K_SIZE)

    for kv_tile_idx in tl.range(
        kv_start_tile_idx, kv_end_tile_idx, num_stages=PIPELINING
    ):
        kv_token_idx = kv_tile_idx * TILE_K_SIZE

        if K_BLOCK_DIVISIBLE:
            kt_tile = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
            )
            v_tile = tl.load(
                tl.advance(v_tile_ptr, (kv_token_idx, 0)),
            )
        else:
            kt_tile = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
                boundary_check=(1,),
            )
            v_tile = tl.load(
                tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                boundary_check=(0,),
            )

        if PRESCALE_QK:
            q_tile = q_tile * softmax_scale
        qk = tl.dot(
            q_tile, kt_tile, input_precision=INPUT_PRECISION, out_dtype=tl.float32
        )

        kv_indices = kv_token_idx + tile_k_arange
        mask = q_lens_mask & (
            kv_indices[None, :] < seq_len
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
        acc = tl.where(q_lens_mask, acc, 0.0)


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


@triton.autotune(
    configs=[
        triton.Config(
            dict(
                TILE_SIZE=tile,
            ),
            num_warps=num_warps,
        )
        for num_warps in [2, 4, 8]
        for tile in [32, 64, 128]
    ],
    key=["HEAD_DIM", "DTYPE", "TIME_BUCKET"],
)
@triton.heuristics(
    dict(
        BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_SIZE'] == 0,
    )
)
@triton.jit
def _streaming_attn_bwd_precompute(
    O, DO, RES,
    stride_ob: int, stride_oh: int, stride_ot: int, stride_ok: int,  #
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: int,  #
    stride_rb: int, stride_rh: int, stride_rt: int,
    T: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DTYPE:  tl.constexpr,  #
    TIME_BUCKET: tl.constexpr,  #
    TILE_SIZE: tl.constexpr,
    BLOCK_DIVISIBLE: tl.constexpr,  #
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    tile = tl.program_id(2)

    token_idx = tile * TILE_SIZE

    obatch_head_offset = batch * stride_ob + head * stride_oh
    o_tile_ptr = tl.make_block_ptr(
        base=O + obatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_ot, stride_ok),
        offsets=(token_idx, 0),
        block_shape=(TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    dobatch_head_offset = batch * stride_dob + head * stride_doh
    do_tile_ptr = tl.make_block_ptr(
        base=DO + dobatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dot, stride_dok),
        offsets=(token_idx, 0),
        block_shape=(TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    if BLOCK_DIVISIBLE:
        o_tile = tl.load(o_tile_ptr, )
        do_tile = tl.load(do_tile_ptr, )
    else:
        o_tile = tl.load(o_tile_ptr, boundary_check=(0,))
        do_tile = tl.load(do_tile_ptr, boundary_check=(0,))

    res = tl.sum(o_tile.to(tl.float32) * do_tile.to(tl.float32), 1)

    rbatch_head_offset = batch * stride_rb + head * stride_rh
    res_ptr = tl.make_block_ptr(
        base=RES + rbatch_head_offset,
        shape=(T,),
        strides=(stride_rt,),
        offsets=(token_idx,),
        block_shape=(TILE_SIZE,),
        order=(0,),
    )

    if BLOCK_DIVISIBLE:
        tl.store(res_ptr, res)
    else:
        tl.store(res_ptr, res, boundary_check=(0,))


@triton.heuristics(
    dict(
        RCP_LN2=lambda _: math.log2(math.e),
        DQ_TILES_NUM=lambda args: triton.cdiv(args['T'], args["TILE_DQ_Q_SIZE"]),
        PERFECT_DKV_MATCHING=lambda args : args['TILE_DK_Q_SIZE'] == args['TILE_DK_K_SIZE'] and args['TILE_DK_K_SIZE'] == args['CONTEXT_SIZE'],
        PERFECT_DQ_MATCHING=lambda args : args['TILE_DQ_Q_SIZE'] == args['TILE_DQ_K_SIZE'] and args['TILE_DQ_K_SIZE'] == args['CONTEXT_SIZE'],
        DQ_Q_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DQ_Q_SIZE'] == 0,
        DQ_K_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DQ_K_SIZE'] == 0,
        DK_Q_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DK_Q_SIZE'] == 0,
        DK_K_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DK_K_SIZE'] == 0,
    )
)
@triton.jit
def _streaming_attn_bwd(
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, L: tl.tensor, #
    DELTA: tl.tensor, LSE: tl.tensor,
    DO: tl.tensor, DQ: tl.tensor, DK: tl.tensor, DV: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,  #
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: int,  #
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,  #
    stride_deltab: int, stride_deltah: int, stride_deltat: int,  #
    stride_mb: int, stride_mh: int, stride_mt: int,  #
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: int,  #
    stride_dqb: int, stride_dqh: int, stride_dqt: int, stride_dqk: int,  #
    stride_dkb: int, stride_dkh: int, stride_dkt: int, stride_dkk: int,  #
    stride_dvb: int, stride_dvh: int, stride_dvt: int, stride_dvk: int,  #
    lens_stride: int,
    T: int,  #
    HEAD_DIM: tl.constexpr,  #
    CONTEXT_SIZE: tl.constexpr,  #
    CONTEXTS_BACK: tl.constexpr,  #
    DTYPE: tl.constexpr,  #
    TIME_BUCKET: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    SM_SCALE: tl.constexpr,  #
    DQ_TILES_NUM: tl.constexpr,  #
    PERFECT_DKV_MATCHING: tl.constexpr,  #
    PERFECT_DQ_MATCHING: tl.constexpr,  #
    DQ_Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    DQ_K_BLOCK_DIVISIBLE: tl.constexpr,  #
    DK_Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    DK_K_BLOCK_DIVISIBLE: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
    TILE_DQ_Q_SIZE: tl.constexpr, TILE_DQ_K_SIZE: tl.constexpr,  #
    TILE_DK_Q_SIZE: tl.constexpr, TILE_DK_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    dkv_worker = tl.program_id(2) >= DQ_TILES_NUM
    tile_id = tl.program_id(2) - (DQ_TILES_NUM * dkv_worker)

    seq_len = tl.load(L + batch * lens_stride)
    seq_len = min(seq_len, T)

    if dkv_worker:
        _streaming_attn_bwd_dkdv_inner(
            Q, K, V, DELTA, LSE, DO, DK, DV,
            stride_qb, stride_qh, stride_qt, stride_qk,
            stride_kb, stride_kh, stride_kt, stride_kk,
            stride_vb, stride_vh, stride_vt, stride_vk,
            stride_deltab, stride_deltah, stride_deltat,
            stride_mb, stride_mh, stride_mt,
            stride_dob, stride_doh, stride_dot, stride_dok,
            stride_dkb, stride_dkh, stride_dkt, stride_dkk,
            stride_dvb, stride_dvh, stride_dvt, stride_dvk,
            batch=batch,
            head=head,
            tile_id=tile_id,
            seq_len=seq_len,
            T=T,
            HEAD_DIM=HEAD_DIM,
            CONTEXT_SIZE=CONTEXT_SIZE,
            CONTEXTS_BACK=CONTEXTS_BACK,
            INPUT_PRECISION=INPUT_PRECISION,
            SM_SCALE=SM_SCALE,
            PERFECT_DKV_MATCHING=PERFECT_DKV_MATCHING,
            DK_Q_BLOCK_DIVISIBLE=DK_Q_BLOCK_DIVISIBLE,
            DK_K_BLOCK_DIVISIBLE=DK_K_BLOCK_DIVISIBLE,
            RCP_LN2=RCP_LN2,
            TILE_DK_Q_SIZE=TILE_DK_Q_SIZE,
            TILE_DK_K_SIZE=TILE_DK_K_SIZE,
            PIPELINING=PIPELINING
        )
    else:
        _streaming_attn_bwd_dq_inner(
            Q, K, V, DELTA, LSE,
            DO, DQ,
            stride_qb, stride_qh, stride_qt, stride_qk,
            stride_kb, stride_kh, stride_kt, stride_kk,
            stride_vb, stride_vh, stride_vt, stride_vk,
            stride_deltab, stride_deltah, stride_deltat,
            stride_mb, stride_mh, stride_mt,
            stride_dob, stride_doh, stride_dot, stride_dok,
            stride_dqb, stride_dqh, stride_dqt, stride_dqk,
            batch=batch,
            head=head,
            tile_id=tile_id,
            seq_len=seq_len,
            T=T,
            HEAD_DIM=HEAD_DIM,
            CONTEXT_SIZE=CONTEXT_SIZE,
            CONTEXTS_BACK=CONTEXTS_BACK,
            INPUT_PRECISION=INPUT_PRECISION,
            SM_SCALE=SM_SCALE,
            PERFECT_DQ_MATCHING=PERFECT_DQ_MATCHING,
            DQ_Q_BLOCK_DIVISIBLE=DQ_Q_BLOCK_DIVISIBLE,
            DQ_K_BLOCK_DIVISIBLE=DQ_K_BLOCK_DIVISIBLE,
            RCP_LN2=RCP_LN2,
            TILE_DQ_Q_SIZE=TILE_DQ_Q_SIZE,
            TILE_DQ_K_SIZE=TILE_DQ_K_SIZE,
            PIPELINING=PIPELINING,
        )


@triton.jit()
def _streaming_attn_bwd_dq_inner(
    Q, K, V, DELTA, LSE,
    DO, DQ,
    stride_qb, stride_qh, stride_qt, stride_qk,
    stride_kb, stride_kh, stride_kt, stride_kk,
    stride_vb, stride_vh, stride_vt, stride_vk,
    stride_deltab, stride_deltah, stride_deltat,
    stride_mb, stride_mh, stride_mt,
    stride_dob, stride_doh, stride_dot, stride_dok,
    stride_dqb, stride_dqh, stride_dqt, stride_dqk,
    batch,
    head,
    tile_id,
    seq_len,
    T,
    HEAD_DIM,
    CONTEXT_SIZE,
    CONTEXTS_BACK,
    INPUT_PRECISION,
    SM_SCALE,
    PERFECT_DQ_MATCHING,
    DQ_Q_BLOCK_DIVISIBLE,
    DQ_K_BLOCK_DIVISIBLE,
    RCP_LN2,
    TILE_DQ_Q_SIZE,
    TILE_DQ_K_SIZE,
    PIPELINING,
):
    q_tile_idx = tile_id
    q_token_idx = q_tile_idx * TILE_DQ_Q_SIZE

    qbatch_head_offset = batch * stride_qb + head * stride_qh
    q_tile_ptr = tl.make_block_ptr(
        base=Q + qbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_qt, stride_qk),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    lsebatch_head_offset = batch * stride_mb + head * stride_mh
    lse_tile_ptr = tl.make_block_ptr(
        base=LSE + lsebatch_head_offset,
        shape=(T,),
        strides=(stride_mt,),
        offsets=(0,),
        block_shape=(TILE_DQ_Q_SIZE,),
        order=(0,),
    )

    delta_tile_ptr = batch * stride_deltab + head * stride_deltah
    delta_tile_ptr = tl.make_block_ptr(
        base=DELTA + delta_tile_ptr,
        shape=(T,),
        strides=(stride_deltat,),
        offsets=(0,),
        block_shape=(TILE_DQ_Q_SIZE,),
        order=(0,),
    )

    dobatch_head_offset = batch * stride_dob + head * stride_doh
    do_tile_ptr = tl.make_block_ptr(
        base=DO + dobatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dot, stride_dok),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    if DQ_Q_BLOCK_DIVISIBLE:
        q = tl.load(q_tile_ptr)
        m = tl.load(lse_tile_ptr)[:, None]
        di = tl.load(delta_tile_ptr)
        do = tl.load(do_tile_ptr)
    else:
        q = tl.load(q_tile_ptr, boundary_check=(0,))
        m = tl.load(lse_tile_ptr, boundary_check=(0,))[:, None]
        di = tl.load(delta_tile_ptr, boundary_check=(0,))
        do = tl.load(do_tile_ptr, boundary_check=(0,))

    kbatch_head_offset = batch * stride_kb + head * stride_kh
    kt_tile_ptr = tl.make_block_ptr(
        base=K + kbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_kk, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_DQ_K_SIZE),
        order=(0, 1),
    )

    vbatch_head_offset = batch * stride_vb + head * stride_vh
    vt_tile_ptr = tl.make_block_ptr(
        base=V + vbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_vk, stride_vt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_DQ_K_SIZE),
        order=(1, 0),
    )

    dq = tl.zeros([TILE_DQ_Q_SIZE, HEAD_DIM], dtype=tl.float32)
    dq = _streaming_attn_bwd_dq(
        dq, q, m, di, do,
        kt_tile_ptr, vt_tile_ptr,
        seq_len=seq_len,
        q_token_idx=q_token_idx,
        HEAD_DIM=HEAD_DIM,
        CONTEXT_SIZE=CONTEXT_SIZE,
        CONTEXTS_BACK=CONTEXTS_BACK,
        TILE_Q_SIZE=TILE_DQ_Q_SIZE,
        TILE_K_SIZE=TILE_DQ_K_SIZE,
        INPUT_PRECISION=INPUT_PRECISION,
        PIPELINING=PIPELINING,
        K_BLOCK_DIVISIBLE=DQ_K_BLOCK_DIVISIBLE,
        PERFECT_MATCHING=PERFECT_DQ_MATCHING,
        RCP_LN2=RCP_LN2,
        SM_SCALE=SM_SCALE,
    )

    dqbatch_head_offset = batch * stride_dqb + head * stride_dqh
    dq_tile_ptr = tl.make_block_ptr(
        base=DQ + dqbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dqt, stride_dqk),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    if DQ_Q_BLOCK_DIVISIBLE:
        tl.store(dq_tile_ptr, dq.to(dq_tile_ptr.type.element_ty))
    else:
        tl.store(dq_tile_ptr, dq.to(dq_tile_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def _streaming_attn_bwd_dkdv_inner(
    Q, K, V, DELTA, LSE, DO, DK, DV,
    stride_qb, stride_qh, stride_qt, stride_qk,
    stride_kb, stride_kh, stride_kt, stride_kk,
    stride_vb, stride_vh, stride_vt, stride_vk,
    stride_deltab, stride_deltah, stride_deltat,
    stride_mb, stride_mh, stride_mt,
    stride_dob, stride_doh, stride_dot,
    stride_dok, stride_dkb, stride_dkh,
    stride_dkt, stride_dkk, stride_dvb,
    stride_dvh, stride_dvt, stride_dvk,
    batch,
    head,
    tile_id,
    seq_len,
    T: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    CONTEXT_SIZE: tl.constexpr,  #
    CONTEXTS_BACK: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    SM_SCALE: tl.constexpr,  #
    PERFECT_DKV_MATCHING: tl.constexpr,  #
    DK_Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    DK_K_BLOCK_DIVISIBLE: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
    TILE_DK_Q_SIZE: tl.constexpr,  #
    TILE_DK_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
):
    kv_tile_idx = tile_id
    kv_token_idx = kv_tile_idx * TILE_DK_K_SIZE

    qbatch_head_offset = batch * stride_qb + head * stride_qh
    qt_tile_ptr = tl.make_block_ptr(
        base=Q + qbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_qk, stride_qt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_DK_Q_SIZE),
        order=(0, 1),
    )

    kbatch_head_offset = batch * stride_kb + head * stride_kh
    k_tile_ptr = tl.make_block_ptr(
        base=K + kbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_kt, stride_kk),
        offsets=(kv_token_idx, 0),
        block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    vbatch_head_offset = batch * stride_vb + head * stride_vh
    v_tile_ptr = tl.make_block_ptr(
        base=V + vbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_vt, stride_vk),
        offsets=(kv_token_idx, 0),
        block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    dobatch_head_offset = batch * stride_dob + head * stride_doh
    do_tile_ptr = tl.make_block_ptr(
        base=DO + dobatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dot, stride_dok),
        offsets=(0, 0),
        block_shape=(TILE_DK_Q_SIZE, HEAD_DIM),
        order=(1, 0),
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

    if DK_K_BLOCK_DIVISIBLE:
        k = tl.load(
                k_tile_ptr,
            )
        v = tl.load(
                v_tile_ptr,
            )
    else:
        k = tl.load(
                k_tile_ptr,
                boundary_check=(0,),
            )
        v = tl.load(
                v_tile_ptr,
                boundary_check=(0,),
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
        PERFECT_MATCHING=PERFECT_DKV_MATCHING,
        PIPELINING=PIPELINING,
        Q_BLOCK_DIVISIBLE=DK_Q_BLOCK_DIVISIBLE,
        RCP_LN2=RCP_LN2,
        SM_SCALE=SM_SCALE,
    )

    dkbatch_head_offset = batch * stride_dkb + head * stride_dkh
    dk_tile_ptr = tl.make_block_ptr(
        base=DK + dkbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dkt, stride_dkk),
        offsets=(kv_token_idx, 0),
        block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    if DK_K_BLOCK_DIVISIBLE:
        tl.store(dk_tile_ptr, dk.to(dk_tile_ptr.type.element_ty))
    else:
        tl.store(dk_tile_ptr, dk.to(dk_tile_ptr.type.element_ty), boundary_check=(0,))

    dvbatch_head_offset = batch * stride_dvb + head * stride_dvh
    dv_tile_ptr = tl.make_block_ptr(
        base=DV + dvbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dvt, stride_dvk),
        offsets=(kv_token_idx, 0),
        block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    if DK_K_BLOCK_DIVISIBLE:
        tl.store(dv_tile_ptr, dv.to(dv_tile_ptr.type.element_ty))
    else:
        tl.store(dv_tile_ptr, dv.to(dv_tile_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def _streaming_attn_bwd_dq(
    dq, q, m, di, do,
    kt_tile_ptr, vt_tile_ptr,
    seq_len,
    q_token_idx,
    HEAD_DIM: tl.constexpr,
    CONTEXT_SIZE: tl.constexpr,
    CONTEXTS_BACK: tl.constexpr,
    TILE_Q_SIZE: tl.constexpr,
    TILE_K_SIZE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    PERFECT_MATCHING: tl.constexpr,
    PIPELINING: tl.constexpr,
    K_BLOCK_DIVISIBLE: tl.constexpr,
    RCP_LN2: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    q_tile_min_context = q_token_idx // CONTEXT_SIZE
    kv_start_tile_idx = max(
        0, ((q_tile_min_context - CONTEXTS_BACK) * CONTEXT_SIZE)
    ) // TILE_K_SIZE

    q_tile_max_token = min(q_token_idx + TILE_Q_SIZE, seq_len)
    q_tile_max_context = (q_tile_max_token - 1) // CONTEXT_SIZE
    kv_end_tile_idx = tl.cdiv(
        min((q_tile_max_context + 1) * CONTEXT_SIZE, seq_len), TILE_K_SIZE
    )

    softmax_scale: tl.constexpr = tl.cast(SM_SCALE, q.dtype)

    q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)
    q_len_mask = q_tile_indices[:, None] < seq_len
    tile_k_arange = tl.arange(0, TILE_K_SIZE)
    for kv_tile_idx in tl.range(
        kv_start_tile_idx, kv_end_tile_idx, num_stages=PIPELINING
    ):
        kv_token_idx = kv_tile_idx * TILE_K_SIZE
        if K_BLOCK_DIVISIBLE:
            kT = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
            )
            vT = tl.load(
                tl.advance(vt_tile_ptr, (0, kv_token_idx)),
            )
        else:
            kT = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
                boundary_check=(1,),
            )
            vT = tl.load(
                tl.advance(vt_tile_ptr, (0, kv_token_idx,)),
                boundary_check=(1,),
            )

        qk = tl.dot(q * softmax_scale, kT, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        p = tl.math.exp2(qk / RCP_LN2 - m)

        kv_indices = kv_token_idx + tile_k_arange
        mask = q_len_mask & (
            kv_indices[None, :] < seq_len
        )
        if not PERFECT_MATCHING:
            q_context_indices = q_tile_indices // CONTEXT_SIZE
            kv_context_indices = kv_indices // CONTEXT_SIZE
            blocks_diff = q_context_indices[:, None] - kv_context_indices[None, :]
            streaming_mask = (blocks_diff >= 0) & (blocks_diff <= CONTEXTS_BACK)
            mask &= streaming_mask
        p = tl.where(mask, p, 0.0)
        dp = tl.dot(do, vT, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        ds = p * (dp - di[:, None]) * softmax_scale
        dq = tl.dot(ds.to(kT.dtype), tl.trans(kT), dq, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
    return dq


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
    PERFECT_MATCHING: tl.constexpr,
    PIPELINING: tl.constexpr,
    Q_BLOCK_DIVISIBLE: tl.constexpr,
    RCP_LN2: tl.constexpr,
    SM_SCALE: tl.constexpr,
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

    softmax_scale: tl.constexpr = tl.cast(SM_SCALE / RCP_LN2, k.dtype)
    tile_q_arange = tl.arange(0, TILE_Q_SIZE)

    kv_lens_mask = (
        kv_indices[:, None] < seq_len
    )

    for q_tile_idx in tl.range(q_start_tile_idx, q_end_tile_idx, num_stages=PIPELINING):
        q_token_idx = q_tile_idx * TILE_Q_SIZE
        # NOTE: triton will not reorder loads
        # if there are problems with shared memory, do and Di loads can be moved just before usage
        # (via constexpr flag)
        if Q_BLOCK_DIVISIBLE:
            qT = tl.load(
                tl.advance(qt_tile_ptr, (0, q_token_idx)),
            )
            m = tl.load(
                tl.advance(lse_tile_ptr, (q_token_idx,)),
            )
            do = tl.load(
                tl.advance(do_tile_ptr, (q_token_idx, 0)),
            )
            Di = tl.load(
                tl.advance(delta_tile_ptr, (q_token_idx,)),
            )
        else:
            qT = tl.load(
                tl.advance(qt_tile_ptr, (0, q_token_idx)),
                boundary_check=(1,),
            )
            m = tl.load(
                tl.advance(lse_tile_ptr, (q_token_idx,)),
                boundary_check=(0,),
            )
            do = tl.load(
                tl.advance(do_tile_ptr, (q_token_idx, 0)),
                boundary_check=(0,),
            )
            Di = tl.load(
                tl.advance(delta_tile_ptr, (q_token_idx,)),
                boundary_check=(0,),
            )
        tl.static_assert(m.dtype == tl.float32)

        qkT = tl.dot(k * softmax_scale, qT, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        pT = tl.math.exp2(qkT - m[None, :])

        q_tile_indices = q_token_idx + tile_q_arange
        mask = kv_lens_mask & (
            q_tile_indices[None, :] < seq_len
        )
        if not PERFECT_MATCHING:
            q_context_indices = q_tile_indices // CONTEXT_SIZE
            blocks_diff = q_context_indices[None, :] - kv_context_indices[:, None]
            streaming_mask = (blocks_diff >= 0) & (blocks_diff <= CONTEXTS_BACK)
            mask &= streaming_mask
        pT = tl.where(mask, pT, 0.0)

        dv = tl.dot(pT.to(do.dtype), do, dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        tl.static_assert(Di.dtype == tl.float32)

        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dk = tl.dot(dsT.to(qT.dtype), tl.trans(qT), dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
    return dk, dv


class StreamingAttention(torch.autograd.Function):
    streaming_forward = triton.heuristics(
        dict(
            PIPELINING=lambda _: 1,
            TILE_Q_SIZE=lambda args: min(
                64, max(MIN_TILE_SIZE, triton.next_power_of_2(args['CONTEXT_SIZE']))
            ),
            TILE_K_SIZE=lambda args: min(
                64, max(MIN_TILE_SIZE, triton.next_power_of_2(args['CONTEXT_SIZE']))
            ),
        )
    )(
        _streaming_attn_fwd
    )
    streaming_forward_autotune = triton.autotune(
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
        ),
    )(
        _streaming_attn_fwd
    )

    streaming_bsckward = triton.heuristics(
        dict(
            PIPELINING=lambda _: 1,
            TILE_DQ_Q_SIZE=lambda args: min(
                64, max(MIN_TILE_SIZE, triton.next_power_of_2(args['CONTEXT_SIZE']))
            ),
            TILE_DQ_K_SIZE=lambda args: min(
                64, max(MIN_TILE_SIZE, triton.next_power_of_2(args['CONTEXT_SIZE']))
            ),
            TILE_DK_Q_SIZE=lambda args: min(
                64, max(MIN_TILE_SIZE, triton.next_power_of_2(args['CONTEXT_SIZE']))
            ),
            TILE_DK_K_SIZE=lambda args: min(
                64, max(MIN_TILE_SIZE, triton.next_power_of_2(args['CONTEXT_SIZE']))
            ),
        )
    )(
        _streaming_attn_bwd
    )
    streaming_bsckward_autotune = triton.autotune(
        configs=[
            triton.Config(
                dict(
                    PIPELINING=pipe,
                    TILE_DQ_Q_SIZE=tile_qq,
                    TILE_DQ_K_SIZE=tile_qk,
                    TILE_DK_Q_SIZE=tile_kq,
                    TILE_DK_K_SIZE=tile_kk,
                ),
                num_warps=num_warps,
                num_stages=pipe,
            )
            for num_warps in [4, 8]
            for pipe in [1, 2, 3]
            for tile_qq in [2 ** i for i in range(int(math.log2(MIN_TILE_SIZE) + 0.1), int(math.log2(MAX_TILE_SIZE) + 0.1) + 1)]
            for tile_qk in [2 ** i for i in range(int(math.log2(MIN_TILE_SIZE) + 0.1), int(math.log2(MAX_TILE_SIZE) + 0.1) + 1)]
            for tile_kq in [2 ** i for i in range(int(math.log2(MIN_TILE_SIZE) + 0.1), int(math.log2(MAX_TILE_SIZE) + 0.1) + 1)]
            for tile_kk in [2 ** i for i in range(int(math.log2(MIN_TILE_SIZE) + 0.1), int(math.log2(MAX_TILE_SIZE) + 0.1) + 1)]
        ],
        key=["HEAD_DIM", "CONTEXT_SIZE", "CONTEXTS_BACK", "INPUT_PRECISION", "DTYPE", "TIME_BUCKET"],
        prune_configs_by=dict(
            early_config_prune=bwd_configs_pruner
        ),
    )(
        _streaming_attn_bwd
    )

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lens: torch.Tensor | None, context_size: int, back_contexts: int, sm_scale: int | None, autotune: bool):
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

        if sm_scale is None:
            sm_scale = HEAD_DIM ** -0.5

        grid = lambda args: (
            batch,
            heads,
            triton.cdiv(T, args["TILE_Q_SIZE"]),
        )

        need_grad = any(i.requires_grad for i in (q, k, v))

        kt = k.transpose(-1, -2)  # just stride tricks, same data
        fwd_fn = StreamingAttention.streaming_forward_autotune if autotune else StreamingAttention.streaming_forward
        fwd_fn[grid](
            q, kt, v, lens,
            LSE, O,
            *strides(q),
            *strides(kt),
            *strides(v),
            *strides(LSE),
            *strides(O),
            *strides(lens),
            T=T,
            HEAD_DIM=HEAD_DIM,
            CONTEXT_SIZE=context_size,
            CONTEXTS_BACK=back_contexts,
            INPUT_PRECISION=(
                "tf32" if torch.get_float32_matmul_precision() != "highest" else "ieee"
            ),
            PRESCALE_QK=True,
            DTYPE=q.dtype,
            TIME_BUCKET=triton.next_power_of_2(T),
            OUTPUT_LOGSUMEXP=need_grad,
            SM_SCALE=sm_scale,
        )

        ctx.save_for_backward(q, k, v, O, LSE, lens)
        ctx.grid = grid
        ctx.context_size = context_size
        ctx.back_contexts = back_contexts
        ctx.HEAD_DIM = HEAD_DIM
        ctx.autotune = autotune
        ctx.sm_scale = sm_scale
        return O

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, lens = ctx.saved_tensors
        batch, heads, T, HEAD_DIM = q.shape

        delta = torch.empty(o.shape[:-1], dtype=torch.float32, device=o.device)
        grid = lambda args: (
            batch,
            heads,
            triton.cdiv(T, args["TILE_SIZE"]),
        )
        _streaming_attn_bwd_precompute[grid](
            o, do, delta,
            *strides(o),
            *strides(do),
            *strides(delta),
            T=T,
            HEAD_DIM=HEAD_DIM,
            DTYPE=q.dtype,
            TIME_BUCKET=triton.next_power_of_2(T),
        )

        DQ = torch.zeros_like(q, memory_format=torch.contiguous_format)
        DK = torch.zeros_like(k, memory_format=torch.contiguous_format)
        DV = torch.zeros_like(v, memory_format=torch.contiguous_format)

        grid = lambda args: (
            batch,
            heads,
            triton.cdiv(T, args["TILE_DQ_Q_SIZE"]) + triton.cdiv(T, args["TILE_DK_K_SIZE"]),
        )

        fwd_fn = StreamingAttention.streaming_bsckward_autotune if ctx.autotune else StreamingAttention.streaming_bsckward
        fwd_fn[grid](
            q, k, v, lens,
            delta, lse,
            do, DQ, DK, DV,
            *strides(q),
            *strides(k),
            *strides(v),
            *strides(delta),
            *strides(lse),
            *strides(do),
            *strides(DQ),
            *strides(DK),
            *strides(DV),
            *strides(lens),
            T=T,
            HEAD_DIM=HEAD_DIM,
            CONTEXT_SIZE=ctx.context_size,
            CONTEXTS_BACK=ctx.back_contexts,
            TIME_BUCKET=triton.next_power_of_2(T),
            INPUT_PRECISION=(
                "tf32" if torch.get_float32_matmul_precision() != "highest" else "ieee"
            ),
            DTYPE=q.dtype,
            SM_SCALE=ctx.sm_scale,
        )

        return DQ, DK, DV, None, None, None, None, None


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


def streaming_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None,
    context_size: int,
    back_contexts: int,
    sm_scale: float | None = None,
    autotune=True,
):
    """
    Computes block-sparse self-attention with chunked attention mask.
    Time is divided into blovks of `context_size`, query can attend to all kv in the current context
    and to `back_contexts` contexts before the current one.


    Unlike traditional attention mechanisms that scale quadratically with sequence length,
    streaming attention maintains a linear runtime and memory footprint.

    Args:
        q (Tensor): The query tensor of shape `(batch, heads_num, time, head_dim)`
        k (Tensor): The key tensor of shape `(batch, heads_num, time, head_dim)`
        v (Tensor): The value tensor of shape `(batch, heads_num, time, head_dim)`
        lens (Tensor | None): Lengths of sequrnces of shape `(batch,)`
        context_size (int): Size of the context block
        back_contexts (int): Number of contexts to look back
        sm_scale (float): Softmax scale, head_dim ** -0.5 by default
        autotune (bool): Use triton autotune for optimal kernel configuration
    """
    return StreamingAttention.apply(
        q, k, v, lens, context_size, back_contexts, sm_scale, autotune
    )


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
