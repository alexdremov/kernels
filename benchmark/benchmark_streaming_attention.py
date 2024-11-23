import contextlib
import functools
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import triton

sys.path.insert(
    0,
    f"{os.path.dirname(os.path.realpath(__file__))}/../src"
)


from torch.nn.attention.flex_attention import \
    _round_up_to_multiple as round_up_to_multiple
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from streaming_attention import (streaming_attention,
                                 streaming_attention_reference)


def _flex_attention_chunked_mask_generator(
    b, h, q_idx, kv_idx, block_size, left_context_blocks_count, input_lengths
):
    q_block_idxes = torch.div(q_idx, block_size, rounding_mode="floor")
    kv_block_idxes = torch.div(kv_idx, block_size, rounding_mode="floor")
    diff = q_block_idxes - kv_block_idxes

    blocks = (diff >= 0) & (diff < left_context_blocks_count)
    if input_lengths is None:
        return blocks

    padding_condition = (q_idx < input_lengths[b]) & (kv_idx < input_lengths[b])
    return blocks & padding_condition


if __name__ == "__main__":
    # import sys
    # test_op(
    #     B=1, H=1, T=256, HEAD_DIM=16, context_size=257, back_contexts=0, dtype=torch.float32, lens='tricky'
    # )
    # sys.exit()
    batches = (96,)
    configs = []
    params = (
        [
            dict(
                batch=batch,
                back_context=9,
                context_size=10,
                dim=128,
                heads=6,
                name="relevant",
            )
            for batch in batches
        ]
        + [
            dict(
                batch=batch,
                back_context=2,
                context_size=32,
                dim=128,
                heads=6,
                name="perfect-fit",
            )
            for batch in batches
        ]
        + [
            dict(
                batch=batch,
                back_context=2,
                context_size=256,
                dim=256,
                heads=6,
                name="perfect-fit-large",
            )
            for batch in batches
        ]
    )
    for mode in ('bwd', 'fwd'):
        for param in params:
            line_vals = [
                f"triton-{mode}",
                f"flex-compile-{mode}",
                f"torch-{mode}",
            ]
            context_size = param["context_size"]
            back_context = param["back_context"]
            dim = param["dim"]
            heads = param["heads"]
            batch = param["batch"]

            x_vals = np.linspace(257, 3000, 6).astype(int).tolist()
            # x_vals = np.linspace(2400, 3000, 1).astype(int).tolist()
            x_vals = np.unique(x_vals)
            x_vals = sorted(x_vals)


            configs.append(
                triton.testing.Benchmark(
                    x_names=["time"],
                    x_vals=x_vals,
                    line_arg="provider",
                    line_vals=line_vals,
                    line_names=line_vals,
                    styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")],
                    ylabel="TFLOPS",
                    plot_name=f"streaming-attention-{mode}-{param['name']}-context-{context_size}-back-{back_context}-dim-{dim}-heads-{heads}-batch-{batch}",
                    args=dict(
                        batch=batch,
                        heads=heads,
                        dim=dim,
                        dtype=torch.float16,
                        context_size=context_size,
                        back_contexts=back_context,
                    ),
                )
            )

    @triton.testing.perf_report(configs)
    def bench_streaming_attention(
        provider, time, batch, heads, dim, dtype, context_size, back_contexts
    ):
        device = "cuda"

        torch.set_float32_matmul_precision("highest")

        q = torch.randn((batch, heads, time, dim), dtype=dtype, device=device).normal_(
            mean=0.0, std=0.01
        ).requires_grad_()
        k = torch.randn((batch, heads, time, dim), dtype=dtype, device=device).normal_(
            mean=0.0, std=0.01
        ).requires_grad_()
        v = torch.randn((batch, heads, time, dim), dtype=dtype, device=device).normal_(
            mean=0.0, std=0.01
        ).requires_grad_()

        lens = None

        if "triton" in provider:
            fn = lambda: streaming_attention(q, k, v, lens, context_size, back_contexts)
        elif "torch" in provider:
            block_size = context_size
            left_context_blocks_count = back_contexts + 1

            block_idxes = torch.div(
                torch.arange(time), block_size, rounding_mode="floor"
            )
            block_idxes_diff = block_idxes.unsqueeze(1) - block_idxes.unsqueeze(0)
            attn_mask = (block_idxes_diff >= 0) & (
                block_idxes_diff < left_context_blocks_count
            )
            attn_mask = attn_mask.cuda()

            if lens is not None:
                key_padding_mask = (
                    torch.arange(time, device="cuda").unsqueeze(0) < lens.unsqueeze(-1)
                ).unsqueeze(-1)
                key_padding_mask = key_padding_mask & key_padding_mask.transpose(-1, -2)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(
                    0
                ) & key_padding_mask.unsqueeze(1)

            fn = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask)

        elif "flex" in provider:
            block_size = context_size
            left_context_blocks_count = back_contexts + 1

            max_time = time

            sparse_block_size = block_size
            sparse_block_size = round_up_to_multiple(sparse_block_size, 128)
            max_time = round_up_to_multiple(max_time, sparse_block_size)

            block_mask = create_block_mask(
                functools.partial(
                    _flex_attention_chunked_mask_generator,
                    block_size=torch.as_tensor(block_size, device=device),
                    left_context_blocks_count=torch.as_tensor(
                        left_context_blocks_count, device=device
                    ),
                    input_lengths=(
                        torch.as_tensor(lens, device=device)
                        if lens is not None
                        else None
                    ),
                ),
                device=device,
                B=len(lens) if lens is not None else None,
                H=None,  # invariant
                Q_LEN=max_time,
                KV_LEN=max_time,
                BLOCK_SIZE=sparse_block_size,  # this is crucial to have full blocks
            )

            if "compile" in provider:
                fn = lambda: torch.compile(
                    flex_attention,
                )(q, k, v, block_mask=block_mask)
            else:
                fn = lambda: flex_attention(q, k, v, block_mask=block_mask)

        if 'bwd' in provider:
            do = torch.randn((batch, heads, time, dim), dtype=torch.float32, device=device).normal_(
                mean=0.0, std=0.01
            )
            def fn_back(fn):
                res = fn()
                res.backward(do)
                return res, q.grad, k.grad, v.grad

            fn = functools.partial(fn_back, fn=fn)

        ref, res_mask, _ = streaming_attention_reference(
            q, k, v, context_size, back_contexts, lens
        )
        ref, res_mask = ref.cuda(), res_mask.cuda()

        print(f"Starting {provider}")
        if 'bwd' in provider:
            ref.backward(do)
            dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()
            q.grad, k.grad, v.grad = [None] * 3

            actual, dq, dk, dv = fn()
            dq, dk, dv = dq.clone(), dk.clone(), dv.clone()
        else:
            actual = fn()

        actual = actual * res_mask.broadcast_to(actual.shape)

        atol = 3e-3
        torch.testing.assert_close(
            actual,
            ref,
            atol=atol,
            rtol=0,
            msg=lambda x: f"error in {provider}\n{x}",
        )

        if 'bwd' in provider and 'triton' in provider:
            for i, (d_ref, d_tri) in enumerate([(dq_ref, dq), (dk_ref, dk), (dv_ref, dv)]):
                atol = 1e-3
                errors = abs(d_ref - d_tri) > atol
                b_mismatch = torch.argmax(errors.sum((1, 2, 3)).view(-1)).item()
                h_mismatch = torch.argmax(errors[b_mismatch].sum((1, 2)).view(-1)).item()

                mask = res_mask.broadcast_to(d_ref.shape)
                mean_err = (abs(d_ref[mask].to(torch.float32) - d_tri[mask].to(torch.float32)).mean() * 1000).item()

                torch.testing.assert_close(
                    d_tri,
                    d_ref,
                    atol=atol,
                    rtol=0,
                    msg=lambda x: f"error in d{'qkv'[i]}\n{x}\n\n{(b_mismatch, h_mismatch)}:\n{(errors[b_mismatch, h_mismatch]).long()} \n\n {(d_tri - d_ref)[errors].view(-1)}\n\nlens:\n{lens}\n{mean_err = }"
                )

        with torch.inference_mode() if 'fwd' in provider else contextlib.nullcontext():
            ms = triton.testing.do_bench(
                fn,
                warmup=500,
                rep=1000,
                return_mode="mean",
                grad_to_none=(q, k, v)
            )

        context_tok_size = context_size * (1 + back_contexts)
        total_flops = 4 * time * context_tok_size * heads * dim * batch
        if 'bwd' in provider:
            total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        return (total_flops / (ms / 1000)) / 1e12

    bench_streaming_attention.run(save_path=".", print_data=True)
