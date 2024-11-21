import torch
import pytest
import os
import numpy as np

from streaming_attention import streaming_attention

@pytest.mark.parametrize("dtype", [torch.float16], ids=lambda x: f"{x}")
@pytest.mark.parametrize("lens", ['none', 'tricky', 'random'], ids=lambda x: f"lens-{x}")
@pytest.mark.parametrize("HEAD_DIM", [16, 128, 256], ids=lambda x: f"dim-{x}")
@pytest.mark.parametrize("B", [1, 7, 67], ids=lambda x: f"batch-{x}")
@pytest.mark.parametrize("H", [1, 6], ids=lambda x: f"heads-{x}")
@pytest.mark.parametrize("T", [1, 10, 16, 313], ids=lambda x: f"time-{x}")
@pytest.mark.parametrize(
    "context_size", [1, 10, 16, 32, 100, 256], ids=lambda x: f"context-{x}"
)
@pytest.mark.parametrize(
    "back_contexts", [0, 1, 5, 1000], ids=lambda x: f"back_contexts-{x}"
)
@torch.inference_mode()
def test_op(
    B,
    H,
    T,
    HEAD_DIM,
    context_size,
    back_contexts,
    dtype,
    lens,
):
    torch.manual_seed(20)
    torch.set_float32_matmul_precision("highest")

    if os.environ.get("TRITON_INTERPRET") == "1" and dtype == torch.bfloat16:
        pytest.skip("skipping bf16 in interpreter mode")

    q = torch.zeros((B, H, T, HEAD_DIM), dtype=dtype, device="cuda").normal_(
        mean=0.0, std=0.01
    )
    k = torch.zeros((B, H, T, HEAD_DIM), dtype=dtype, device="cuda").normal_(
        mean=0.0, std=0.01
    )
    v = torch.zeros((B, H, T, HEAD_DIM), dtype=dtype, device="cuda").normal_(
        mean=0.0, std=0.01
    )

    if lens == "none":
        lens = None
    elif lens == "tricky":
        tricky_lens = [
            2,
            5,
            context_size,
            context_size + 1,
            max(context_size - 1, 1),
            max(context_size // 2, 1),
            context_size // 2 + context_size,
            max(context_size * back_contexts, 1),
            context_size * back_contexts + 1,
            max(context_size * back_contexts - 1, 1),
            T + 1,
            T,
            max(T // 2, 1),
            max(T // 4, 1),
        ]
        lens = torch.tensor(
            np.random.choice(tricky_lens, B), dtype=torch.int32, device="cuda"
        )
    else:
        lens = torch.randint(1, T + 1, (B,), dtype=torch.int32, device="cuda")

    ref, res_mask, _ = _get_reference(q, k, v, context_size, back_contexts, lens)
    ref, res_mask = ref.cuda(), res_mask.cuda()
    tri_out = attention(q, k, v, lens, context_size, back_contexts)
    tri_out = torch.where(res_mask.broadcast_to(tri_out.shape), tri_out, 0)

    # torch.set_printoptions(linewidth=400, profile="full")

    atol = 1e-3
    errors = abs(tri_out - ref) > atol
    b_mismatch = torch.argmax(errors.sum((1, 2, 3)).view(-1))
    h_mismatch = torch.argmax(errors[b_mismatch].sum((1, 2)).view(-1))

    torch.testing.assert_close(
        ref,
        tri_out,
        atol=atol,
        rtol=0,
        msg=lambda x: f"{x}\n\n{(b_mismatch, h_mismatch)}:\n{(errors[b_mismatch, h_mismatch]).long()} \n\n {(tri_out - ref)[errors].view(-1)}\n\nlens:\n{lens}\n{ref}\n{tri_out}",
    )


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
    batches = (128, )
    configs = []
    params = [
        dict(
            batch=batch,
            back_context=9,
            context_size=10,
            dim=128,
            heads=6,
            name='relevant',
        )
        for batch in batches
    ] + [
        dict(
            batch=batch,
            back_context=2,
            context_size=32,
            dim=128,
            heads=6,
            name='perfect-fit',
        )
        for batch in batches
    ] + [
        dict(
            batch=batch,
            back_context=2,
            context_size=256,
            dim=256,
            heads=6,
            name='perfect-fit-large',
        )
        for batch in batches
    ]
    for param in params:
        line_vals = [
            "triton",
            "flex-compile",
            "torch",
        ]
        context_size = param['context_size']
        back_context = param['back_context']
        dim = param['dim']
        heads = param['heads']
        batch = param['batch']

        x_vals = np.linspace(4000, 5000, 8).astype(int).tolist()
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
                plot_name=f"streaming-attention-{param['name']}-context-{context_size}-back-{back_context}-dim-{dim}-heads-{heads}-batch-{batch}",
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

        torch.set_float32_matmul_precision('highest')

        q = torch.randn((batch, heads, time, dim), dtype=dtype, device=device).normal_(
            mean=0.0, std=0.01
        )
        k = torch.randn((batch, heads, time, dim), dtype=dtype, device=device).normal_(
            mean=0.0, std=0.01
        )
        v = torch.randn((batch, heads, time, dim), dtype=dtype, device=device).normal_(
            mean=0.0, std=0.01
        )

        lens = torch.randint(1, time + 1, (batch,), dtype=torch.long, device="cuda")

        if "triton" in provider:
            fn = lambda: attention(q, k, v, lens, context_size, back_contexts)
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

        ref, res_mask, sparsity_fraction = _get_reference(q, k, v, context_size, back_contexts, lens)
        ref, res_mask = ref.cuda(), res_mask.cuda()

        print(f"Starting {provider}")
        actual = fn()
        actual = torch.where(res_mask.broadcast_to(actual.shape), actual, 0)

        atol = 3e-3
        torch.testing.assert_close(
            actual,
            ref,
            atol=atol,
            rtol=0,
            msg=lambda x: f"error in {provider}\n{x}",
        )

        with torch.inference_mode():
            ms = triton.testing.do_bench(fn, warmup=500, rep=1000, return_mode='mean')

        context_tok_size = context_size * (1 + back_contexts)
        total_flops = (
            4 * time * context_tok_size * heads * dim * batch
        )
        return (total_flops / (ms / 1000)) / 1e12

    bench_streaming_attention.run(save_path=".", print_data=True)


def _get_reference(q, k, v, context_size, back_contexts, lens):
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
