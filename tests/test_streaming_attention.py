import os

import numpy as np
import pytest
import torch

from streaming_attention import (streaming_attention,
                                 streaming_attention_reference)


@pytest.mark.parametrize("dtype", [torch.float16], ids=lambda x: f"{x}")
@pytest.mark.parametrize(
    "lens", ["tricky", "random"], ids=lambda x: f"lens-{x}"
)
@pytest.mark.parametrize(
    "noncontiguous", [False, True], ids=lambda x: f"noncontiguous-{x}"
)
@pytest.mark.parametrize("HEAD_DIM", [16, 128, 256], ids=lambda x: f"dim-{x}")
@pytest.mark.parametrize("B", [1, 40], ids=lambda x: f"batch-{x}")
@pytest.mark.parametrize("H", [1, 6], ids=lambda x: f"heads-{x}")
@pytest.mark.parametrize("T", [1, 10, 16, 800], ids=lambda x: f"time-{x}")
@pytest.mark.parametrize(
    "context_size", [1, 10, 16, 32, 10000], ids=lambda x: f"context-{x}"
)
@pytest.mark.parametrize(
    "back_contexts", [0, 5, 10000], ids=lambda x: f"back_contexts-{x}"
)
@pytest.mark.parametrize(
    "autotune", [False, True], ids=lambda x: f"autotune-{x}"
)
def test_streaming_attention(
    B,
    H,
    T,
    HEAD_DIM,
    context_size,
    back_contexts,
    dtype,
    lens,
    noncontiguous,
    autotune,
):
    torch.manual_seed(20)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.empty_cache()

    if os.environ.get("TRITON_INTERPRET") == "1" and dtype == torch.bfloat16:
        pytest.skip("skipping bf16 in interpreter mode")

    if autotune and not (
        back_contexts == 5 and
        context_size in {10, 16} and
        T in {16, 800} and
        H == 1 and
        B == 67 and
        noncontiguous and
        lens == 'tricky'
    ):
         pytest.skip("reduced tests for autotune")

    shape_mul = 2 if noncontiguous else 1

    q, k, v = [
        torch.testing.make_tensor(
            (B * shape_mul, H * shape_mul, T * shape_mul, HEAD_DIM * shape_mul),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
            noncontiguous=noncontiguous,
            low=-0.01,
            high=0.01
        )
        for _ in range(3)
    ]

    dout = torch.testing.make_tensor(
        (B * shape_mul, H * shape_mul, T * shape_mul, HEAD_DIM * shape_mul),
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
        noncontiguous=noncontiguous,
        low=-0.01,
        high=0.01
    )

    if noncontiguous:
        q = q[1::2, 1::2, 1::2, 1::2].detach().clone().requires_grad_()
        k = k[1::2, 1::2, 1::2, 1::2].detach().clone().requires_grad_()
        v = v[1::2, 1::2, 1::2, 1::2].detach().clone().requires_grad_()
        dout = dout[1::2, 1::2, 1::2, 1::2]

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

    ref, res_mask, _ = streaming_attention_reference(
        q, k, v, context_size, back_contexts, lens
    )
    ref.backward(dout)

    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    tri_out = streaming_attention(q, k, v, lens, context_size, back_contexts, autotune=autotune)
    tri_out.backward(dout)

    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None

    # torch.set_printoptions(linewidth=400, profile="full")

    tri_out = tri_out * res_mask.broadcast_to(tri_out.shape)
    atol = 1e-2
    errors = abs(tri_out - ref) > atol
    b_mismatch = torch.argmax(errors.sum((1, 2, 3)).view(-1)).item()
    h_mismatch = torch.argmax(errors[b_mismatch].sum((1, 2)).view(-1)).item()

    torch.testing.assert_close(
        ref,
        tri_out,
        atol=atol,
        rtol=0,
        msg=lambda x: f"{x}\n\n{(b_mismatch, h_mismatch)}:\n{(errors[b_mismatch, h_mismatch]).long()} \n\n {(tri_out - ref)[errors].view(-1)}\n\nlens:\n{lens}\n{ref}\n{tri_out}",
    )
    for i, (d_ref, d_tri) in enumerate([(ref_dk, tri_dk), (ref_dv, tri_dv), (ref_dq, tri_dq)]):
        atol = 1e-3
        errors = abs(d_ref - d_tri) > atol
        b_mismatch = torch.argmax(errors.sum((1, 2, 3)).view(-1)).item()
        h_mismatch = torch.argmax(errors[b_mismatch].sum((1, 2)).view(-1)).item()

        mask = res_mask.broadcast_to(tri_out.shape)
        mean_err = (abs(d_ref[mask].to(torch.float32) - d_tri[mask].to(torch.float32)).mean() * 1000).item()

        torch.testing.assert_close(
            d_tri,
            d_ref,
            atol=atol,
            rtol=0,
            msg=lambda x: f"error in d{'kvq'[i]}\n{x}\n\n{(b_mismatch, h_mismatch)}:\n{(errors[b_mismatch, h_mismatch]).long()} \n\n {(d_tri - d_ref)[errors].view(-1)}\n\nlens:\n{lens}\n{mean_err = }"
        )
