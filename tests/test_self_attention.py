import os

import numpy as np
import pytest
import torch

from self_attention import (
    self_attention,
    self_attention_reference
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=lambda x: f"{x}")
@pytest.mark.parametrize(
    "lens", ["none", "tricky", "random"], ids=lambda x: f"lens-{x}"
)
@pytest.mark.parametrize(
    "noncontiguous", [False, True], ids=lambda x: f"noncontiguous-{x}"
)
@pytest.mark.parametrize("HEAD_DIM", [16, 128, 256], ids=lambda x: f"dim-{x}")
@pytest.mark.parametrize("B", [1, 40, 64], ids=lambda x: f"batch-{x}")
@pytest.mark.parametrize("H", [1, 6, 8], ids=lambda x: f"heads-{x}")
@pytest.mark.parametrize("T", [1, 10, 16, 800, 1025], ids=lambda x: f"time-{x}")
@pytest.mark.parametrize(
    "autotune", [False, True], ids=lambda x: f"autotune-{x}"
)
def test_self_attention(
    B,
    H,
    T,
    HEAD_DIM,
    dtype,
    lens,
    noncontiguous,
    autotune,
):
    torch._dynamo.reset()

    torch.manual_seed(20)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.empty_cache()

    if os.environ.get("TRITON_INTERPRET") == "1" and dtype == torch.bfloat16:
        pytest.skip("skipping bf16 in interpreter mode")

    if autotune and not (
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
            low=-0.1,
            high=0.1
        )
        for _ in range(3)
    ]

    if noncontiguous:
        q = q[1::2, 1::2, 1::2, 1::2].detach().clone().requires_grad_()
        k = k[1::2, 1::2, 1::2, 1::2].detach().clone().requires_grad_()
        v = v[1::2, 1::2, 1::2, 1::2].detach().clone().requires_grad_()

    if lens == "none":
        lens = None
    elif lens == "tricky":
        tricky_lens = [
            1,
            2,
            5,
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

    ref, res_mask = self_attention_reference(
        q, k, v, lens
    )
    tri_out = self_attention(q, k, v, lens, autotune=autotune)

    # torch.set_printoptions(linewidth=400, profile="full")

    tri_out = tri_out * res_mask.broadcast_to(tri_out.shape)
    atol = 1e-3
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
