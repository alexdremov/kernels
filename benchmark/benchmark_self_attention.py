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

from self_attention import (self_attention,
                            self_attention_reference,
                            self_attention_reference_naive)

if __name__ == "__main__":
    batches = (64,)
    configs = []
    params = (
        [
            dict(
                batch=batch,
                dim=128,
                heads=6,
                name="dim-128",
            )
            for batch in batches
        ] + [
            dict(
                batch=batch,
                dim=256,
                heads=6,
                name="dim-256",
            )
            for batch in batches
        ]
    )
    for param in params:
        line_vals = [
            f"triton",
            f"torch-sdpa",
            f"torch-naive",
        ]
        dim = param["dim"]
        heads = param["heads"]
        batch = param["batch"]

        x_vals = np.linspace(257, 8000, 6).astype(int).tolist()
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
                ylabel="ms",
                plot_name=f"self-attention-{param['name']}-dim-{dim}-heads-{heads}-batch-{batch}",
                args=dict(
                    batch=batch,
                    heads=heads,
                    dim=dim,
                    dtype=torch.float16,
                ),
            )
        )

    @triton.testing.perf_report(configs)
    def bench_self_attention(
        provider, time, batch, heads, dim, dtype,
    ):
        device = "cuda"

        torch.set_float32_matmul_precision("highest")

        q = torch.randn((batch, heads, time, dim), dtype=dtype, device=device).normal_(
            mean=0.0, std=0.01
        )
        k = torch.randn((batch, heads, time, dim), dtype=dtype, device=device).normal_(
            mean=0.0, std=0.01
        )
        v = torch.randn((batch, heads, time, dim), dtype=dtype, device=device).normal_(
            mean=0.0, std=0.01
        )

        lens = None
        if "triton" in provider:
            fn = lambda: self_attention(q, k, v, lens)
        elif "torch-sdpa" in provider:
            def torch_test():
                return self_attention_reference(q, k, v, lens)[0]

            fn = torch_test
        elif "torch-naive" in provider:
            def torch_test():
                return self_attention_reference_naive(q, k, v, lens)[0]

            fn = torch_test
        else:
            assert False

        ref, res_mask = self_attention_reference(
            q, k, v, lens
        )
        ref, res_mask = ref.cuda(), res_mask.cuda()

        print(f"Starting {provider}")

        try:
            actual = fn()
        except torch.OutOfMemoryError:
            return 0

        actual = actual * res_mask.broadcast_to(actual.shape)

        atol = 3e-3
        torch.testing.assert_close(
            actual,
            ref,
            atol=atol,
            rtol=0,
            msg=lambda x: f"error in {provider}\n{x}",
        )

        with torch.inference_mode():
            try:
                ms = triton.testing.do_bench(
                    fn,
                    warmup=500,
                    rep=1000,
                    return_mode="mean",
                )
            except torch.OutOfMemoryError:
                return 0

        return ms
        total_flops = 4 * time * time * heads * dim * batch
        return (total_flops / (ms / 1000)) / 1e12

    bench_self_attention.run(save_path=".", print_data=True)
