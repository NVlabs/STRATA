# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    patch_ui = mo.ui.slider(0, 5, value=3, show_value=True, label="log_2(patch_size)")
    patch_ui
    return mo, patch_ui


@app.cell(hide_code=True)
def _(mo):
    kernel_size = mo.ui.slider(1, 5, value=1, label="kernel_size", show_value=True)
    decoder_type = mo.ui.dropdown(
        options=["conv_transpose", "bilinear"],
        value="conv_transpose",
        label="decoder_type",
    )
    kernel_size
    decoder_type
    return decoder_type, kernel_size


@app.cell(hide_code=True)
def _(c, eigenvalues, eigenvectors, np, patch_ui, s):
    import math

    def _():
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        fig, axes = plt.subplots(math.ceil(len(eigenvectors) / 2), 2, figsize=(10, 3))
        if len(eigenvectors) == 1:
            axes = [axes]

        axes = np.ravel(axes)
        fig.suptitle(f"Patch size = { 2 ** patch_ui.value}")
        xtick = 32

        # Hide spines on all axes
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for i, eig_vec in enumerate(eigenvectors):
            ax = axes[i]
            channel_0 = eig_vec.reshape(c, s)[0, :]
            ax.plot(channel_0.real, linewidth=1, label="Real", color="C0")
            ax.plot(channel_0.imag, linewidth=1, label="Imag", color="C1")
            ax.set_title(f"λ{i+1} = {eigenvalues[i]:.4f}")
            ax.set_xlabel("Spectral Index")
            ax.set_ylabel("Component Value")
            ax.set_xticks(xtick * np.arange(s // xtick))
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8, loc="upper left")

            # Make room for inset by adjusting y limits
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            ax.set_ylim(y_min, y_max + 1.0 * y_range)

            # Add power spectrum inset
            axins = inset_axes(ax, width="30%", height="30%", loc="upper right")
            power_spectrum = np.abs(np.fft.fft(channel_0)) ** 2
            freqs = np.fft.fftfreq(len(channel_0))
            axins.loglog(
                freqs[: len(freqs) // 2],
                power_spectrum[: len(power_spectrum) // 2],
                linewidth=0.8,
            )

            # Set power-of-2 ticks for x, power-of-10 for y
            pow2_ticks_x = [2.0**i for i in range(-5, 1)]
            pow10_ticks_y = [10.0**i for i in range(-10, 6)]
            axins.set_xticks(pow2_ticks_x)
            axins.set_yticks(pow10_ticks_y)
            axins.set_xticklabels([f"$2^{{{i}}}$" for i in range(-5, 1)], fontsize=6)
            axins.set_yticklabels([f"$10^{{{i}}}$" for i in range(-10, 6)], fontsize=6)

            axins.set_title("Power Spectrum", fontsize=8)
            axins.set_xlabel("Freq", fontsize=8)
            axins.set_ylabel("Power", fontsize=8)
            axins.tick_params(labelsize=6)
            axins.spines["top"].set_visible(False)
            axins.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        return fig

    _()
    return


@app.cell
def _(decoder_type, kernel_size, patch_ui):
    import torch
    from torch import nn

    _, c, s = 1, 1, 128

    class PatchUnpatch(torch.nn.Module):
        def __init__(self, patch_size=16, kernel_size=1, decoder="conv_transpose"):
            super().__init__()
            self.proj = nn.Conv1d(
                in_channels=c,
                out_channels=c,
                kernel_size=kernel_size,
                stride=patch_size,
                padding_mode="circular",
            )
            self.decoder = decoder
            self.patch_size = patch_size
            if decoder == "conv_transpose":
                self.unproj = nn.ConvTranspose1d(
                    in_channels=c,
                    out_channels=c,
                    kernel_size=kernel_size * patch_size,
                    stride=patch_size,
                    padding=patch_size // 2 if kernel_size == 2 else 0,
                )
            elif decoder == "bilinear":
                self.unproj = nn.Conv1d(c, c, kernel_size=1)

        def forward(self, x):
            x = self.proj(x)
            if self.decoder == "bilinear":
                # Handle complex interpolation by interpolating real and imaginary parts separately
                if torch.is_complex(x):
                    x_real = nn.functional.interpolate(
                        x.real, scale_factor=self.patch_size, mode="linear"
                    )
                    x_imag = nn.functional.interpolate(
                        x.imag, scale_factor=self.patch_size, mode="linear"
                    )
                    x = torch.complex(x_real, x_imag)
                else:
                    x = nn.functional.interpolate(
                        x, scale_factor=self.patch_size, mode="linear"
                    )
                x = self.unproj(x)
            else:
                x = self.unproj(x)
            return x

    torch.manual_seed(0)
    patch_size = 2**patch_ui.value
    mod = PatchUnpatch(
        patch_size=patch_size, kernel_size=kernel_size.value, decoder=decoder_type.value
    )
    return c, mod, s, torch


@app.cell
def _(c, mod, s, torch):
    import numpy as np

    # Manually convert float parameters to complex float
    for param in mod.parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.cfloat)

    def power_iteration(matvec_fn, size, num_iter=100):
        """Find the dominant eigenvalue and eigenvector using power iteration with complex support."""
        # Initialize with complex random vector
        v = np.random.randn(size) + 1j * np.random.randn(size)
        v = v / np.linalg.norm(v)

        for _ in range(num_iter):
            # Matrix-vector product
            v_new = matvec_fn(v)
            # Normalize
            v = v_new / np.linalg.norm(v_new)

        # Compute eigenvalue
        Av = matvec_fn(v)
        eigenvalue = np.dot(v.conj(), Av) / np.dot(v.conj(), v)
        return eigenvalue, v

    def create_matvec(module, shape):
        """Create a matrix-vector product function for the module supporting complex values."""

        def matvec(v):
            with torch.no_grad():
                v_tensor = torch.from_numpy(v).cfloat().reshape(shape).unsqueeze(0)
                out = module(v_tensor).squeeze(0).flatten()
                return out.numpy()

        return matvec

    # Create matvec function
    shape = (c, s)
    matvec_fn = create_matvec(mod, shape)
    size = np.prod(shape)

    # Find first few eigenvalues using power iteration with deflation
    num_eigenvalues = 2
    eigenvalues = []
    eigenvectors = []

    original_matvec = matvec_fn
    deflation_terms = []

    for k in range(num_eigenvalues):
        # Create deflated matvec with complex support
        def deflated_matvec(
            v, original_fn=original_matvec, deflation_list=deflation_terms
        ):
            result = original_fn(v)
            for eig_val, eig_vec in deflation_list:
                result = result - eig_val * np.dot(eig_vec.conj(), v) * eig_vec
            return result

        eig_val, eig_vec = power_iteration(deflated_matvec, size, num_iter=100)
        eigenvalues.append(eig_val)
        eigenvectors.append(eig_vec)
        deflation_terms.append((eig_val, eig_vec))

    print("Power Iteration Results for PatchUnpatch:")
    for i, eig_val in enumerate(eigenvalues):
        print(f"Eigenvalue {i+1}: {eig_val:.6f}")
    return eigenvalues, eigenvectors, np


if __name__ == "__main__":
    app.run()
