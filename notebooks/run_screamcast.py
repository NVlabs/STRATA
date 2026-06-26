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

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    import marimo as mo

    root_path = mo.notebook_dir().parent.as_posix()
    print(root_path)
    sys.path.append(root_path)

    from data_catalog import scream

    return mo, scream


@app.cell
def _():
    import dotenv

    dotenv.load_dotenv()

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from screamcast.earth2studio_wrappers import ScreamcastModel

    return ScreamcastModel, np, plt, torch


@app.cell
def _(ScreamcastModel):
    # Point checkpoint_path at your local checkpoint copy — see "Global Rollout"
    # in README.md for the rclone command that fetches the production checkpoint.
    checkpoint_path = "../pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_unfreeze_3src_const1em5_t128_4stepft/output/best.pth"
    domain_size = 256

    model = ScreamcastModel.from_checkpoint(checkpoint_path, bf16=True)
    model = model.cuda()
    model.eval()
    model.set_tile_size(domain_size)
    model.compile()

    in_coords = dict(model.input_coords())
    print(f"Loaded model from {checkpoint_path}")
    print(f"  tile_size={domain_size}, in_channels={len(in_coords['variable'])}")
    return domain_size, in_coords, model


@app.cell
def _(domain_size, in_coords, np, scream):
    # Load data source and fetch initial condition for face 0
    ds = scream()

    t0 = np.datetime64("2020-10-13T00:00:00")
    face = np.array([0])

    fetch_coords = {
        "time": np.array([t0]),
        "variable": list(in_coords["variable"]),
        "face": face,
        "x": np.arange(domain_size),
        "y": np.arange(domain_size),
    }

    ic_da = ds(fetch_coords)
    x_np = ic_da.values  # (1, C, 1, tile, tile)
    print(f"Fetched initial condition: shape={x_np.shape}, time={t0}, face=0")
    return ds, face, t0, x_np


@app.cell
def _(in_coords):
    in_coords["variable"]
    return


@app.cell
def _(in_coords, mo, plt, x_np):
    # Plot initial condition — lowest available qv level (highest index = near surface)
    def plot():
        all_vars = list(in_coords["variable"])
        qv_var = [v for v in all_vars if v.startswith("qv_")][-1]
        qv_vi = all_vars.index(qv_var)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(x_np[0, qv_vi, 0], cmap="Blues", origin="lower")
        ax.set_title(f"IC: {qv_var} (face 0)", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        return qv_var, mo.mpl.interactive(fig)

    qv_var, out = plot()
    out
    return (qv_var,)


@app.cell
def _(face, in_coords, model, np, t0, torch, x_np):
    # Run 12 forecast steps
    n_steps = 12
    x_tensor = torch.from_numpy(x_np).cuda()  # (1, C_in, 1, tile, tile)

    coords_model = dict(in_coords)
    coords_model["face"] = face
    coords_model["batch"] = np.empty(1)
    coords_model["time"] = np.array([t0])

    dt_minutes = model.dt / np.timedelta64(1, "s") / 60
    all_outputs = []  # list of (1, C_out, 1, H, W) numpy arrays, one per step
    out_coords = None
    with torch.inference_mode():
        iterator = model.create_iterator(x_tensor, coords_model)
        next(iterator)  # initial condition at lead time 0
        for _step in range(n_steps):
            out_tensor, out_coords = next(iterator)
            all_outputs.append(out_tensor.cpu().numpy())
            print(f"step {_step + 1}/{n_steps}", end="\r")

    print(f"\nForecast done: {n_steps} steps = {dt_minutes * n_steps / 60:.1f} h")
    return all_outputs, dt_minutes, out_coords


@app.cell
def _(mo, out_coords, qv_var):
    def make_selector():
        out_var_names = list(out_coords["variable"])
        return mo.ui.dropdown(options=out_var_names, value=qv_var, label="Variable")

    var_selector = make_selector()
    var_selector
    return (var_selector,)


@app.cell
def _(all_outputs, dt_minutes, mo, out_coords, plt, var_selector):
    import tempfile

    from matplotlib.animation import FuncAnimation

    def render():
        out_var_names = list(out_coords["variable"])
        vi = out_var_names.index(var_selector.value)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.axis("off")
        vmin = min(o[0, vi, 0].min() for o in all_outputs)
        vmax = max(o[0, vi, 0].max() for o in all_outputs)
        if "omega" in var_selector.value:
            vlim = max(abs(vmin), abs(vmax))
            vmin, vmax = -vlim, vlim
        im = ax.imshow(
            all_outputs[0][0, vi, 0],
            cmap="RdBu_r",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        title = ax.set_title("")

        def update(step):
            im.set_data(all_outputs[step][0, vi, 0])
            title.set_text(
                f"{var_selector.value}  +{dt_minutes * (step + 1) / 60:.1f} h"
            )
            return [im, title]

        anim = FuncAnimation(fig, update, frames=len(all_outputs), blit=True)
        # agent: save to mp4 and use mo.video
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        anim.save(tmp.name, writer="ffmpeg", fps=5)
        plt.close(fig)
        return mo.video(open(tmp.name, "rb").read())

    render()
    return


@app.cell
def _(all_outputs, mo):
    # agent: for the select channel and timestep add a comparison between the truth and model-drive forecast
    step_slider = mo.ui.slider(1, len(all_outputs), value=1, label="Forecast step")
    step_slider
    return (step_slider,)


@app.cell(hide_code=True)
def _(
    all_outputs,
    domain_size,
    ds,
    dt_minutes,
    face,
    mo,
    np,
    out_coords,
    plt,
    step_slider,
    t0,
    var_selector,
):
    def _():
        step = step_slider.value
        out_var_names = list(out_coords["variable"])
        vi = out_var_names.index(var_selector.value)
        # Fetch truth at the forecast time
        t = t0 + np.timedelta64(int(dt_minutes * step), "m")
        v = out_coords["variable"][vi : vi + 1]
        truth = ds(
            {
                "time": np.array([t]),
                "variable": v,
                "face": face,
                "x": np.arange(domain_size),
                "y": np.arange(domain_size),
            }
        )
        truth = truth.squeeze()
        forecast = all_outputs[step - 1][0, vi, 0]  # (H, W)

        vmin = min(truth.min(), forecast.min())
        vmax = max(truth.max(), forecast.max())
        if "omega" in var_selector.value:
            vlim = max(abs(vmin), abs(vmax))
            vmin, vmax = -vlim, vlim

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
        for ax, data, label in [(ax1, truth, "Truth"), (ax2, forecast, "Forecast")]:
            im = ax.imshow(data, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(
                f"{label}: {var_selector.value}\n+{dt_minutes * step / 60:.1f} h"
            )
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        diff = forecast - truth
        dlim = max(abs(diff.min()), abs(diff.max())) or 1.0
        im3 = ax3.imshow(diff, cmap="RdBu_r", origin="lower", vmin=-dlim, vmax=dlim)
        ax3.set_title("Forecast - Truth")
        ax3.axis("off")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        plt.tight_layout()
        return mo.mpl.interactive(fig)

    _()
    return


if __name__ == "__main__":
    app.run()
