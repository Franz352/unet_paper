"""
unet_tools.py
=============

Shared code for the one-dimensional U-Net cross-correlation experiments of

    W. Yan, D. Gruen, O. Friedrich & A. Kovacs,
    "Optimal cosmological cross-correlation analysis with U-Net-based processing"

The two notebooks ``unet_gaussian.ipynb`` (Sect. 3.1, 3.2) and
``unet_lognormal.ipynb`` (Sect. 3.3, 3.4) import this module.

Contents
--------
1. Grid, Fourier conventions and Gaussian random fields   (Sect. 3.1.1, Eqs. 10-13)
2. Datasets of (input, target) pairs
3. Loss function                                           (Eq. 2)
4. The 1D U-Net architectures                              (Sect. 2.1, Fig. 1)
5. Training and inference
6. Linear estimators                                       (Sect. 2.2, Eq. 7)
7. Pearson-correlation tables                              (Tables 1-4)
8. Plot helpers                                            (Figs. 2-7)
"""

import multiprocessing
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 1. Grid, Fourier conventions and Gaussian random fields
# ---------------------------------------------------------------------------
N_grid = 4096                      # pixels per 1D field
N_grid_half = N_grid // 2
k_max = 3.0
dk = 2 * k_max / N_grid            # spacing of the FFT modes, mode j has k = j * dk
k_min = -k_max + dk
k_values = np.linspace(k_min, k_max, N_grid)          # k_values[N_grid_half - 1] == 0
k_positive = k_values[N_grid_half - 1:N_grid - 1]     # 0 <= k < k_max, one entry per FFT mode

# fiducial broken power law, Eq. (10)
k0 = 0.01
ns_low = 1.0
ns_high = -1.5

# high-frequency cutoff of the target fields: all modes with |k| > K_CUTOFF are
# removed, i.e. only the modes j = 0 ... N_MODES_KEPT (k = j dk <= 0.0293) survive
K_CUTOFF = 0.03
N_MODES_KEPT = 20

data_type = torch.float            # dtype of the fields handed to the network
data_type_c = np.cdouble           # dtype of the Fourier coefficients


def fourier_to_real(coeffs):
    """Fourier coefficients -> real-space field (``return_fourier_transform`` in the
    original analysis code)."""
    return np.fft.fft(coeffs) * dk / np.sqrt(2.0 * np.pi)


def real_to_fourier(field):
    """Real-space field -> Fourier coefficients (``return_fourier_backtransform`` in the
    original analysis code).  Entry j is the mode k = j dk for j < N_grid/2; the second
    half of the array holds the negative-k modes."""
    return np.fft.ifft(field) * np.sqrt(2.0 * np.pi) / dk


def power_spectrum_broken_power_law(k, amplitude=1.0, k_cutoff=None):
    """Eq. (10).  With ``k_cutoff`` the spectrum is set to zero for |k| > k_cutoff."""
    k_abs = np.abs(k)
    pk = amplitude * (k_abs / k0) ** ns_low / (1.0 + (k_abs / k0) ** (-ns_high + ns_low))
    if k_cutoff is not None:
        pk = np.where(k_abs > k_cutoff, 0.0, pk)
    return pk


def gaussian_random_field(seed, amplitude=1.0, k_cutoff=None):
    """Real-valued Gaussian random field on the N_grid pixels, Eqs. (11)-(13).

    The complex Fourier coefficients are drawn mode by mode with variance P(k); the
    negative-frequency modes follow from Hermitian symmetry.  Two calls with the same
    ``seed`` use the same underlying normal deviates, so e.g.
    ``gaussian_random_field(s)`` and ``gaussian_random_field(s, k_cutoff=K_CUTOFF)``
    share every mode below the cutoff.
    """
    rng = np.random.RandomState(seed)
    sigma = np.sqrt(power_spectrum_broken_power_law(k_positive, amplitude, k_cutoff))
    coeffs = np.zeros(N_grid, dtype=data_type_c)
    coeffs[:N_grid_half] += np.sqrt(0.5) * rng.normal(np.zeros(N_grid_half), sigma)
    coeffs[:N_grid_half] += np.sqrt(0.5) * 1.0j * rng.normal(np.zeros(N_grid_half), sigma)
    coeffs[N_grid_half + 1:] = np.flip(coeffs[1:N_grid_half].conj())
    return fourier_to_real(coeffs).real


def estimate_powerspectrum(field):
    """|delta(k)|^2 for the positive modes j = 1 ... N_grid/2 - 1."""
    return np.abs(real_to_fourier(field)[1:N_grid_half]) ** 2


# ---------------------------------------------------------------------------
# 2. Datasets of (input, target) pairs
# ---------------------------------------------------------------------------
class PairDataset(Dataset):
    """Pairs (I, T) of 1D fields stored as float tensors of shape [n, 1, N_grid]."""

    def __init__(self, inputs, targets):
        self.data = inputs
        self.labels = targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# The pair-generating function is handed to the worker processes through this module
# global, which forked workers inherit.  Nothing has to be pickled, so make_pair may be
# defined anywhere, e.g. in a notebook cell.
_fork_make_pair = None


def _pair_worker(i, seed):
    return _fork_make_pair(seed + i)


def generate_dataset(n, make_pair, seed=0, n_workers=None):
    """Generate ``n`` pairs, in parallel where the 'fork' start method is available.

    ``make_pair(seed)`` must return the tuple ``(I, T)`` of two numpy arrays of length
    N_grid; pair number ``i`` of the dataset is generated with ``seed + i``.
    """
    global _fork_make_pair
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), n)
    inputs = torch.zeros((n, N_grid), dtype=data_type)
    targets = torch.zeros((n, N_grid), dtype=data_type)

    def store(i, pair):
        inputs[i] = torch.from_numpy(np.asarray(pair[0], dtype=np.float32))
        targets[i] = torch.from_numpy(np.asarray(pair[1], dtype=np.float32))

    if n_workers > 1 and "fork" in multiprocessing.get_all_start_methods():
        _fork_make_pair = make_pair
        try:
            with multiprocessing.get_context("fork").Pool(n_workers) as pool:
                for i, pair in enumerate(pool.imap(partial(_pair_worker, seed=seed), range(n), chunksize=16)):
                    store(i, pair)
        finally:
            _fork_make_pair = None
    else:
        for i in range(n):
            store(i, make_pair(seed + i))
    return PairDataset(inputs.unsqueeze(1), targets.unsqueeze(1))


def rescale_dataset(dataset, scale):
    """Divide inputs AND targets by one global constant (non-mutating).

    The raw Gaussian fields have a standard deviation of a few 1e-3, far below the
    weight scale of a freshly initialised network; bringing them to unit variance makes
    the optimisation start in a sensible regime.  The loss is scale free, and one common
    factor leaves every amplitude ratio between I and T unchanged.
    """
    return PairDataset(dataset.data / scale, dataset.labels / scale)


def get_fields(dataset, idx):
    """Return (I, T) as numpy arrays: 1D for an integer ``idx``, 2D stacks for an iterable."""
    if np.isscalar(idx):
        I, T = dataset[int(idx)]
        return I.squeeze().cpu().numpy(), T.squeeze().cpu().numpy()
    I = np.stack([dataset[int(i)][0].squeeze().cpu().numpy() for i in idx])
    T = np.stack([dataset[int(i)][1].squeeze().cpu().numpy() for i in idx])
    return I, T


# ---------------------------------------------------------------------------
# 3. Loss function
# ---------------------------------------------------------------------------
def cross_var_loss(output, target, eps=1e-8):
    """Negative Pearson correlation between output and target, Eq. (2), averaged over
    the batch.  Tensors have shape [batch, 1, N_grid].

    The means are not subtracted explicitly: every target field used in the paper has
    zero mean by construction (the k = 0 mode of the generated fields vanishes, and the
    log-normal targets are mean-subtracted), so that for each sample
    mean(O T) / (std(O) std(T)) is exactly the Pearson coefficient.
    """
    output_std = torch.std(output, dim=2, keepdim=True) + eps
    target_std = torch.std(target, dim=2, keepdim=True) + eps
    return -torch.mean((output / output_std) * (target / target_std))


def pearson(x, y, eps=1e-8):
    """Pearson correlation coefficient of two 1D numpy arrays (Eq. 2 without the sign)."""
    x = (x - np.mean(x)) / (np.std(x) + eps)
    y = (y - np.mean(y)) / (np.std(y) + eps)
    return float(np.mean(x * y))


# ---------------------------------------------------------------------------
# 4. The 1D U-Net architectures
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """1D U-Net of Fig. 1 (depth 3, channels 64/128/256, kernels 321/161/81).

    Valid convolutions and max-pooling in the encoder, transposed convolutions in the
    decoder, two skip connections, ReLU after every layer including the output.
    Feature-map lengths for a 4096-pixel input: 3776 / 1728 / 784 / 1728 / 3776 / 4096.
    """

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder1 = nn.Sequential(nn.Conv1d(in_channels, 64, kernel_size=321), nn.ReLU(inplace=True))
        self.encoder2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=161), nn.ReLU(inplace=True))
        self.encoder3 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=81), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.decoder1 = nn.Sequential(nn.ConvTranspose1d(256, 128, kernel_size=162, stride=2), nn.ReLU(inplace=True))
        self.decoder2 = nn.Sequential(nn.ConvTranspose1d(256, 64, kernel_size=322, stride=2), nn.ReLU(inplace=True))
        self.decoder3 = nn.Sequential(nn.ConvTranspose1d(128, out_channels, kernel_size=321, bias=False), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.encoder1(x)                              # [B,  64, 3776]
        x2 = self.encoder2(self.pool(x1))                  # [B, 128, 1728]
        x3 = self.encoder3(self.pool(x2))                  # [B, 256,  784]
        d1 = torch.cat((self.decoder1(x3), x2), dim=1)     # [B, 256, 1728]
        d2 = torch.cat((self.decoder2(d1), x1), dim=1)     # [B, 128, 3776]
        return self.decoder3(d2)                           # [B,   1, 4096]


class UNetCircular(nn.Module):
    """Variant of ``UNet`` with periodic boundary conditions.

    Same channels, kernel sizes, pooling and skip connections as ``UNet``, but every
    convolution is made length preserving by explicit circular padding (the fields are
    generated by FFT and are therefore exactly periodic), and resolution is recovered by
    nearest-neighbour interpolation followed by a convolution instead of a transposed
    convolution.  Feature-map lengths: 4096 / 2048 / 1024 / 2048 / 4096.

    Parameters
    ----------
    negative_slope : slope of the LeakyReLU after the hidden layers (0.0 = plain ReLU).
    linear_output  : if True the output layer has no activation (required when the target
                     field can be negative); if False the output is passed through the same
                     (Leaky)ReLU as the hidden layers.
    """

    def __init__(self, in_channels=1, out_channels=1, negative_slope=0.0, linear_output=False):
        super().__init__()
        self.encoder1 = nn.Conv1d(in_channels, 64, kernel_size=321)
        self.encoder2 = nn.Conv1d(64, 128, kernel_size=161)
        self.encoder3 = nn.Conv1d(128, 256, kernel_size=81)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.decoder1 = nn.Conv1d(256, 128, kernel_size=161)                      # on the up-sampled bottleneck
        self.decoder2 = nn.Conv1d(256, 64, kernel_size=161)                       # on cat(up(d1), x2)
        self.decoder3 = nn.Conv1d(128, out_channels, kernel_size=81, bias=False)  # on cat(up(d2), x1)
        self.negative_slope = negative_slope
        self.linear_output = linear_output

    def _cconv(self, layer, x, act=True):
        """Circular 'same' convolution, optionally followed by the activation."""
        p = (layer.kernel_size[0] - 1) // 2
        y = layer(F.pad(x, (p, p), mode="circular"))
        return F.leaky_relu(y, self.negative_slope) if act else y

    def forward(self, x):
        x1 = self._cconv(self.encoder1, x)                       # [B,  64, N]
        x2 = self._cconv(self.encoder2, self.pool(x1))           # [B, 128, N/2]
        x3 = self._cconv(self.encoder3, self.pool(x2))           # [B, 256, N/4]
        d1 = F.interpolate(x3, scale_factor=2, mode="nearest")   # [B, 256, N/2]
        d1 = torch.cat((self._cconv(self.decoder1, d1), x2), dim=1)
        d2 = F.interpolate(d1, scale_factor=2, mode="nearest")   # [B, 256, N]
        d2 = torch.cat((self._cconv(self.decoder2, d2), x1), dim=1)
        return self._cconv(self.decoder3, d2, act=not self.linear_output)   # [B, 1, N]


# ---------------------------------------------------------------------------
# 5. Training and inference
# ---------------------------------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    for X, y in loader:
        total += loss_fn(model(X.to(device)), y.to(device)).item()
    return total / len(loader)


def train_model(model, train_set, val_set, epochs, lr, batch_size, loss_fn=cross_var_loss,
                num_workers=4, log_every=500, device=None):
    """Train with Adam on ``train_set`` and report the validation loss after every epoch."""
    device = device or get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=10, shuffle=False,
                            num_workers=min(2, num_workers), pin_memory=pin)
    n_train = len(train_set)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            loss = loss_fn(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % log_every == 0:
                print(f"  loss: {loss.item():>9.6f}  [{batch * len(X):>6d}/{n_train:>6d}]")
        print(f"  validation loss: {evaluate_loss(model, val_loader, loss_fn, device):.6f}")
    return model


@torch.no_grad()
def predict(model, fields, batch_size=20, device=None):
    """Apply the network to numpy fields of shape [N_grid] or [n, N_grid]; returns numpy
    arrays of the same shape."""
    device = device or next(model.parameters()).device
    model.eval()
    x = np.atleast_2d(np.asarray(fields, dtype=np.float32))
    out = []
    for start in range(0, len(x), batch_size):
        xb = torch.tensor(x[start:start + batch_size], dtype=data_type).unsqueeze(1).to(device)
        out.append(model(xb).squeeze(1).cpu().numpy())
    out = np.concatenate(out)
    return out[0] if np.ndim(fields) == 1 else out


# ---------------------------------------------------------------------------
# 6. Linear estimators
# ---------------------------------------------------------------------------
def low_pass_filter(field, n_modes=N_MODES_KEPT):
    """Low-pass filtered field: the Fourier coefficients with index
    n_modes + 1 ... N_grid - n_modes - 2 are set to zero and the real part of the
    back-transform is returned.  This is the optimal linear estimator W(k) = Theta(k) of
    Sect. 3.1 / 3.2 (the same slice as in the original analysis, which also keeps the
    negative mode -(n_modes + 1))."""
    coeffs = real_to_fourier(field)
    coeffs[n_modes + 1:N_grid - n_modes - 1] = 0.0
    return fourier_to_real(coeffs).real


def measure_spectra(dataset, n=None):
    """Auto spectra P_II, P_TT, cross spectrum P_IT (all N_grid-long, over every FFT
    mode) averaged over the first ``n`` pairs, and the optimal linear filter
    W = P_IT / P_II of Eq. (7)."""
    n = len(dataset) if n is None else n
    P_II = np.zeros(N_grid)
    P_TT = np.zeros(N_grid)
    P_IT = np.zeros(N_grid)
    for i in range(n):
        I, T = get_fields(dataset, i)
        I_k = real_to_fourier(I.astype(np.float64))
        T_k = real_to_fourier(T.astype(np.float64))
        P_II += np.abs(I_k) ** 2
        P_TT += np.abs(T_k) ** 2
        P_IT += (I_k * np.conj(T_k)).real   # the imaginary part cancels between +k and -k
    P_II /= n
    P_TT /= n
    P_IT /= n
    W = np.zeros(N_grid)
    nz = P_II > 0
    W[nz] = P_IT[nz] / P_II[nz]
    return P_II, P_TT, P_IT, W


def apply_filter(field, W):
    """Linear estimator L of Eq. (3): multiply the Fourier coefficients of ``field`` by
    the filter ``W`` (N_grid-long, e.g. from ``measure_spectra``) and transform back."""
    return fourier_to_real(real_to_fourier(field) * W).real


# ---------------------------------------------------------------------------
# 7. Pearson-correlation tables
# ---------------------------------------------------------------------------
def correlation_table(dataset, estimators, n=200):
    """Mean and standard deviation over the first ``n`` pairs of the Pearson correlation
    between each estimator and the target T.

    ``estimators`` maps a label to a function ``f(I)`` returning an estimate of T from
    the input (use ``lambda I: I`` for the raw input).  Returns {label: (mean, std)}.
    """
    values = {label: [] for label in estimators}
    for i in range(n):
        I, T = get_fields(dataset, i)
        for label, estimate in estimators.items():
            values[label].append(pearson(estimate(I), T))
    summary = {label: (float(np.mean(v)), float(np.std(v))) for label, v in values.items()}
    print(f"Pearson correlation with T, mean +- std over {n} realizations")
    for label, (mean, std) in summary.items():
        print(f"  {label:>4s} vs T : {mean:.3f} +- {std:.3f}")
    return summary


# ---------------------------------------------------------------------------
# 8. Plot helpers
# ---------------------------------------------------------------------------
TICK_SIZE, LABEL_SIZE, LEGEND_SIZE = 20, 20, 16

LABEL_I = r"$\mathbf{I}$nput"
LABEL_T = r"$\mathbf{T}$arget"
LABEL_L = r"$\mathbf{L}$inear estimator"
LABEL_O = r"$\mathbf{O}$utput"

STYLE_I = dict(alpha=0.7, linestyle="--", linewidth=1.5)
STYLE_T = dict(alpha=0.7, linestyle=":", linewidth=2.0)
STYLE_L = dict(alpha=0.7, linestyle="-.", linewidth=3.0)
STYLE_O = dict(alpha=0.7, linestyle="-", linewidth=2.0)
STYLE_RESIDUAL = dict(color="black", alpha=0.8, linewidth=2.0)


def set_paper_style():
    """Serif text and Computer Modern math, so that the labels match the LaTeX body
    font (matplotlib's bundled cmr10, no LaTeX installation needed)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,     # cmr10 has no U+2212 glyph
    })


def save_figure(fig, filename, output_dir="figures"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, format="pdf", bbox_inches="tight")
    print(f"Saved figure to: {output_dir / filename}")


def normalize(x):
    """Zero mean, unit variance."""
    return (x - np.mean(x)) / np.std(x)


def shifted(label, offset):
    return label + r" (shifted by $+%g$)" % offset


def plot_fields(curves, residuals=None, baseline=None, figsize=None, legend_size=LEGEND_SIZE):
    """Real-space comparison of fields (Figs. 2a, 3a, 4a, 6, 7a).

    ``curves`` and ``residuals`` are lists of ``(y, label, style)`` with ``style`` a dict
    of ``plt.plot`` keyword arguments; the residuals go into a lower panel.
    ``baseline`` draws a thin horizontal line, e.g. at the offset applied to a curve.
    """
    if residuals:
        fig, (ax, ax_res) = plt.subplots(2, 1, figsize=figsize or (18, 7), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    else:
        fig, ax = plt.subplots(figsize=figsize or (18, 6))
        ax_res = None
    for y, label, style in curves:
        ax.plot(y, label=label, **style)
    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)
    ax.legend(fontsize=legend_size)
    ax.set_ylabel("Value", fontsize=LABEL_SIZE)
    if ax_res is not None:
        for y, label, style in residuals:
            ax_res.plot(y, label=label, **style)
        ax_res.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax_res.legend(loc="upper right", fontsize=legend_size)
        ax_res.set_ylabel("Residual", fontsize=LABEL_SIZE)
        ax_res.set_xlabel("Index", fontsize=LABEL_SIZE)
    else:
        ax.set_xlabel("Index", fontsize=LABEL_SIZE)
    for a in (ax, ax_res):
        if a is not None:
            a.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    return fig


def mean_powerspectrum(fields, scales=None):
    """Mean power spectrum of a stack of realizations.  Every field is first normalized
    to zero mean and unit variance and then multiplied by the matching entry of
    ``scales`` (default 1)."""
    if scales is None:
        scales = np.ones(len(fields))
    return np.mean([estimate_powerspectrum(normalize(f) * s) for f, s in zip(fields, scales)], axis=0)


def plot_power_spectra(I_ps, T_ps, O_ps, kmax=100, k_norm=5, ymin=None, floor_factor=1e5):
    """Power spectra of input, target and output on a log scale (Figs. 2b, 3b, 4b, 7b).

    The input spectrum is rescaled to coincide with the target at mode ``k_norm`` so
    that I and T lie on top of each other below the cutoff.  The lower y limit defaults
    to the smallest (rescaled) input amplitude in the plotted range divided by
    ``floor_factor``: the target is band limited, so beyond the cutoff its estimate is
    FFT round-off that would otherwise dominate the log axis.
    """
    I_ps = I_ps * (T_ps[k_norm] / I_ps[k_norm])
    fig = plt.figure(figsize=(10, 6))
    plt.plot(I_ps, label=LABEL_I, alpha=0.7, linestyle="--", linewidth=2.0, zorder=3)
    plt.plot(T_ps, label=LABEL_T, alpha=0.9, linestyle=":", linewidth=3.0, zorder=5)
    plt.plot(O_ps, label=LABEL_O, alpha=0.9, linestyle="-", linewidth=2.0, zorder=4)
    plt.yscale("log")
    plt.xlabel("k", fontsize=LABEL_SIZE)
    plt.ylabel("Power Spectrum", fontsize=LABEL_SIZE)
    plt.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    plt.xlim(0, kmax)
    sel = slice(0, kmax + 1)
    if ymin is None:
        ymin = 0.5 * I_ps[sel].min() / floor_factor
    plt.ylim(ymin, 2.0 * max(I_ps[sel].max(), T_ps[sel].max(), O_ps[sel].max()))
    plt.legend(fontsize=LEGEND_SIZE)
    return fig


def plot_spectra_and_filter(P_II, P_TT, P_IT, W):
    """Measured auto/cross spectra and the resulting filter W = P_IT / P_II (Eq. 7)."""
    j = np.arange(1, N_grid_half)
    k_pos = j * dk
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(k_pos, P_II[j], label=r"$P_{II}$", linewidth=2.0)
    ax1.plot(k_pos, P_TT[j], label=r"$P_{TT}$", linewidth=2.0, linestyle="--")
    ax1.plot(k_pos, np.abs(P_IT[j]), label=r"$|P_{IT}|$", linewidth=2.0, linestyle="-.")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$k$", fontsize=LABEL_SIZE)
    ax1.set_ylabel(r"$P(k)$", fontsize=LABEL_SIZE)
    ax1.legend(fontsize=LEGEND_SIZE)
    ax2.plot(k_pos, W[j], linewidth=2.0)
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$k$", fontsize=LABEL_SIZE)
    ax2.set_ylabel(r"$W(k) = P_{IT} / P_{II}$", fontsize=LABEL_SIZE)
    for a in (ax1, ax2):
        a.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    return fig


def binned_mean(x, y, n_bins=50):
    """Mean of ``y`` in equally spaced bins of ``x``; returns (bin centres, means)."""
    bins = np.linspace(np.min(x), np.max(x), n_bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(x, bins)
    means = np.array([np.mean(y[idx == i]) if np.any(idx == i) else np.nan
                      for i in range(1, len(bins))])
    return centers, means


def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


def fit_parabola(centers, means, x_range=(-3.0, 2.5)):
    """Least-squares parabola through the binned means inside ``x_range`` (Eq. 14).
    Returns the coefficients (a, b, c) and the x values used for the fit."""
    from scipy.optimize import curve_fit
    mask = (centers > x_range[0]) & (centers < x_range[1]) & np.isfinite(means)
    popt, _ = curve_fit(parabola, centers[mask], means[mask])
    return popt, centers[mask]


def plot_hist2d(x, y, centers=None, means=None, fit=None,
                xlabel=r"$\mathbf{I}$", ylabel=r"$\mathbf{O}$", legend_size=LEGEND_SIZE):
    """2D histogram of two normalized fields with the binned mean and an optional fit
    curve ``fit = (x_fit, y_fit, label)`` (Fig. 5)."""
    fig = plt.figure(figsize=(8, 6))
    plt.hist2d(x, y, bins=200, cmap="Blues")
    cbar = plt.colorbar()
    cbar.set_label("Counts", size=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    if centers is not None:
        plt.plot(centers, means, "o-", color="green", linewidth=2, markersize=5,
                 label=r"Mean $\mathbf{O}$ per bin")
    if fit is not None:
        x_fit, y_fit, label = fit
        plt.plot(x_fit, y_fit, "r-", linewidth=2, label=label)
    plt.xlabel(xlabel, fontsize=LABEL_SIZE)
    plt.ylabel(ylabel, fontsize=LABEL_SIZE)
    plt.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    plt.legend(fontsize=legend_size)
    return fig
