# U-Net based cross-correlation optimization: 1D experiments

Code accompanying Sect. 3 of

> W. Yan, D. Gruen, O. Friedrich & A. Kovács,
> *Optimal cosmological cross-correlation analysis with U-Net-based processing*

A 1D U-Net is trained on pairs of simulated fields to transform an input tracer field
**I** into an output **O** that is maximally Pearson-correlated with a target field **T**.
For Gaussian fields the result is compared with the analytic linear estimator **L**
(matched filter); for log-normal fields the network is shown to use information beyond
the power spectrum.

| File | Content |
| --- | --- |
| `unet_tools.py` | shared code: grid and Fourier conventions, Gaussian random fields (Eqs. 10-13), the U-Net architectures (Fig. 1), the Pearson loss (Eq. 2), dataset generation, training, linear estimators (Eq. 7), correlation tables and plot helpers |
| `unet_gaussian.ipynb` | Sect. 3.1 and 3.2 (Gaussian input and target): Figs. 2, 3 and Tables 1, 2 |
| `unet_lognormal.ipynb` | Sect. 3.3 and 3.4 (log-normal fields): Figs. 4-7 and Tables 3, 4 |

Each section of a notebook is self-contained: run the setup cell at the top, then the
cells of one section in order. Figures are written to `figures/`, trained weights to
`*.pt`.

## Requirements

Python >= 3.9 with `numpy`, `scipy`, `matplotlib` and `torch`. Training uses a CUDA GPU
if available (on a CPU it is impractically slow). The training sets (50 000-200 000
fields of 4096 pixels) are generated in memory with all CPU cores and need up to ~7 GB.
