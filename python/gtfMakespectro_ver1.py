"""
gtfMakespectro.py  —  Python port of gtfMakespectro_e.m

Port of the MATLAB function used to generate 224x224x3 spectrogram images
for the gulf toadfish CNN classifier (ResNet50 fine-tuned).

PARULA COLORMAP
--------------
The _PARULA array below is built from extracted control points via PCHIP
interpolation and is approximate.  For pixel-exact match with MATLAB, export
the colormap once:

    % in MATLAB
    writematrix(parula(256), 'parula256.csv')

then pass it to the function:

    cmap = gtf_load_parula_csv('parula256.csv')
    im, lims = gtf_make_spectro(v, cmap=cmap)

VALIDATION
----------
Run both the MATLAB and Python functions on the same audio clip and compare
the resulting images before using at inference.

Dependencies
------------
    numpy, scipy, pillow
"""

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation
from PIL import Image


# ---------------------------------------------------------------------------
# Parula colormap
# ---------------------------------------------------------------------------

def _build_parula_approx() -> np.ndarray:
    """
    Generate approximate parula(256) from known control points (PCHIP).
    Rows 0-32 and row 255 are extracted from MATLAB; rows 48-240 are
    approximate — replace with gtf_load_parula_csv() for exact values.
    """
    # fmt: off
    pts = np.array([
        #  idx    R        G        B
        [   0,  0.2081,  0.1663,  0.5292],  # exact
        [   4,  0.1959,  0.2474,  0.6952],  # exact
        [   7,  0.0591,  0.3113,  0.8135],  # exact  (blue peak)
        [   9,  0.0072,  0.3413,  0.7795],  # exact
        [  10,  0.0162,  0.3692,  0.7575],  # exact
        [  16,  0.0643,  0.4913,  0.6425],  # exact
        [  20,  0.0870,  0.5448,  0.5740],  # exact
        [  27,  0.2026,  0.6530,  0.4734],  # exact
        [  32,  0.2771,  0.6926,  0.4181],  # exact
        [  48,  0.3947,  0.7210,  0.3053],  # approx
        [  64,  0.5148,  0.7238,  0.1841],  # approx
        [  80,  0.6429,  0.7124,  0.0583],  # approx
        [  96,  0.7411,  0.6919,  0.0070],  # approx
        [ 128,  0.8290,  0.6619,  0.0001],  # approx
        [ 160,  0.9104,  0.6227,  0.0575],  # approx
        [ 192,  0.9466,  0.5998,  0.1203],  # approx
        [ 224,  0.9554,  0.7765,  0.1393],  # approx
        [ 240,  0.9659,  0.8934,  0.0596],  # approx
        [ 255,  0.9769,  0.9839,  0.0805],  # exact
    ], dtype=np.float64)
    # fmt: on

    t_ctrl = pts[:, 0] / 255.0
    t_all  = np.arange(256) / 255.0
    R = np.clip(PchipInterpolator(t_ctrl, pts[:, 1])(t_all), 0.0, 1.0)
    G = np.clip(PchipInterpolator(t_ctrl, pts[:, 2])(t_all), 0.0, 1.0)
    B = np.clip(PchipInterpolator(t_ctrl, pts[:, 3])(t_all), 0.0, 1.0)
    return np.column_stack([R, G, B])


_PARULA: np.ndarray = _build_parula_approx()


def gtf_load_parula_csv(path: str) -> np.ndarray:
    """
    Load exact parula(256) exported from MATLAB.

    In MATLAB:
        writematrix(parula(256), 'parula256.csv')

    Parameters
    ----------
    path : str
        Path to the CSV file produced by MATLAB's writematrix.

    Returns
    -------
    cmap : ndarray, shape (256, 3), float64 in [0, 1]
    """
    cmap = np.loadtxt(path, delimiter=',')
    if cmap.shape != (256, 3):
        raise ValueError(f"Expected 256×3 colormap, got {cmap.shape}.")
    return cmap


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def gtf_make_spectro(
    v24,
    *,
    out_for: str = 'net',
    output_size: tuple = (224, 224),
    scale: str = 'robust',
    min_input: float | None = None,
    max_input: float | None = None,
    cmap: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Make a 224×224×3 spectrogram image for the gulf toadfish CNN.

    Matches gtfMakespectro_e.m step-for-step:
      1. Pad / truncate to 1.1 s at 24 kHz
      2. Hamming-windowed STFT → one-sided PSD → dB
      3. Bandpass 50–1400 Hz
      4. Percussive-component removal (medfilt along frequency)
      5. Background subtraction (median across time, smoothed in frequency)
      6. Flip so low frequencies are at the bottom
      7. Robust ±2σ scaling → parula colormap → resize to 224×224

    Parameters
    ----------
    v24 : array-like
        1-D audio samples recorded at 24 kHz.
    out_for : {'net', 'file', 'double'}
        Output dtype.  'net' / 'file' → uint8 [0, 255] (matches JPEG read
        via imageDatastore).  'double' → float64 [0, 1].
    output_size : (H, W)
        Target image size in pixels.  Default (224, 224).
    scale : {'robust', 'fixed'}
        Colour scaling mode.
    min_input, max_input : float, optional
        Required when scale='fixed'.
    cmap : ndarray (256, 3), optional
        Colormap table.  Defaults to the built-in approximate parula.
        Pass gtf_load_parula_csv('parula256.csv') for exact MATLAB match.

    Returns
    -------
    im : ndarray
        uint8 [H, W, 3] for out_for='net'/'file', or float64 [H, W, 3]
        for out_for='double'.
    lims : (float, float)
        (min_input, max_input) actually used for scaling.
    """
    if cmap is None:
        cmap = _PARULA

    # ---- constants (must match MATLAB training script) ------------------
    Fs    = 24_000
    DurS  = 1.1
    Nreq  = round(Fs * DurS)            # 26 400
    nFFT  = 2 ** 12                      # 4 096
    W     = nFFT // 2                    # 2 048  (window length)
    OL    = int(np.floor(W * 0.80))      # 1 638  (overlap)
    Fmin  = 50
    Fmax  = 1400

    # ---- length enforce -------------------------------------------------
    v = np.asarray(v24, dtype=np.float64).ravel()
    if v.size < Nreq:
        v = np.concatenate([v, np.zeros(Nreq - v.size)])
    elif v.size > Nreq:
        v = v[:Nreq]

    # ---- spectrogram → dB ----------------------------------------------
    # MATLAB: spectrogram(v, W, OL, nFFT, Fs) uses hamming(W) window
    f, _, P = signal.spectrogram(
        v,
        fs=Fs,
        window=signal.windows.hamming(W),
        nperseg=W,
        noverlap=OL,
        nfft=nFFT,
        scaling='density',      # PSD, matches MATLAB default
        mode='psd',
    )

    # Bandpass
    k = (f >= Fmin) & (f <= Fmax)
    P = P[k, :]

    # pow2db: 10*log10(P + eps)  — matches MATLAB pow2db(P + eps('like',P))
    P = 10.0 * np.log10(P + np.finfo(np.float64).eps)

    # ---- percussive removal + background subtraction --------------------
    nFreq  = P.shape[0]
    winMed = max(3, round(nFreq / 5))
    if winMed % 2 == 0:
        winMed += 1

    # medfilt1(P, winMed, 'truncate', 1) — along frequency axis
    # mode='nearest' is the closest scipy equivalent to MATLAB 'truncate'
    Pmed = median_filter(P, size=(winMed, 1), mode='nearest')
    P    = P - Pmed

    # movmean(median(P,2), winMed, 'Endpoints','shrink')
    # median across time (axis 1) → [nFreq]; then centered moving mean
    # that shrinks the window at the edges (min_periods=1 equivalent)
    med_freq = np.median(P, axis=1)
    half = winMed // 2
    bg = np.array([
        med_freq[max(0, i - half): i + half + 1].mean()
        for i in range(nFreq)
    ])
    P = P - bg[:, np.newaxis]

    # ---- flip low frequency to bottom (matches MATLAB flipud) -----------
    P = P[::-1, :]

    # ---- scale limits ---------------------------------------------------
    if scale.lower() == 'fixed':
        if min_input is None or max_input is None:
            raise ValueError("scale='fixed' requires min_input and max_input.")
        mininput = float(min_input)
        maxinput = float(max_input)
    else:  # robust
        sigma    = 1.4826 * float(median_abs_deviation(P.ravel()))
        mininput = float(max(-15.0, -2.0 * sigma))
        maxinput = float(min( 15.0,  2.0 * sigma))
    lims = (mininput, maxinput)

    # ---- map to colormap ------------------------------------------------
    # Matches: im2uint8(rescale(P, 0,1, 'InputMin',mininput,'InputMax',maxinput))
    P_clip = np.clip(P, mininput, maxinput)
    P_norm = (P_clip - mininput) / (maxinput - mininput)
    Iidx   = np.round(P_norm * 255).astype(np.uint8)   # 0–255

    # ind2rgb equivalent (uint8 → 0-based index into 256-row colormap)
    Irgb = cmap[Iidx]    # [nFreq, nTime, 3] float64 [0, 1]

    # ---- resize (matches MATLAB imresize default: bicubic) --------------
    H, W_out = output_size
    img_pil = Image.fromarray((Irgb * 255).astype(np.uint8))
    img_pil = img_pil.resize((W_out, H), Image.BICUBIC)
    Iout = np.array(img_pil)    # uint8 [H, W, 3]

    # ---- output dtype ---------------------------------------------------
    if out_for.lower() in ('net', 'file'):
        im = Iout                                # uint8 [0, 255]
    elif out_for.lower() == 'double':
        im = Iout.astype(np.float64) / 255.0    # float64 [0, 1]
    else:
        raise ValueError("out_for must be 'net', 'file', or 'double'.")

    return im, lims
