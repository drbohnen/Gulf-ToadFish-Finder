"""
gtfMatchedFilterDet_ver1.py  —  Python port of gtfMatchedFilterDet_ver1.m

Spectrogram cross-correlation detector for Gulf toadfish boatwhistle calls.

Key MATLAB → Python mappings
-----------------------------
spectrogram(y, W, OL, nFFT, fs)   →  scipy.signal.spectrogram (hamming window)
conv2(A, rot90(B,2), 'full')       →  fftconvolve(A, B[::-1,::-1], 'full')
medfilt1(P, win, 'truncate', 1)    →  ndimage.median_filter(..., mode='nearest')
movmean(x, win, 'Endpoints','shrink') →  explicit shrink-window mean loop
findpeaks(...)                     →  scipy.signal.find_peaks(...)
interp1(xw, yw, F, 'linear')      →  np.interp(F, xw, yw)
"""

import warnings

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
from scipy.signal import fftconvolve, find_peaks


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _shrink_movmean(x: np.ndarray, win: int) -> np.ndarray:
    """Centered moving mean with shrinking window at edges (MATLAB 'Endpoints','shrink')."""
    half = win // 2
    n    = len(x)
    return np.array([x[max(0, i - half): i + half + 1].mean() for i in range(n)])


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

def gtf_matched_filter_det(
    y,
    frange,
    s,
    sweep,
    thres,
    predffreq,
    predffreq_uncert,
    ploton: bool = False,
):
    """
    Spectrogram cross-correlation detector for Gulf toadfish boatwhistle calls.

    Parameters
    ----------
    y : array-like
        1-D audio at 24 kHz.
    frange : (float, float)
        Fundamental-frequency search range [f_low, f_high] Hz.
    s : float
        Frequency spread (Hz) for the Mexican-hat kernel.
    sweep : float
        Boatwhistle frequency sweep (Hz); F1b = F1a − sweep.
    thres : float
        Detection threshold for normalised cross-correlation scores.
    predffreq : float
        Predicted fundamental frequency (Hz) — used to weight kernel selection.
    predffreq_uncert : float
        Uncertainty in predffreq (Hz).
    ploton : bool
        If True, generate a 5-panel diagnostic figure.

    Returns
    -------
    det_time, det_freq1, det_score, l_detscore, u_detscore, fo_calls, f1_calls
        All 1-D numpy arrays.  All empty if no detections.
    """
    _empty = tuple(np.array([]) for _ in range(7))

    # ---- constants (must match MATLAB) ----------------------------------
    W    = 4096
    fs   = 24_000
    np2  = 13          # nFFT = 2^13 = 8192
    OL   = int(np.floor(W * 0.8))
    nFFT = 2 ** np2
    eps  = np.finfo(np.float64).eps

    y = np.asarray(y, dtype=np.float64).ravel()
    print(f'  [det ENTRY] ny={len(y)}  frange=[{frange[0]:.1f} {frange[1]:.1f}]  predffreq={predffreq:.1f}')

    # ---- spectrogram ----------------------------------------------------
    F, T, Pxx = signal.spectrogram(
        y, fs=fs,
        window=signal.windows.hamming(W),
        nperseg=W, noverlap=OL, nfft=nFFT,
        scaling='density',
    )
    nT = len(T)
    print(f'  [det SPEC] nT={nT}  nF_full={len(F)}')
    if nT == 0:
        print('  [det] EARLY RETURN: nT==0')
        return _empty

    # ---- trim band generously: [frange[0]-3s, 2*frange[1]+3s] ----------
    band = (F >= (frange[0] - 3 * s)) & (F <= (2 * frange[1] + 3 * s))
    F    = F[band]
    Pxx  = Pxx[band, :]
    nF   = len(F)
    print(f'  [det BAND] nF_trimmed={nF}  bandHz=[{frange[0]-3*s:.1f} {2*frange[1]+3*s:.1f}]')
    if nF == 0:
        print('  [det] EARLY RETURN: nF==0')
        return _empty

    # ---- build F1 (inclusive edges) and F2 ------------------------------
    F1a = F[(F >= frange[0]) & (F <= frange[1])]
    print(f'  [det F1a] nkerns={len(F1a)}')
    if len(F1a) == 0:
        print('  [det] EARLY RETURN: F1a empty')
        return _empty
    F1b    = F1a - sweep
    F1     = np.vstack([F1a, F1b])    # (2, nkerns)
    F2     = 2 * F1
    nkerns = F1.shape[1]

    # ---- percussive filtering + background subtraction (dB) -------------
    PxxMod = 10.0 * np.log10(Pxx + eps)

    winMed = max(3, round(nF / 6))
    if winMed % 2 == 0:
        winMed += 1

    # Display copy: background-subtracted only (before percussive filter)
    bg_disp        = _shrink_movmean(np.median(PxxMod, axis=1), winMed)
    PxxMod_display = PxxMod - bg_disp[:, np.newaxis]

    # 1) percussive removal along frequency axis
    PxxMedC = median_filter(PxxMod, size=(winMed, 1), mode='nearest')
    PxxMod  = PxxMod - PxxMedC

    # 2) background subtraction across time (per frequency bin)
    bg     = _shrink_movmean(np.median(PxxMod, axis=1), winMed)
    PxxMod = PxxMod - bg[:, np.newaxis]

    # ---- kernel time grid -----------------------------------------------
    dt = (T[1] - T[0]) if nT >= 2 else (W - OL) / fs

    klengthsec = 0.334
    n_steps    = int(np.floor(klengthsec / dt)) + 1
    t          = np.arange(n_steps) * dt
    len_k      = len(t)
    alpha      = t / max(t[-1], eps)    # 0 → 1

    # ---- allocate correlation matrices ----------------------------------
    n_full    = nT + len_k - 1
    cmatrix   = np.zeros((nkerns, n_full), dtype=np.float32)
    u_cmatrix = np.zeros((nkerns, n_full), dtype=np.float32)
    l_cmatrix = np.zeros((nkerns, n_full), dtype=np.float32)
    oneK      = np.ones(len_k, dtype=np.float32)

    # ---- loop over kernels ----------------------------------------------
    F_col     = F[:, np.newaxis]        # (nF, 1)  for broadcasting
    alpha_row = alpha[np.newaxis, :]    # (1, len_k)

    for j in range(nkerns):
        # Upper harmonic kernel  (2*F1)
        fo2, f12 = F2[0, j], F2[1, j]
        X2  = F_col - (fo2 + alpha_row * (f12 - fo2))   # (nF, len_k)
        u_k = (1 - X2**2 / s**2) * np.exp(-X2**2 / (2 * s**2))

        # Lower (fundamental) kernel  (F1)
        fo1, f11 = F1[0, j], F1[1, j]
        X1  = F_col - (fo1 + alpha_row * (f11 - fo1))   # (nF, len_k)
        l_k = (1 - X1**2 / s**2) * np.exp(-X1**2 / (2 * s**2))

        k = u_k + l_k   # combined kernel  (nF, len_k)

        # Trim ±3s Hz around each band centre for efficiency
        u_trim = (F > fo2 - 3 * s) & (F < fo2 + 3 * s)
        l_trim = (F > fo1 - 3 * s) & (F < fo1 + 3 * s)
        if not np.any(u_trim) or not np.any(l_trim):
            continue
        u_P = PxxMod[u_trim, :];  uK = u_k[u_trim, :]
        l_P = PxxMod[l_trim, :];  lK = l_k[l_trim, :]

        # 2D cross-correlation along time:
        #   MATLAB: conv2(A, rot90(B,2), 'full')[size(B,1), :]
        #   Python: fftconvolve(A, B[::-1,::-1], 'full')[B.shape[0]-1, :]
        c   = fftconvolve(PxxMod, k[::-1, ::-1],   mode='full')[k.shape[0]  - 1, :]
        u_c = fftconvolve(u_P,    uK[::-1, ::-1], mode='full')[uK.shape[0] - 1, :]
        l_c = fftconvolve(l_P,    lK[::-1, ::-1], mode='full')[lK.shape[0] - 1, :]

        u_knorm    = np.sqrt(np.sum(uK ** 2)) + eps
        l_knorm    = np.sqrt(np.sum(lK ** 2)) + eps

        E_u        = np.convolve(np.sum(u_P ** 2, axis=0).astype(np.float32), oneK, 'full')
        E_l        = np.convolve(np.sum(l_P ** 2, axis=0).astype(np.float32), oneK, 'full')
        u_norm     = np.sqrt(E_u) + eps
        l_norm     = np.sqrt(E_l) + eps
        PnormComb  = np.sqrt(E_l + E_u) + eps
        kComb_norm = np.sqrt(np.sum(lK ** 2) + np.sum(uK ** 2)) + eps

        cmatrix[j, :]   = (c   / (PnormComb  * kComb_norm)).astype(np.float32)
        u_cmatrix[j, :] = (u_c / (u_norm     * u_knorm  )).astype(np.float32)
        l_cmatrix[j, :] = (l_c / (l_norm     * l_knorm  )).astype(np.float32)

    # ---- trim to nT columns (discard ramp-up) ---------------------------
    # MATLAB: cmatrix(:, len_k:end)  →  Python: [:, len_k-1:]
    cmatrix   = cmatrix[:,   len_k - 1:]
    u_cmatrix = u_cmatrix[:, len_k - 1:]
    l_cmatrix = l_cmatrix[:, len_k - 1:]
    if cmatrix.shape[1] == 0:
        return _empty

    # ---- frequency prior taper ------------------------------------------
    predffreq_orig   = predffreq
    predffreq        = float(np.clip(predffreq, frange[0], frange[1]))
    predffreq_uncert = max(predffreq_uncert, 25.0)
    if predffreq != predffreq_orig:
        warnings.warn(
            f'predffreq {predffreq_orig:.1f} Hz outside frange [{frange[0]:.1f} {frange[1]:.1f}]'
            f' — clamped to {predffreq:.1f} Hz.',
            RuntimeWarning,
        )

    lowcut  = predffreq - predffreq_uncert
    highcut = predffreq + predffreq_uncert

    #  flat-top region [lowcut, highcut], exponential decay outside, floor 0.25
    xprd = np.arange(1, 21) * 2.94     # 20 decay steps covering ~59 Hz
    yprd = np.exp(-np.arange(1, 21) / 15)
    xw   = np.concatenate([[-12000.0], np.flip(lowcut - xprd),
                            [lowcut, highcut], highcut + xprd, [12000.0]])
    yw   = np.concatenate([[0.25], np.flip(yprd), [1.0, 1.0], yprd, [0.25]])
    w    = np.interp(F1[0, :], xw, yw)
    w    = _shrink_movmean(w, 5)

    # ---- pick best kernel per time step ---------------------------------
    # w weights kernel selection AND scales cout (off-freq calls suppressed)
    weighted = cmatrix * w[:, np.newaxis]   # (nkerns, nT)
    iout     = np.argmax(weighted, axis=0)  # (nT,)  0-indexed
    t_idx    = np.arange(cmatrix.shape[1])
    cout     = cmatrix[iout, t_idx].astype(np.float64) * w[iout]
    u_cout   = u_cmatrix[iout, t_idx].astype(np.float64)
    l_cout   = l_cmatrix[iout, t_idx].astype(np.float64)

    # ---- diagnostic output ----------------------------------------------
    print(f'  [det] nF={nF}  nT={nT}  nkerns={nkerns}  len_k={len_k}')
    print(f'  [det] max(cmatrix)={np.max(cmatrix):.4f}  max(l_cmatrix)={np.max(l_cmatrix):.4f}'
          f'  max(u_cmatrix)={np.max(u_cmatrix):.4f}')
    print(f'  [det] max(cout)={np.max(cout):.4f}  max(l_cout)={np.max(l_cout):.4f}'
          f'  max(u_cout)={np.max(u_cout):.4f}')
    print(f'  [det] predffreq={predffreq:.1f}  frange=[{frange[0]:.1f} {frange[1]:.1f}]')

    # ---- peak picking ---------------------------------------------------
    min_peak_dist = max(4, round(klengthsec / dt))

    # Combined peaks — candidates for f1_calls (both harmonics present)
    LOCS_c, _ = find_peaks(cout,   height=thres,
                            distance=min_peak_dist, prominence=thres / 5)
    # Lower-band-only peaks — fo_calls (fundamental only)
    LOCS_l, _ = find_peaks(l_cout, height=thres,
                            distance=min_peak_dist, prominence=thres / 5)

    # Remove LOCS_l entries already covered by a combined peak
    if len(LOCS_l) > 0 and len(LOCS_c) > 0:
        D      = np.abs(LOCS_l[:, np.newaxis] - LOCS_c[np.newaxis, :])
        LOCS_l = LOCS_l[np.all(D > 9, axis=1)]

    LOCS_all = np.sort(np.concatenate([LOCS_c, LOCS_l]))
    print(f'  [det] nLOCS_c={len(LOCS_c)}  nLOCS_l={len(LOCS_l)}'
          f'  nLOCS_all={len(LOCS_all)}  thres={thres:.3f}')
    if len(LOCS_all) == 0:
        return _empty

    # ---- score / time / freq arrays -------------------------------------
    scores   = cout[LOCS_all]
    u_scores = u_cout[LOCS_all]
    l_scores = l_cout[LOCS_all]
    times    = T[LOCS_all]
    freqs    = F1[0, iout[LOCS_all]]

    # ---- call-type logic ------------------------------------------------
    from_combined = np.isin(LOCS_all, LOCS_c)
    f1_calls = from_combined & (l_scores > thres) & (u_scores > thres * 0.8)
    fo_calls = (l_scores > thres) & ~f1_calls

    # ---- final prune: keep rows with a detection ------------------------
    mask   = fo_calls | f1_calls
    locs_m = LOCS_all[mask]

    det_time   = times[mask]
    det_freq1  = freqs[mask]
    det_score  = scores[mask]
    u_detscore = u_scores[mask]
    l_detscore = l_scores[mask]
    fo_calls   = fo_calls[mask]
    f1_calls   = f1_calls[mask]

    # ---- optional diagnostic plots --------------------------------------
    if ploton:
        import matplotlib.pyplot as plt
        try:
            from gtfMakespectro_ver1 import _PARULA
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list('parula', _PARULA)
        except ImportError:
            cmap = 'viridis'

        fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

        # 1: background-subtracted spectrogram
        axes[0].imshow(PxxMod_display,
                       extent=[T[0], T[-1], F[0], F[-1]],
                       aspect='auto', origin='lower', vmin=-35, vmax=35, cmap=cmap)
        axes[0].set_ylabel('Hz')
        axes[0].set_title('spectrogram')

        # 2: percussive-filtered spectrogram with detections
        axes[1].imshow(PxxMod,
                       extent=[T[0], T[-1], F[0], F[-1]],
                       aspect='auto', origin='lower', vmin=-25, vmax=25, cmap=cmap)
        axes[1].set_ylabel('Hz')
        axes[1].set_title('percussive-filtered spectrogram with detections')
        if len(det_time) > 0:
            axes[1].plot(det_time[fo_calls], det_freq1[fo_calls],
                         '+r', markersize=4, linewidth=1)
            axes[1].plot(det_time[f1_calls], det_freq1[f1_calls],
                         'ok', markersize=4, linewidth=1)

        # 3: detection scores over time
        axes[2].plot(T, cout,   'r',  label='cout')
        axes[2].plot(T, l_cout, 'g.', label='l_cout', markersize=2)
        axes[2].plot(T, u_cout, 'b',  label='u_cout')
        if len(locs_m) > 0:
            axes[2].plot(T[locs_m], det_score, 'or')
        axes[2].set_ylim([0.1, 0.6])
        axes[2].set_ylabel('det score')
        axes[2].grid(True)
        axes[2].legend(fontsize=7)

        # 4: correlation matrix (kernel vs time)
        axes[3].imshow(cmatrix,
                       extent=[T[0], T[-1], F1[0, 0], F1[0, -1]],
                       aspect='auto', origin='lower', vmin=-0.5, vmax=0.5, cmap=cmap)
        axes[3].set_ylabel('F0 of kernel')
        axes[3].set_title('correlation with kernels (weighted)')

        # 5: correlation vs kernel freq for each candidate
        for jj in range(len(locs_m)):
            axes[4].plot(T[locs_m[jj]] + cmatrix[:, locs_m[jj]], F1[0, :])
        axes[4].set_title('correlation vs kernel freq for candidates')
        axes[4].set_xlabel('time')
        axes[4].set_ylabel('F0 of kernel')

        plt.tight_layout()
        plt.show(block=False)

    return det_time, det_freq1, det_score, l_detscore, u_detscore, fo_calls, f1_calls
