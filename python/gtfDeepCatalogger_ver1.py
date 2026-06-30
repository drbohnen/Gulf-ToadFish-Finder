"""
gtfDeepCatalogger_ver1.py  —  Python port of gtfDeepCatalogger_ver1.m

Per-segment matched-filter detector + deep-net (ResNet50) classifier.

classifier dict fields
----------------------
    net      : torch.nn.Module   (fine-tuned ResNet50, 2-class head)
    bestThr  : float             (decision threshold on P(bwhistle))
    posIdx   : int               (column index of bwhistle in softmax, = 1)

Returns
-------
Bcount, Ocount, Btable, Otable, Dtable
    Bcount / Ocount : int
    B/O/Dtable      : pandas.DataFrame (empty if nothing found)
"""

import io
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from gtfMakespectro_ver1 import gtf_make_spectro, gtf_load_parula_csv, _PARULA
from gtfMatchedFilterDet_ver1 import gtf_matched_filter_det

# ImageNet normalisation — must match training transforms in gtfDeepToadFinetuneScript.py
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# Load exact parula colormap once at import time.
# Must match what gtfParseTrainJpg.py used when building train_jpg_py — if
# parula256.csv was present then (it was), inference must use it too.
_PARULA_CSV = Path(__file__).parent / 'parula256.csv'
_CMAP = gtf_load_parula_csv(str(_PARULA_CSV)) if _PARULA_CSV.exists() else _PARULA
if not _PARULA_CSV.exists():
    import warnings
    warnings.warn('parula256.csv not found — using approximate colormap. '
                  'Classifier accuracy may be degraded.', RuntimeWarning)

# ---- column schemas -------------------------------------------------------
_BTAB_COLS = ['Site','CallID','ffreq','ccscore','l_ccscore','u_ccscore',
              'f1_calls','pdscore','rel_time','abs_time','segN','seg_min','file']
_DTAB_COLS = ['Site','ffreq','ccscore','l_ccscore','u_ccscore',
              'fo_calls','f1_calls','rel_time','abs_time','segN','seg_min','file']


def _empty_btab():
    return pd.DataFrame(columns=_BTAB_COLS)


def _empty_dtab():
    return pd.DataFrame(columns=_DTAB_COLS)


# ---- internal helpers -----------------------------------------------------

def _spectro_tensor(ydet: np.ndarray) -> torch.Tensor:
    """
    Spectrogram → normalised float32 tensor shape (3, 224, 224).

    Uses the exact parula256.csv colormap and a quality-95 JPEG round-trip
    so pixel values match what the network saw during training.
    """
    im, _ = gtf_make_spectro(ydet, out_for='net', cmap=_CMAP)  # uint8 (224, 224, 3)
    buf = io.BytesIO()
    Image.fromarray(im).save(buf, format='JPEG', quality=95)
    buf.seek(0)
    pil = Image.open(buf).convert('RGB')
    return _EVAL_TF(pil)                                        # (3, 224, 224)


def _batch_predict(v24, det_in_samples_f1, k1, k2,
                   half1Win, winLen, net, pos_idx, device):
    """
    Build spectrogram batch for samples[k1:k2], run net, return P(bwhistle).
    """
    det_slice = det_in_samples_f1[k1:k2]
    nv        = len(v24)

    tensors = []
    for samp in det_slice:
        win_start = max(0, samp - half1Win)
        win_end   = min(nv, win_start + winLen)
        if (win_end - win_start) < winLen:
            win_start = max(0, win_end - winLen)
        tensors.append(_spectro_tensor(v24[win_start:win_end]))

    batch = torch.stack(tensors).to(device)   # (nb, 3, 224, 224)
    net.eval()
    with torch.no_grad():
        logits = net(batch)
        probs  = F.softmax(logits, dim=1)[:, pos_idx].cpu().numpy()

    return probs.astype(np.float32)


def _write_images(v24, det_in_samples_f1, det_time_f1, half1Win, winLen,
                  out_dir_o, file_base, minute_start, parula_cmap):
    """Write all F1-candidate spectrograms as JPEGs into out_dir_o."""
    nv = len(v24)
    for j, samp in enumerate(det_in_samples_f1):
        win_start = max(0, samp - half1Win)
        win_end   = min(nv, win_start + winLen)
        if (win_end - win_start) < winLen:
            win_start = max(0, win_end - winLen)
        ydet = v24[win_start:win_end]

        im, _    = gtf_make_spectro(ydet, out_for='file', scale='robust', cmap=parula_cmap)
        img_name = f'{file_base}_{minute_start:03.0f}_{det_time_f1[j]:011.7f}.jpg'
        Image.fromarray(im).save(str(out_dir_o / img_name), format='JPEG', quality=90)


# ---- main function --------------------------------------------------------

def gtf_deep_catalogger(
    v24, site, segN, fileN, rootdir, classifier,
    Frange, predffreq, predffreq_uncert,
    fstartdatetime, NSEC,
    save_images: bool = False,
    keep_images: str  = 'all',
    device=None,
):
    """
    Run the matched-filter detector on one 2-min audio segment, then pass
    high-confidence f1 candidates through the fine-tuned ResNet50 classifier.

    Parameters
    ----------
    v24 : np.ndarray
        1-D audio at 24 kHz.
    site, segN, fileN : str
        Site name, zero-padded segment number ('001'), source filename.
    rootdir : str or Path
        Root output directory for JPEG images.
    classifier : dict
        {'net': nn.Module, 'bestThr': float, 'posIdx': int}
    Frange : (float, float)
        Frequency search range [f_low, f_high] Hz.
    predffreq : float
        Predicted fundamental frequency (Hz).
    predffreq_uncert : float
        Uncertainty (Hz).
    fstartdatetime : datetime or pd.Timestamp
        UTC start time of this segment.
    NSEC : float
        Segment duration (s).
    save_images : bool
    keep_images : str   'all' | 'bwhistle' | 'other' | 'none'
    device : torch.device or None

    Returns
    -------
    Bcount, Ocount, Btable, Otable, Dtable
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    keep_images  = keep_images.lower()
    minute_start = (int(segN) - 1) * (NSEC / 60)

    # ---- detector params (must match MATLAB) ----------------------------
    s = 10; sweep = 6; ploton = False; thres = 0.45
    fs       = 24_000
    winLen   = round(1.1 * fs)    # 26400 samples
    half1Win = round(0.4 * fs)    # 9600  samples (lookback)

    v24 = np.asarray(v24, dtype=np.float64).ravel()

    # ---- matched-filter detector ----------------------------------------
    det_times_mf, ffreq, ccscore, l_detscore, u_detscore, fo_calls, f1_calls = \
        gtf_matched_filter_det(v24, Frange, s, sweep, thres,
                               predffreq, predffreq_uncert, ploton)

    if len(det_times_mf) == 0:
        return 0, 0, _empty_btab(), _empty_btab(), _empty_dtab()

    # ---- timing conversions ---------------------------------------------
    half2Win       = winLen - half1Win
    nv             = len(v24)
    det_in_samples = np.round(fs * det_times_mf).astype(int)

    # Clip detections too close to the end of the segment
    too_late = (det_in_samples + half2Win) >= nv
    det_in_samples[too_late] = max(1, nv - (winLen + 1))
    det_time = det_in_samples / fs

    if not isinstance(fstartdatetime, pd.Timestamp):
        fstartdatetime = pd.Timestamp(fstartdatetime, tz='UTC')
    elif fstartdatetime.tzinfo is None:
        fstartdatetime = fstartdatetime.tz_localize('UTC')
    det_time2 = [fstartdatetime + pd.Timedelta(seconds=float(t)) for t in det_time]

    # ---- D table (all detector candidates) ------------------------------
    n_det  = len(det_times_mf)
    Dtable = pd.DataFrame({
        'Site'      : [str(site)]   * n_det,
        'ffreq'     : ffreq,
        'ccscore'   : ccscore,
        'l_ccscore' : l_detscore,
        'u_ccscore' : u_detscore,
        'fo_calls'  : fo_calls.astype(bool),
        'f1_calls'  : f1_calls.astype(bool),
        'rel_time'  : det_time,
        'abs_time'  : det_time2,
        'segN'      : [str(segN)]   * n_det,
        'seg_min'   : [minute_start] * n_det,
        'file'      : [str(fileN)]  * n_det,
    })
    Dtable = Dtable.sort_values('abs_time').reset_index(drop=True)

    # ---- F1 subset: strong enough for deep-net scoring ------------------
    isF1              = f1_calls.astype(bool) & (l_detscore > thres * 1.1) & (ccscore > thres * 0.8)
    det_in_samples_f1 = det_in_samples[isF1]
    det_time_f1       = det_time[isF1]
    det_time2_f1      = [t for t, m in zip(det_time2, isF1) if m]
    ffreq_f1          = ffreq[isF1]
    ccscore_f1        = ccscore[isF1]
    l_detscore_f1     = l_detscore[isF1]
    u_detscore_f1     = u_detscore[isF1]

    if len(det_in_samples_f1) == 0:
        return 0, 0, _empty_btab(), _empty_btab(), Dtable

    # ---- batched deep-net inference -------------------------------------
    net      = classifier['net']
    pos_idx  = classifier['pos_idx']
    best_thr = classifier['bestThr']
    BATCH    = 128

    nF1  = len(det_in_samples_f1)
    pd_b = np.zeros(nF1, dtype=np.float32)
    for k1 in range(0, nF1, BATCH):
        k2 = min(k1 + BATCH, nF1)
        pd_b[k1:k2] = _batch_predict(
            v24, det_in_samples_f1, k1, k2, half1Win, winLen,
            net, pos_idx, device)

    # ---- classify -------------------------------------------------------
    is_bwhistle = pd_b >= best_thr
    bb = np.where(is_bwhistle)[0]
    oo = np.where(~is_bwhistle)[0]
    Bcount = len(bb)
    Ocount = len(oo)

    def _make_tab(idx_arr, label):
        n = len(idx_arr)
        if n == 0:
            return _empty_btab()
        df = pd.DataFrame({
            'Site'      : [str(site)]    * n,
            'CallID'    : [label]        * n,
            'ffreq'     : ffreq_f1[idx_arr],
            'ccscore'   : ccscore_f1[idx_arr],
            'l_ccscore' : l_detscore_f1[idx_arr],
            'u_ccscore' : u_detscore_f1[idx_arr],
            'f1_calls'  : np.ones(n, dtype=int),
            'pdscore'   : pd_b[idx_arr].astype(float),
            'rel_time'  : det_time_f1[idx_arr],
            'abs_time'  : [det_time2_f1[i] for i in idx_arr],
            'segN'      : [str(segN)]    * n,
            'seg_min'   : [minute_start] * n,
            'file'      : [str(fileN)]   * n,
        })
        return df.sort_values('abs_time').reset_index(drop=True)

    Btable = _make_tab(bb, 'bwhistle')
    Otable = _make_tab(oo, 'other')

    # ---- optional JPEG output -------------------------------------------
    if save_images:
        parula_cmap = _CMAP   # loaded at module import from parula256.csv

        file_base = Path(fileN).stem
        out_dir   = Path(rootdir) / file_base
        out_dir_o = out_dir / 'other'
        out_dir_b = out_dir / 'bwhistle'
        out_dir_o.mkdir(parents=True, exist_ok=True)
        out_dir_b.mkdir(parents=True, exist_ok=True)

        # Write all F1 candidates to 'other' folder first
        _write_images(v24, det_in_samples_f1, det_time_f1,
                      half1Win, winLen, out_dir_o, file_base,
                      minute_start, parula_cmap)

        # Move bwhistle classifications
        if keep_images in ('all', 'bwhistle') and Bcount > 0:
            for _, row in Btable.iterrows():
                img_name = f'{file_base}_{minute_start:03.0f}_{row["rel_time"]:011.7f}.jpg'
                src = out_dir_o / img_name
                dst = out_dir_b / img_name
                if src.exists():
                    src.rename(dst)

        if keep_images == 'bwhistle':
            try:
                shutil.rmtree(str(out_dir_o))
            except Exception:
                pass

    return Bcount, Ocount, Btable, Otable, Dtable
