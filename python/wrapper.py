"""
wrapper.py  —  Python port of wrapper.m

Loops over recordings in a site master index, runs the matched-filter
detector and fine-tuned ResNet50 classifier on every 2-minute segment,
and writes per-file CSV result tables.

===========================================================================
ONE-TIME SETUP — export MATLAB master tables to CSV (run once in MATLAB):
---------------------------------------------------------------------------
    % For each site, in MATLAB:
    t = tt_BK_20180109;                    % load the mat if not already in workspace
    t.t = string(t.t, 'yyyy-MM-dd HH:mm:ss.SSS');
    writetable(t, 'tt_BK_20180109.csv');

The CSV needs columns: t (ISO datetime string), fs, fname, nsamp
===========================================================================

Dependencies
------------
    numpy scipy soundfile pandas torch torchvision pillow tqdm
"""

import warnings
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from torchvision.models import resnet50
from tqdm import tqdm

from gtfDeepCatalogger_ver1 import gtf_deep_catalogger

# ===========================================================================
# Config — edit this block
# ===========================================================================

SITE = 'JB201801099'      # << change site here

SITE_CFG = {
    'BK20180109': dict(
        dir_in   = Path(r'K:\FLA\BK_20180109'),
        dir_out  = Path(r'..\betaout_py\GTFdet_BK_20180109'),
        wt_csv   = Path(r'K:\FLA\FLA_temperature\BK_gov-nps-ever-bkyf1_a830_1301_7798.csv'),
        master   = Path(r'..\timetables_csv\tt_BK_20180109.csv'),
    ),
    'BK20180726': dict(
        dir_in   = Path(r'K:\FLA\BK_20180726'),
        dir_out  = Path(r'..\betaout_py\GTFdet_BK_20180726'),
        wt_csv   = Path(r'K:\FLA\FLA_temperature\BK_gov-nps-ever-bkyf1_a830_1301_7798.csv'),
        master   = Path(r'..\timetables_csv\tt_BK_20180726.csv'),
    ),
    'JB20180109': dict(
        dir_in   = Path(r'K:\FLA\JB_20180109'),
        dir_out  = Path(r'..\betaout_py\GTFdet_JB_20180109'),
        wt_csv   = Path(r'K:\FLA\FLA_temperature\JB_gov-nps-ever-jbyf1_a393_7ce3_e678.csv'),
        master   = Path(r'..\timetables_csv\tt_JB_20180109.csv'),
    ),
    'JB20151130': dict(
        dir_in   = Path(r'K:\FLA\JB_20151130'),
        dir_out  = Path(r'..\betaout_py\GTFdet_JB_20151130'),
        wt_csv   = Path(r'K:\FLA\FLA_temperature\JB_gov-nps-ever-jbyf1_a393_7ce3_e678.csv'),
        master   = Path(r'..\timetables_csv\tt_JB_20151130.csv'),
    ),
    'LM20180109': dict(
        dir_in   = Path(r'K:\FLA\LM_20180109'),
        dir_out  = Path(r'..\betaout_py\GTFdet_LM_20180109'),
        wt_csv   = Path(r'K:\FLA\FLA_temperature\LM_gov-nps-ever-lmdf1_a830_1301_7798.csv'),
        master   = Path(r'..\timetables_csv\tt_LM_20180109.csv'),
    ),
}

CLASSIFIER_PATH = Path(__file__).parent / 'gtfclassifier_tuned_py.pt'

NSEC         = 120          # segment length (s)
FS_OUT       = 24_000       # target sample rate
NFFT_POW     = 13           # matches spectrogram bin grid (2^13)
PREDF0_UNCERT = 50          # Hz uncertainty on predicted F0
SAVE_IMAGES  = True         # write JPEG spectrograms per detection
KEEP_IMAGES  = 'all'        # 'all' | 'bwhistle' | 'other' | 'none'

# ===========================================================================
# Helpers
# ===========================================================================

def _load_classifier(path: Path, device: torch.device) -> dict:
    """
    Load the PyTorch classifier saved by gtfDeepToadFinetuneScript.py.

    The .pt checkpoint stores model_state_dict (not the full model), so we
    rebuild the same ResNet50 + 2-class head used during training, then
    load the state dict.

    Returns dict: {'net': model, 'bestThr': float, 'pos_idx': int}
    """
    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    # Rebuild architecture: ResNet50 with 2-class FC head
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    return {
        'net'     : model,
        'bestThr' : float(ckpt['bestThr']),
        'pos_idx' : int(ckpt['pos_idx']),   # 1 = bwhistle
    }


def _load_master(csv_path: Path) -> pd.DataFrame:
    """
    Load master table.  Accepts either:
      - A CSV exported from MATLAB (columns: t, fs, fname, nsamp)
      - The original .mat file directly (auto-detected by extension)

    MATLAB export (run once per site):
        t = tt_LM_20180109;
        t.t = string(t.t, 'yyyy-MM-dd HH:mm:ss.SSS');
        writetable(t, 'tt_LM_20180109.csv');
    """
    path = Path(csv_path)

    # ---- try the CSV path first, then the .mat sibling ------------------
    mat_path = path.with_suffix('.mat')
    if not path.exists() and mat_path.exists():
        print(f'  CSV not found — loading .mat directly: {mat_path}')
        return _load_master_mat(mat_path)
    elif not path.exists():
        raise FileNotFoundError(
            f'Master file not found: {path}\n'
            f'Export from MATLAB with:\n'
            f'    t = {path.stem};\n'
            f'    t.t = string(t.t, \'yyyy-MM-dd HH:mm:ss.SSS\');\n'
            f'    writetable(t, \'{path}\');'
        )

    df = pd.read_csv(str(path), parse_dates=['t'])
    if df['t'].dt.tz is None:
        df['t'] = df['t'].dt.tz_localize('UTC')
    else:
        df['t'] = df['t'].dt.tz_convert('UTC')
    return df


def _load_master_mat(mat_path: Path) -> pd.DataFrame:
    """
    Load a MATLAB master .mat file directly without needing a CSV export.
    Handles both v5 format (scipy) and v7.3/HDF5 format (h5py).
    The MATLAB variable name is inferred from the filename stem.
    """
    import scipy.io as sio

    var_name = mat_path.stem   # e.g. 'tt_LM_20180109'

    # ---- try scipy (v5 / v6 format) -------------------------------------
    try:
        raw = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        if var_name not in raw:
            raise KeyError(f"Variable '{var_name}' not found in {mat_path}. "
                           f"Available: {[k for k in raw if not k.startswith('_')]}")
        obj = raw[var_name]

        # MATLAB table saved by scipy comes out as a structured numpy array
        t_raw  = np.array(obj['t']).ravel()
        fs     = np.array(obj['fs']).ravel().astype(int)
        fname  = np.array(obj['fname']).ravel()
        nsamp  = np.array(obj['nsamp']).ravel().astype(int)

        # Convert MATLAB datenum (days since 0000-Jan-0) to pandas Timestamp
        # datenum 1 = 0001-Jan-01 = Python ordinal 1
        def _dn_to_ts(dn):
            from datetime import datetime, timedelta
            base = datetime.fromordinal(1)   # 0001-01-01
            return pd.Timestamp(base + timedelta(days=float(dn) - 1), tz='UTC')

        t_ts = pd.to_datetime([_dn_to_ts(d) for d in t_raw], utc=True)
        fname = [str(f).strip() for f in fname]
        return pd.DataFrame({'t': t_ts, 'fs': fs, 'fname': fname, 'nsamp': nsamp})

    except NotImplementedError:
        pass   # v7.3 HDF5 — fall through to h5py

    # ---- try h5py (v7.3 / HDF5 format) ---------------------------------
    try:
        import h5py
    except ImportError:
        raise ImportError(
            f'{mat_path} appears to be MATLAB v7.3 format.\n'
            'Install h5py:  pip install h5py\n'
            'Or export from MATLAB to CSV instead (see wrapper.py docstring).'
        )

    with h5py.File(str(mat_path), 'r') as f:
        grp = f[var_name]

        # MATLAB stores datetime as uint64 (100-ns ticks since 0000-Jan-00)
        # offset from Python epoch (1970-01-01):
        #   MATLAB tick 0 = 0000-Jan-00 = 621355968000000000 ns before Unix epoch
        _MATLAB_EPOCH_NS = np.uint64(621_355_968_000_000_000)
        t_ticks = grp['t'][()].ravel().astype(np.uint64)
        t_ns    = (t_ticks - _MATLAB_EPOCH_NS) * np.uint64(100)
        t_ts    = pd.to_datetime(t_ns.astype(np.int64), unit='ns', utc=True)

        fs    = grp['fs'][()].ravel().astype(int)
        nsamp = grp['nsamp'][()].ravel().astype(int)

        # HDF5 stores strings as object references or fixed-length bytes
        fname_ds = grp['fname']
        if fname_ds.dtype.kind in ('O', 'S', 'U'):
            fname = [str(x.decode() if isinstance(x, bytes) else x) for x in fname_ds[()].ravel()]
        else:
            # cell array of references
            fname = [''.join(chr(c) for c in f[ref][()].ravel()) for ref in fname_ds[()].ravel()]

    return pd.DataFrame({'t': t_ts, 'fs': fs, 'fname': fname, 'nsamp': nsamp})


def _load_wt(wt_csv: Path) -> pd.DataFrame:
    """Load water temperature CSV (columns: UTC, wtemp_C)."""
    wt = pd.read_csv(str(wt_csv), parse_dates=['UTC'])
    wt = wt.dropna(subset=['wtemp_C'])
    if wt['UTC'].dt.tz is None:
        wt['UTC'] = wt['UTC'].dt.tz_localize('UTC')
    else:
        wt['UTC'] = wt['UTC'].dt.tz_convert('UTC')
    return wt


_FRANGE_HALF = 75    # ±Hz around predf0 used as kernel search band (matches MATLAB wrapper)


def _pred_f0(wt: pd.DataFrame, seg_times: list, fs: int, nfft_pow: int):
    """
    Interpolate water temperature, compute predicted F0 per segment,
    snap to FFT bin grid, and return (predf0_arr, frange_arr).

    frange uses ±75 Hz (kernel search band).
    PREDF0_UNCERT (50 Hz) is the separate taper flat-top passed to the catalogger.
    """
    wt_t  = wt['UTC'].values.astype('datetime64[ns]').astype(np.float64)
    wt_c  = wt['wtemp_C'].values.astype(np.float64)
    st_ns = np.array([t.value for t in seg_times], dtype=np.float64)

    cwtemp = np.interp(st_ns, wt_t, wt_c, left=np.nan, right=np.nan)

    predf0 = -27.25 + 12.32 * cwtemp
    predf0 = np.clip(predf0, 130 + _FRANGE_HALF, 470 - _FRANGE_HALF)

    # Snap to FFT bin centre (matches MATLAB's fcenter grid)
    df      = fs / 2**nfft_pow
    bin_idx = np.round(predf0 / df).astype(int)
    fcenter = np.arange(int(fs / df) + 1) * df
    bin_idx = np.clip(bin_idx, 0, len(fcenter) - 1)
    predf0  = fcenter[bin_idx]

    frange = np.column_stack([
        np.maximum(predf0 - _FRANGE_HALF, 130),
        np.minimum(predf0 + _FRANGE_HALF, 470),
    ])
    return predf0, frange


def _read_segment(sf_file: sf.SoundFile, i1: int, i2: int, fsraw: int) -> np.ndarray:
    """Read samples [i1, i2) from an open SoundFile, return mono float64."""
    n_req = i2 - i1
    sf_file.seek(i1)
    y = sf_file.read(n_req, dtype='float64', always_2d=True)
    if y.shape[1] > 1:
        y = y[:, 0]    # first channel
    else:
        y = y[:, 0]
    if fsraw != FS_OUT:
        g = gcd(FS_OUT, fsraw)
        y = resample_poly(y, FS_OUT // g, fsraw // g)
    return y.ravel()


def _save_results(dir_out: Path, fbase: str,
                  B_all: pd.DataFrame, O_all: pd.DataFrame, D_all: pd.DataFrame,
                  summary: pd.DataFrame):
    """Write per-file CSV results."""
    dir_out.mkdir(parents=True, exist_ok=True)
    B_all.to_csv(str(dir_out / f'{fbase}_bwTable.csv'),    index=False)
    O_all.to_csv(str(dir_out / f'{fbase}_oTable.csv'),     index=False)
    D_all.to_csv(str(dir_out / f'{fbase}_DetTable.csv'),   index=False)
    summary.to_csv(str(dir_out / f'{fbase}_SummaryTab.csv'), index=False)


# ===========================================================================
# Main loop
# ===========================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    cfg    = SITE_CFG[SITE]
    dir_in = cfg['dir_in']
    dir_out = cfg['dir_out']
    dir_out.mkdir(parents=True, exist_ok=True)

    print(f'Loading classifier from {CLASSIFIER_PATH}')
    classifier = _load_classifier(CLASSIFIER_PATH, device)
    print(f'  bestThr={classifier["bestThr"]:.4f}  pos_idx={classifier["pos_idx"]}')

    print(f'Loading master from {cfg["master"]}')
    master = _load_master(cfg['master'])
    print(f'  {len(master)} files in master index')

    print(f'Loading water temperature from {cfg["wt_csv"]}')
    wt = _load_wt(cfg['wt_csv'])

    for m, row in enumerate(master.itertuples(), start=1):
        fname = str(row.fname)
        fsraw = int(row.fs)
        nsamp = int(row.nsamp)
        master_time = pd.Timestamp(row.t)
        if master_time.tzinfo is None:
            master_time = master_time.tz_localize('UTC')

        site_file = f'{SITE}_{m:03d}'
        full_in   = dir_in / fname
        if not full_in.exists():
            print(f'  WARNING: {full_in} not found, skipping')
            continue

        # Segment start times (every 2 min across file)
        fdurhrs  = (nsamp / fsraw) / 3600
        seg_times = pd.date_range(
            start=master_time,
            end  =master_time + pd.Timedelta(hours=fdurhrs) - pd.Timedelta(seconds=NSEC),
            freq =f'{int(NSEC)}s',
            tz='UTC',
        )
        nSeg = len(seg_times)
        print(f'Processing {site_file}  ({fname})  —  {nSeg} segments')
        if nSeg == 0:
            print(f'  Skipping {site_file} — too short for a full segment.')
            continue

        # Vectorised offsets and priors
        offset_samples = np.round(fsraw * np.array(
            [t.value / 1e9 - master_time.value / 1e9 for t in seg_times]
        )).astype(int)
        seg_samples = round(NSEC * fsraw)
        seg_ids     = [f'{i+1:03d}' for i in range(nSeg)]

        predf0, frange = _pred_f0(wt, list(seg_times), FS_OUT, NFFT_POW)

        Btabs  = []
        Otabs  = []
        Dtabs  = []
        Bcount = np.zeros(nSeg, dtype=int)
        Ocount = np.zeros(nSeg, dtype=int)

        try:
            with sf.SoundFile(str(full_in)) as sf_file:
                for i, (seg_id, st) in enumerate(
                        tqdm(list(zip(seg_ids, seg_times)),
                             desc=site_file, unit='seg', leave=False)):
                    try:
                        i1 = max(0, int(offset_samples[i]))
                        i2 = min(i1 + seg_samples, nsamp)
                        y  = _read_segment(sf_file, i1, i2, fsraw)
                        if len(y) == 0 or np.any(np.isnan(y)):
                            continue

                        bc, oc, btab, otab, dtab = gtf_deep_catalogger(
                            y, SITE, seg_id, fname, str(dir_out),
                            classifier, tuple(frange[i]), float(predf0[i]),
                            PREDF0_UNCERT, st, NSEC,
                            save_images=SAVE_IMAGES, keep_images=KEEP_IMAGES,
                            device=device,
                        )
                        Btabs.append(btab); Otabs.append(otab); Dtabs.append(dtab)
                        Bcount[i] = bc; Ocount[i] = oc

                    except Exception as e:
                        print(f'  ERROR seg {seg_id}: {e}')

        except Exception as e:
            print(f'ERROR opening {full_in}: {e}')
            continue

        # Combine and save
        B_all = pd.concat([t for t in Btabs if len(t) > 0], ignore_index=True) \
                if any(len(t) > 0 for t in Btabs) else pd.DataFrame()
        O_all = pd.concat([t for t in Otabs if len(t) > 0], ignore_index=True) \
                if any(len(t) > 0 for t in Otabs) else pd.DataFrame()
        D_all = pd.concat([t for t in Dtabs if len(t) > 0], ignore_index=True) \
                if any(len(t) > 0 for t in Dtabs) else pd.DataFrame()

        summary = pd.DataFrame({
            'n_bwhistle' : Bcount,
            'n_other'    : Ocount,
            'site'       : SITE,
            'fnumber'    : m,
            'file'       : fname,
            'segnumber'  : np.arange(1, nSeg + 1),
            'time'       : list(seg_times),
        })

        fbase = Path(fname).stem
        _save_results(dir_out, fbase, B_all, O_all, D_all, summary)
        print(f'  Done {site_file}  —  {int(Bcount.sum())} bwhistle  {int(Ocount.sum())} other')


if __name__ == '__main__':
    main()
