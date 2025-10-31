import numpy as np
from scipy.interpolate import interp1d
from ligotools.utils import whiten, write_wavfile, reqshift

def test_whiten_runs():
    fs = 4096
    t = np.linspace(0, 1, fs)
    x = np.sin(2 * np.pi * 100 * t)
    freqs = np.linspace(0, fs / 2, len(x)//2 + 1)
    psd = interp1d(freqs, np.ones_like(freqs), bounds_error=False, fill_value=1)
    w = whiten(x, psd, 1/fs)
    assert len(w) == len(x)

def test_reqshift_and_write(tmp_path):
    fs = 4096
    t = np.linspace(0, 1, fs)
    x = np.sin(2 * np.pi * 100 * t)
    y = reqshift(x, fshift=200, sample_rate=fs)
    out = tmp_path / "test.wav"
    write_wavfile(out, fs, y)
    assert out.exists()
