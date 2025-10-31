import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.interpolate import interp1d
from matplotlib import mlab
# Function 1: Whiten
def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strain)
    norm = 1.0 / np.sqrt(1.0 / (dt * 2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht
# Function 2: Write WAV
def write_wavfile(filename, fs, data):
    d = np.int16(data / np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename, int(fs), d)
# Function 3: Frequency Shift
def reqshift(data, fshift=100, sample_rate=4096):
    x = np.fft.rfft(data)
    T = len(data) / float(sample_rate)
    df = 1.0 / T
    nbins = int(fshift / df)
    y = np.roll(x.real, nbins) + 1j * np.roll(x.imag, nbins)
    y[0:nbins] = 0.0
    z = np.fft.irfft(y)
    return z

    
# Function 4: Plot PSD (from the notebook cell)
def plot_matched_filter_figs(
    time, timemax, SNR, pcolor, det, eventname, plottype,
    tevent, strain_whitenbp, template_match,
    datafreq, template_fft, freqs, data_psd, fs, d_eff
):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time - timemax, SNR, pcolor, label=det + ' SNR(t)')
    plt.grid(True)
    plt.ylabel('SNR')
    plt.xlabel(f'Time since {timemax:.4f}')
    plt.legend(loc='upper left')
    plt.title(det + ' matched filter SNR around event')

    # zoom in
    plt.subplot(2, 1, 2)
    plt.plot(time - timemax, SNR, pcolor, label=det + ' SNR(t)')
    plt.grid(True)
    plt.ylabel('SNR')
    plt.xlim([-0.15, 0.05])
    plt.xlabel(f'Time since {timemax:.4f}')
    plt.legend(loc='upper left')
    plt.savefig('figures/' + eventname + "_" + det + "_SNR." + plottype)
    plt.close()

    # -- Whitened data + template, and residuals
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time - tevent, strain_whitenbp, pcolor, label=det + ' whitened h(t)')
    plt.plot(time - tevent, template_match, 'k', label='Template(t)')
    plt.ylim([-10, 10])
    plt.xlim([-0.15, 0.05])
    plt.grid(True)
    plt.xlabel(f'Time since {timemax:.4f}')
    plt.ylabel('whitened strain (units of noise stdev)')
    plt.legend(loc='upper left')
    plt.title(det + ' whitened data around event')

    plt.subplot(2, 1, 2)
    plt.plot(time - tevent, strain_whitenbp - template_match, pcolor, label=det + ' resid')
    plt.ylim([-10, 10])
    plt.xlim([-0.15, 0.05])
    plt.grid(True)
    plt.xlabel(f'Time since {timemax:.4f}')
    plt.ylabel('whitened strain (units of noise stdev)')
    plt.legend(loc='upper left')
    plt.title(det + ' Residual whitened data after subtracting template around event')
    plt.savefig('figures/' + eventname + "_" + det + "_matchtime." + plottype)
    plt.close()

    # -- Display PSD and template (multiply by sqrt(f) to plot template fft on top of ASD)
    plt.figure(figsize=(10, 6))
    template_f = np.absolute(template_fft) * np.sqrt(np.abs(datafreq)) / d_eff
    plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
    plt.loglog(freqs, np.sqrt(data_psd), pcolor, label=det + ' ASD')
    plt.xlim(20, fs / 2)
    plt.ylim(1e-24, 1e-20)
    plt.grid(True)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
    plt.legend(loc='upper left')
    plt.title(det + ' ASD and template around event')
    plt.savefig('figures/' + eventname + "_" + det + "_matchfreq." + plottype)
    plt.close()
