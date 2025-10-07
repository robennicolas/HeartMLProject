import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
import pywt
import torch
import torch.nn as nn

#Denoising class
class WaveletDenoiser():
    def __init__(self, wavelet='db6', level=4, threshold_factor=0.5, mode='soft'):
        self.wavelet = wavelet
        self.level = level
        self.threshold_factor = threshold_factor
        self.mode = mode

    def _denoise(self, signal):
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        coeffs[1:] = [pywt.threshold(c,
                                     value=np.std(c) * self.threshold_factor,
                                     mode=self.mode) for c in coeffs[1:]]
        return pywt.waverec(coeffs, self.wavelet)
    
    def transform(self, X):
        # Applique le débruitage lead par lead
        return np.array([self._denoise(ecg) for ecg in X])
    
    
import matplotlib.pyplot as plt

def plot_ECG(ecg, fs=100):

    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
             'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    t = np.arange(ecg.shape[0]) / fs  # vecteur temps en secondes
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
    axes = axes.ravel()
    
    for i in range(12):
        axes[i].plot(t, ecg[:, i], color='black', linewidth=0.8)
        axes[i].set_title(leads[i])
        axes[i].set_ylabel('µV')
        axes[i].grid(True, linestyle='--', linewidth=0.5)
    
    axes[-2].set_xlabel('Temps (s)')
    plt.tight_layout()
    plt.show()




    



# Fast reading
ECGs = np.load('X.npy')
MetaData = pd.read_pickle('Y.pkl')


Y= MetaData['diagnostic_superclass']
X = MetaData[['age', 'sex', 'height', 'weight', 'heart_axis', 'baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems', 'extra_beats', 'pacemaker']]

print(MetaData)


