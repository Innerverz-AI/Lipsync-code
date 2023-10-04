import librosa
import numpy as np
import python_speech_features
from scipy.io import wavfile


# audio pp
def _amp_to_db(x):
    min_level = np.exp(-100 / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    return np.clip((2 * 4) * ((S + 100) / 100) - 4, -4, 4)


def _linear_to_mel(spectogram, sr=16000):
    _mel_basis = _build_mel_basis(sr)
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis(sr=16000, n_fft=800, num_mels=80, fmin=55, fmax=7600):
    assert 7600 <= sr // 2
    return librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )


def get_mel(audio_path, n_fft=800, sr=16000):
    audio, _ = librosa.load(audio_path, sr=sr)
    D = librosa.stft(y=audio, n_fft=n_fft, hop_length=200, win_length=800)
    S = _amp_to_db(_linear_to_mel(np.abs(D), sr)) - 20
    mel = _normalize(S)
    return mel


def get_mfcc(audio_path):
    sample_rate, audio = wavfile.read(audio_path)
    
    mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
    mfcc = np.stack([np.array(i) for i in mfcc])
    return mfcc

def get_audio_features(audio_path, target=['mel', 'mfcc']):
    audio_dict = {
        'mel':None,
        'mfcc':None
    }
    
    if 'mel' in target:
        mel_feature = get_mel(audio_path)
        audio_dict['mel'] = mel_feature
        print('mel')
    
    if 'mfcc' in target:
        mfcc_feature = get_mfcc(audio_path)
        audio_dict['mfcc'] = mfcc_feature
        print('mfcc')
        
    return audio_dict