import os
import numpy as np
import pandas as pd
import wave
import librosa
from sentence_transformers import SentenceTransformer

def min_max_scaler(data):
    '''Scale the data, which is a 2D matrix, to the range [0, 1].'''
    return (data - data.min()) / (data.max() - data.min())

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def pre_check(data_df):
    '''Pre-process DataFrame by coercing non-numeric entries to NaN, and filling NaN values with the minimum.'''
    data_df = data_df.apply(pd.to_numeric, errors='coerce')
    data_np = data_df.to_numpy()
    data_min = data_np[np.where(~(np.isnan(data_np[:, 4:])))].min()
    data_df.where(~(np.isnan(data_df)), data_min, inplace=True)
    return data_df

def load_audio(audio_path):
    '''Load an audio file and return the signal and sample rate.'''
    wavefile = wave.open(audio_path)
    audio_sr = wavefile.getframerate()
    n_samples = wavefile.getnframes()
    signal = np.frombuffer(wavefile.readframes(n_samples), dtype=np.short)
    return signal.astype(float), audio_sr

def audio_clipping(audio, audio_sr, text_df, zero_padding=False):
    if zero_padding:
        edited_audio = np.zeros(audio.shape[0])
        for t in text_df.itertuples():
            start = getattr(t, 'Start_Time')
            stop = getattr(t, 'End_Time')
            start_sample = int(start * audio_sr)
            stop_sample = int(stop * audio_sr)
            edited_audio[start_sample:stop_sample] = audio[start_sample:stop_sample]

        # cut head and tail of interview
        first_start = text_df['Start_Time'][0]
        last_stop = text_df['End_Time'][len(text_df)-1]
        edited_audio = edited_audio[int(first_start*audio_sr):int(last_stop*audio_sr)]
    else:
        edited_audio = []
        for t in text_df.itertuples():
            start = getattr(t, 'Start_Time')
            stop = getattr(t, 'End_Time')
            start_sample = int(start * audio_sr)
            stop_sample = int(stop * audio_sr)
            edited_audio = np.hstack((edited_audio, audio[start_sample:stop_sample]))

    return edited_audio

def convert_mel_spectrogram(audio, audio_sr, frame_size=2048, hop_size=533, num_mel_bands=80):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=audio_sr,
                                                     n_fft=frame_size,
                                                     hop_length=hop_size,
                                                     n_mels=num_mel_bands)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram  # in dB

def sentence_embedding(text_df, model):
    sentences = [getattr(t, 'Text') for t in text_df.itertuples()]
    return model.encode(sentences)

if __name__ == '__main__':
    # Load text and audio data for a single participant
    text_path = 'path_to_text_file.csv'  # Update this with your path
    audio_path = 'path_to_audio_file.wav'  # Update this with your path

    # Load transcript and audio
    text_df = pd.read_csv(text_path)
    text_df = text_df.iloc[:-2, :]  # Drop last two rows if they are not relevant
    audio, audio_sr = load_audio(audio_path)

    # Process text features
    sent2vec = SentenceTransformer('all-mpnet-base-v2')
    text_feature = sentence_embedding(text_df, model=sent2vec)

    # Clip audio and extract features
    clipped_audio = audio_clipping(audio, audio_sr, text_df, zero_padding=False)
    mel_spectrogram = normalize(convert_mel_spectrogram(clipped_audio, audio_sr, frame_size=2048, hop_size=533, num_mel_bands=80))

    print("Processed audio and text features extracted successfully!")
