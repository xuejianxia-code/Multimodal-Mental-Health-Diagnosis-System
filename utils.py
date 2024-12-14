import os
import numpy as np
import pandas as pd
import wave
import librosa
from sentence_transformers import SentenceTransformer

def create_folders(root_dir):
    folders = ['original_data', 'clipped_data']
    subfolders = {'facial_keypoints': ['only_coordinate', 'coordinate+confidence'],
                  'gaze_vectors': ['only_coordinate', 'coordinate+confidence'],
                  'audio': ['spectrogram', 'mel-spectrogram'],
                  'text': ['sentence_embeddings']}

    os.makedirs(root_dir, exist_ok=True)
    for i in folders:
        for k, v in subfolders.items():
            for m in v:
                # print(os.path.join(root_dir, i, k, m))
                os.makedirs(os.path.join(root_dir, i, k, m), exist_ok=True)


def min_max_scaler(data):
    '''recale the data, which is a 2D matrix, to 0-1'''
    return (data - data.min()) / (data.max() - data.min())


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


def pre_check(data_df):
    data_df = data_df.apply(pd.to_numeric, errors='coerce')
    data_np = data_df.to_numpy()
    data_min = data_np[np.where(~(np.isnan(data_np[:, 4:])))].min()
    data_df.where(~(np.isnan(data_df)), data_min, inplace=True)
    return data_df


def load_gaze(gaze_path):
    gaze_df = pre_check(pd.read_csv(gaze_path, low_memory=False))
    # process into format TxVxC
    gaze_conf = gaze_df[' confidence'].to_numpy()
    gaze_coor = gaze_df.iloc[:, 4:].to_numpy().reshape(len(gaze_df), 4, 3)  # 4 gaze vectors, 3 axes
    T, V, C = gaze_coor.shape

    # initialize the final gaze_3D which contains coordinate and confidence score
    gaze_final = np.zeros((T, V, C + 1))

    gaze_final[:, :, :3] = gaze_coor
    for i in range(V):
        gaze_final[:, i, 3] = gaze_conf

    return gaze_coor, gaze_final


def load_keypoints(keypoints_path):
    fkps_df = pre_check(pd.read_csv(keypoints_path, low_memory=False))
    # process into format TxVxC
    fkps_conf = fkps_df[' confidence'].to_numpy()
    x_coor = min_max_scaler(fkps_df[fkps_df.columns[4: 72]].to_numpy())
    y_coor = min_max_scaler(fkps_df[fkps_df.columns[72: 140]].to_numpy())
    z_coor = min_max_scaler(fkps_df[fkps_df.columns[140: 208]].to_numpy())
    fkps_coor = np.stack([x_coor, y_coor, z_coor], axis=-1)
    T, V, C = fkps_coor.shape

    # initialize the final facial key points which contains coordinate and confidence score
    fkps_final = np.zeros((T, V, C + 1))

    fkps_final[:, :, :3] = fkps_coor
    for i in range(V):
        fkps_final[:, i, 3] = fkps_conf

    return fkps_coor, fkps_final


def load_audio(audio_path):
    wavefile = wave.open(audio_path)
    audio_sr = wavefile.getframerate()
    n_samples = wavefile.getnframes()
    signal = np.frombuffer(wavefile.readframes(n_samples), dtype=np.short)

    return signal.astype(float), audio_sr


def visual_clipping(visual_data, visual_sr, text_df):
    counter = 0
    for t in text_df.itertuples():
        if getattr(t, 'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t, 'value'):
                continue
            else:
                start = getattr(t, 'start_time')
                stop = getattr(t, 'stop_time')
                start_sample = int(start * visual_sr)
                stop_sample = int(stop * visual_sr)
                if counter == 0:
                    edited_vdata = visual_data[start_sample:stop_sample]
                else:
                    edited_vdata = np.vstack((edited_vdata, visual_data[start_sample:stop_sample]))

                counter += 1

    return edited_vdata


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
        last_stop = text_df['End_Time'][len(text_df) - 1]
        edited_audio = edited_audio[int(first_start * audio_sr):int(last_stop * audio_sr)]

    else:
        edited_audio = []
        for t in text_df.itertuples():
            start = getattr(t, 'Start_Time')
            stop = getattr(t, 'End_Time')
            start_sample = int(start * audio_sr)
            stop_sample = int(stop * audio_sr)
            edited_audio = np.hstack((edited_audio, audio[start_sample:stop_sample]))

    return edited_audio


def convert_spectrogram(audio, frame_size=2048, hop_size=533):
    # extracting with Short-Time Fourier Transform
    S_scale = librosa.stft(audio, n_fft=frame_size, hop_length=hop_size)
    spectrogram = np.abs(S_scale) ** 2
    # convert amplitude to DBs
    log_spectrogram = librosa.power_to_db(spectrogram)

    return log_spectrogram  # in dB


def convert_mel_spectrogram(audio, audio_sr, frame_size=2048, hop_size=533, num_mel_bands=80):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=audio_sr,
                                                     n_fft=frame_size,
                                                     hop_length=hop_size,
                                                     n_mels=num_mel_bands)
    # convert amplitude to DBs
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    return log_mel_spectrogram  # in dB


def sentence_embedding(text_df, model):
    sentences = []
    for t in text_df.itertuples():
        sentences.append(getattr(t, 'Text'))

    return model.encode(sentences)


def get_num_frame(data, frame_size, hop_size):
    T = data.shape[1]
    if (T - frame_size) % hop_size == 0:
        num_frame = (T - frame_size) // hop_size + 1
    else:
        num_frame = (T - frame_size) // hop_size + 2
    return num_frame


def get_text_hop_size(text, frame_size, num_frame):
    T = text.shape[0]
    return (T - frame_size) // (num_frame - 1)


def visual_padding(data, pad_size):
    if data.shape[0] != pad_size:
        size = tuple()
        size = size + (pad_size,) + data.shape[1:]
        padded_data = np.zeros(size)
        padded_data[:data.shape[0]] = data
    else:
        padded_data = data

    return padded_data


def audio_padding(data, pad_size):
    if data.shape[1] != pad_size:
        size = tuple((data.shape[0], pad_size))
        padded_data = np.zeros(size)
        padded_data[:, :data.shape[1]] = data
    else:
        padded_data = data

    return padded_data


def text_padding(data, pad_size):
    if data.shape[0] != pad_size:
        size = tuple((pad_size, data.shape[1]))
        padded_data = np.zeros(size)
        padded_data[:data.shape[0]] = data
    else:
        padded_data = data

    return padded_data


def random_shift_fkps(fkps_coor, fkps_coor_conf):
    shifted_fc = np.copy(fkps_coor)
    shifted_fcc = np.copy(fkps_coor_conf)

    for i in range(3):
        factor = np.random.uniform(-0.05, 0.05)
        shifted_fc[:, :, i] = shifted_fc[:, :, i] + factor
        shifted_fcc[:, :, i] = shifted_fcc[:, :, i] + factor

    return shifted_fc, shifted_fcc

def sliding_window_text(text_feature, window_size, overlap_size):
    # 假设文本特征每 10 帧一个切片
    text_frame_size = window_size
    text_hop_size = window_size - overlap_size
    if (len(text_feature) - window_size) % text_hop_size == 0:
        num_frame = (len(text_feature) - window_size) // text_hop_size + 1
    else:
        num_frame = (len(text_feature) - window_size) // text_hop_size + 2

    frame_sample_text_all = []
    for i in range(num_frame):
        # 文本特征切片
        frame_sample_text = text_padding(text_feature[i * text_hop_size:i * text_hop_size + text_frame_size],
                                         text_frame_size)
        frame_sample_text_all.append(frame_sample_text)

    return num_frame, frame_sample_text_all

def sliding_window_audio(mel_spectro, window_size, overlap_size):
    # 计算每块语音时长对应的帧数，window_size 和 overlap_size 单位为秒
    frame_size = window_size * 30
    hop_size = (window_size - overlap_size) * 30

    # # 音频的总帧数，1 分钟音频对应的帧数
    # num_frame = mel_spectro.shape[1] // hop_size
    num_frame = get_num_frame(mel_spectro, frame_size, hop_size)

    # # 文本特征的滑窗处理参数
    # text_frame_size = 10  # 假设文本特征每 10 帧一个切片
    # text_hop_size = len(text_feature) // num_frame
    frame_sample_mspec_all=[]
    # frame_sample_text_all=[]
    for i in range(num_frame):
        # frame_sample_spec = audio_padding(spectro[:, i * hop_size:i * hop_size + frame_size], frame_size)
        # 音频梅尔谱切片
        frame_sample_mspec = audio_padding(mel_spectro[:, i * hop_size:i * hop_size + frame_size], frame_size)
        frame_sample_mspec_all.append(frame_sample_mspec)


    return num_frame, frame_sample_mspec_all

def sliding_window(mel_spectro, text_feature, window_size, overlap_size):
    # 计算每块语音时长对应的帧数，window_size 和 overlap_size 单位为秒
    frame_size = window_size * 30
    hop_size = (window_size - overlap_size) * 30

    # # 音频的总帧数，1 分钟音频对应的帧数
    # num_frame = mel_spectro.shape[1] // hop_size
    num_frame = get_num_frame(mel_spectro, frame_size, hop_size)

    # 文本特征的滑窗处理参数
    text_frame_size = 10  # 假设文本特征每 10 帧一个切片
    text_hop_size = len(text_feature) // num_frame
    frame_sample_mspec_all=[]
    frame_sample_text_all=[]
    for i in range(num_frame):
        # frame_sample_spec = audio_padding(spectro[:, i * hop_size:i * hop_size + frame_size], frame_size)
        # 音频梅尔谱切片
        frame_sample_mspec = audio_padding(mel_spectro[:, i * hop_size:i * hop_size + frame_size], frame_size)
        # 文本特征切片
        frame_sample_text = text_padding(text_feature[i * text_hop_size:i * text_hop_size + text_frame_size],
                                         text_frame_size)
        frame_sample_mspec_all.append(frame_sample_mspec)
        frame_sample_text_all.append(frame_sample_text)
        # 保存音频梅尔谱和文本特征
        # np.save(os.path.join(output_root, 'audio', 'spectrogram', f'{ID}-{i:02}_audio.npy'), frame_sample_spec)
        # np.save(os.path.join(output_root, 'audio', 'mel-spectrogram', f'{ID}-{i:02}_audio.npy'), frame_sample_mspec)
        # np.save(os.path.join(output_root, 'text', f'{ID}-{i:02}_text.npy'), frame_sample_text)


    return num_frame, frame_sample_mspec_all, frame_sample_text_all