# database_generation_validation.py

'''
Generate a new database from DAIC-WOZ Database

 Symbol convention:
- N: batch size
- T: number of samples per batch (each sample includes _v_ key points)
- V: number of key point per person
- M: number of person
- C: information per point, like (x,y,z)-Coordinate or (x,y,z,p)-Coordinate + Confident probability

Architecture of generated database:
[DAIC_WOZ-generated_database]
 └── [train]
      ├── [original_data]
      │    ├── [no_gender_balance]
      │    │    ├── [facial_keypoints]
      │    │    │    ├── [only_coordinate], shape of each npy file: (1800, 68, 3)
      │    │    │    └── [coordinate+confidence], shape of each npy file: (1800, 68, 4)
      │    │    ├── [gaze_vector]
      │    │    │    ├── [only_coordinate], shape of each npy file: (1800, 4, 3)
      │    │    │    └── [coordinate+confidence], shape of each npy file: (1800, 4, 4)
      │    │    ├── [audio]
      │    │    │    ├── [spectrogram], shape of each npy file: (1025, 1800)
      │    │    │    └── [Mel-spectrogram], shape of each npy file: (80, 1800)
      │    │    ├── [text]
      │    │    │    └── [sentence_embeddings], shape of each npy file: (10, 768)
      │    │    ├── PHQ_Binary_GT.npy
      │    │    ├── PHQ_Score_GT.npy
      │    │    ├── PHQ_Subscore_GT.npy
      │    │    └── PHQ_Gender_GT.npy
      │    └── [gender_balance]
      │         ├── [facial_keypoints]
      │         │    ├── [only_coordinate]
      │         │    └── [coordinate+confidence]
      │         ├── [gaze_vector]
      │         │    ├── [only_coordinate]
      │         │    └── [coordinate+confidence]
      │         ├── [audio]
      │         │    ├── [spectrogram]
      │         │    └── [Mel-spectrogram]
      │         ├── [text]
      │         │    └── [sentence_embeddings]
      │         ├── PHQ_Binary_GT.npy
      │         ├── PHQ_Score_GT.npy
      │         ├── PHQ_Subscore_GT.npy
      │         └── PHQ_Gender_GT.npy
      └── [clipped_data]
           ├── [no_gender_balance]
           │    ├── [facial_keypoints]
           │    │    ├── [only_coordinate]
           │    │    └── [coordinate+confidence]
           │    ├── [gaze_vector]
           │    │    ├── [only_coordinate]
           │    │    └── [coordinate+confidence]
           │    ├── [audio]
           │    │    ├── [spectrogram]
           │    │    └── [Mel-spectrogram]
           │    └── [text]
           │         └── [sentence_embeddings]
           └── [gender_balance]
                ├── [facial_keypoints]
                │    ├── [only_coordinate]
                │    └── [coordinate+confidence]
                ├── [gaze_vector]
                │    ├── [only_coordinate]
                │    └── [coordinate+confidence]
                ├── [audio]
                │    ├── [spectrogram]
                │    └── [Mel-spectrogram]
                └── [text]
                     └── [sentence_embeddings]
 '''

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
    return (data - data.min())/(data.max() - data.min())


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
    gaze_final = np.zeros((T, V, C+1))

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
    fkps_final = np.zeros((T, V, C+1))

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
        if getattr(t,'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t,'value'):
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


# def sliding_window(spectro, mel_spectro, text_feature,
#                    window_size, overlap_size, output_root, ID):
#     frame_size = window_size
#     hop_size = window_size - overlap_size
#     num_frame = get_num_frame(spectro, frame_size, hop_size)
#     text_frame_size = 10
#     text_hop_size = get_text_hop_size(text_feature, text_frame_size, num_frame)
#
#     for i in range(num_frame):
#         frame_sample_spec = audio_padding(spectro[:, i * hop_size:i * hop_size + frame_size], frame_size)
#         frame_sample_mspec = audio_padding(mel_spectro[:, i * hop_size:i * hop_size + frame_size], frame_size)
#         frame_sample_text = text_padding(text_feature[i * text_hop_size:i * text_hop_size + text_frame_size],
#                                          text_frame_size)
#         # 创建必要的文件夹
#         os.makedirs(os.path.join(output_root, 'audio', 'spectrogram'), exist_ok=True)
#         os.makedirs(os.path.join(output_root, 'audio', 'mel-spectrogram'), exist_ok=True)
#         os.makedirs(os.path.join(output_root, 'text', 'sentence_embeddings'), exist_ok=True)
#
#         np.save(os.path.join(output_root, 'audio', 'spectrogram', f'{ID}-{i:02}_audio.npy'), frame_sample_spec)
#         np.save(os.path.join(output_root, 'audio', 'mel-spectrogram', f'{ID}-{i:02}_audio.npy'), frame_sample_mspec)
#         np.save(os.path.join(output_root, 'text', 'sentence_embeddings', f'{ID}-{i:02}_text.npy'), frame_sample_text)
#
#     return num_frame

def sliding_window(spectro, mel_spectro, text_feature, window_size, overlap_size, output_root, ID):
    # 计算每块语音时长对应的帧数，window_size 和 overlap_size 单位为秒
    frame_size = window_size * 30
    hop_size = (window_size - overlap_size) * 30

    # # 音频的总帧数，1 分钟音频对应的帧数
    # num_frame = mel_spectro.shape[1] // hop_size
    num_frame = get_num_frame(spectro, frame_size, hop_size)

    # 文本特征的滑窗处理参数
    text_frame_size = 10  # 假设文本特征每 10 帧一个切片
    text_hop_size = len(text_feature) // num_frame

    for i in range(num_frame):
        # frame_sample_spec = audio_padding(spectro[:, i * hop_size:i * hop_size + frame_size], frame_size)
        # 音频梅尔谱切片
        frame_sample_mspec = audio_padding(mel_spectro[:, i * hop_size:i * hop_size + frame_size], frame_size)
        # 文本特征切片
        frame_sample_text = text_padding(text_feature[i * text_hop_size:i * text_hop_size + text_frame_size],
                                         text_frame_size)

        # 保存音频梅尔谱和文本特征
        # np.save(os.path.join(output_root, 'audio', 'spectrogram', f'{ID}-{i:02}_audio.npy'), frame_sample_spec)
        np.save(os.path.join(output_root, 'audio', 'mel-spectrogram', f'{ID}-{i:02}_audio.npy'), frame_sample_mspec)
        np.save(os.path.join(output_root, 'text', f'{ID}-{i:02}_text.npy'), frame_sample_text)


    return num_frame


if __name__ == '__main__':

    # output root
    root = 'D:\\博士资料\\all class_3semester\\AI_project_class\\DepressionEstimation-main\\output_validation'
    root_dir = os.path.join(root, 'DAIC_WOZ-generated_database', 'validation')
    create_folders(root_dir)
    np.random.seed(1)

    # read  gt file
    # gt_path = 'D:\\博士资料\\all class_3semester\\AI_project_class\\DAIC-WoZ-Summarization-main\\data\\datasets\\labels\\dev_split.csv'
    gt_path = 'D:\博士资料\\all class_3semester\AI_project_class\DepressionEstimation-main\daic_woz_preprocessing\Excel for splitting data\\corrected_dev_split_Depression_AVEC2017.csv'
    gt_df = pd.read_csv(gt_path)

    # initialization
    sent2vec = SentenceTransformer('all-mpnet-base-v2')
    window_size = 60   # 60s
    overlap_size = 10  # 10s
    GT = {'original_data':
          {'ID_gt':[], 'gender_gt': [], 'phq_binary_gt': [], 'phq_score_gt':[], 'phq_subscores_gt':[]}, 
          'clipped_data':
          {'ID_gt':[], 'gender_gt': [], 'phq_binary_gt': [], 'phq_score_gt':[], 'phq_subscores_gt':[]}}

    for i in range(len(gt_df)):
        # extract training gt details
        patient_ID = gt_df['Participant_ID'][i]
        phq_binary_gt = gt_df['PHQ8_Binary'][i]
        phq_score_gt = gt_df['PHQ8_Score'][i]
        gender_gt = gt_df['Gender'][i]
        phq_subscores_gt = gt_df.iloc[i, 4:].to_numpy().tolist()
        print(f'Processing Participant {patient_ID}, Gender: {gender_gt} ...')
        print(f'- PHQ Binary: {phq_binary_gt}, PHQ Score: {phq_score_gt}, Subscore: {phq_subscores_gt}')

        # get all files path of participant
        text_path = f'D:\\博士资料\\all class_3semester\\AI_project_class\\DAIC-WoZ-Summarization-main\\data\\datasets/{patient_ID}_P/{patient_ID}_TRANSCRIPT.csv'
        audio_path = f'D:\\博士资料\\all class_3semester\\AI_project_class\\DAIC-WoZ-Summarization-main\\data\\datasets/{patient_ID}_P/{patient_ID}_AUDIO.wav'

        # read transcipt file
        text_df = pd.read_csv(text_path)
        text_df = text_df.iloc[:-2, :]
        first_start_time = text_df['Start_Time'][0]
        last_stop_time = text_df['End_Time'][len(text_df)-1]


        # read audio file
        audio, audio_sr = load_audio(audio_path)

        # extract text feature
        text_feature = sentence_embedding(text_df, model=sent2vec)


        ########################################
        # feature extraction for original_data #
        ########################################

        print(f'Extracting feature of Participant {patient_ID} for original_data...')

        # audio
        filtered_audio = audio_clipping(audio, audio_sr, text_df, zero_padding=True)
        # spectrogram, mel spectrogram
        spectro = normalize(convert_spectrogram(filtered_audio, frame_size=2048, hop_size=533))
        mel_spectro = normalize(convert_mel_spectrogram(filtered_audio, audio_sr, 
                                                        frame_size=2048, hop_size=533, num_mel_bands=80))


        #################################################
        # start creating data in 'original_data' folder #
        #################################################

        output_root = os.path.join(root_dir, 'original_data')
        num_frame = sliding_window(spectro, mel_spectro, text_feature,
                                   window_size, overlap_size, output_root, patient_ID)

        # replicate GT
        for _ in range(num_frame):
            GT['original_data']['ID_gt'].append(patient_ID)
            GT['original_data']['gender_gt'].append(gender_gt)
            GT['original_data']['phq_binary_gt'].append(phq_binary_gt)
            GT['original_data']['phq_score_gt'].append(phq_score_gt)
            GT['original_data']['phq_subscores_gt'].append(phq_subscores_gt)


        #######################################
        # feature extraction for clipped_data #
        #######################################

        print(f'Extracting feature of Participant {patient_ID} for clipped_data...')

        # audio
        clipped_audio = audio_clipping(audio, audio_sr, text_df, zero_padding=False)
        # spectrogram, mel spectrogram
        spectro = normalize(convert_spectrogram(clipped_audio, frame_size=2048, hop_size=533))
        mel_spectro = normalize(convert_mel_spectrogram(clipped_audio, audio_sr, 
                                                        frame_size=2048, hop_size=533, num_mel_bands=80))


        ################################################
        # start creating data in 'clipped_data' folder #
        ################################################

        output_root = os.path.join(root_dir, 'clipped_data')
        num_frame = sliding_window(spectro, mel_spectro, text_feature,
                                   window_size, overlap_size, output_root, patient_ID)

        # replicate GT
        for _ in range(num_frame):
            GT['clipped_data']['ID_gt'].append(patient_ID)
            GT['clipped_data']['gender_gt'].append(gender_gt)
            GT['clipped_data']['phq_binary_gt'].append(phq_binary_gt)
            GT['clipped_data']['phq_score_gt'].append(phq_score_gt)
            GT['clipped_data']['phq_subscores_gt'].append(phq_subscores_gt)


    # store new GT
    for k1, v1 in GT.items():
        for k2, v2 in v1.items():
            # print(os.path.join(root_dir, k1, f'{k2}.npy'))
            np.save(os.path.join(root_dir, k1, f'{k2}.npy'), v2)

    print('All done!')
