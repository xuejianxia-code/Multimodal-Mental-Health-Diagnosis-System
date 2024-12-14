import pandas as pd
import librosa
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from daic_woz_preprocessing.database_generation_v1.database_generation_test_input_app import sentence_embedding,load_audio,audio_clipping,normalize,convert_mel_spectrogram
import matplotlib.pyplot as plt
import torch
import streamlit as st
import yaml
from models.Audio_ConvLSTM.models.convlstm import ConvLSTM_Audio
from models.Audio_ConvLSTM.models.evaluator import Evaluator
import torch.nn as nn
from datetime import datetime
import time

from models.Text_ConvLSTM.models.convlstm import ConvLSTM_Text
# from models.Text_ConvLSTM.models.evaluator import Evaluator
import models.Text_ConvLSTM.models.evaluator
import models.AT_ConvLSTM_Attention.models.convlstm
import models.AT_ConvLSTM_Attention.models.evaluator
import models.AT_ConvLSTM_Attention.models.fusion1
from utils import sliding_window
from utils import sliding_window_audio,sliding_window_text

import numpy as np
import os
import random
from streamlit_mic_recorder import mic_recorder
import io
import scipy.io.wavfile as wav
from pydub import AudioSegment
import tempfile

from speech_recognition import Recognizer,AudioData





# streamlit run app.py

nltk.download('stopwords')


SAVE_PATH = "D:\\Development of a Multimodal Mental Health Diagnosis Model\\user_inputs.csv"

if not os.path.exists(SAVE_PATH):
    df = pd.DataFrame(columns=["Text", "Audio"])
    df.to_csv(SAVE_PATH, index=False)

sidebar_bg_img = '''
<style>
    [data-testid="stSidebar"] {
        background-image: url("https://www.centurium.com/wp-content/uploads/2022/04/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20220406143916-1024x464.jpg");
        background-size: cover;
        background-position: top left;
        background-repeat: no-repeat;
    }
</style>
'''


st.markdown(sidebar_bg_img, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .custom-selectbox-label {
        font-size: 40px; 
        font-weight: bold; 
        color: #333; 
        margin-bottom: 1px; 
        width: 100%; 
        text-align: center; 
        # display: block; 
    }
    </style>
    <div class="custom-selectbox-label">Welcome to the Multimodal Mental Health Diagnosis System:</div>
    """,
    unsafe_allow_html=True
)

input_option0 = st.selectbox(
    "",
    ("Offline Diagnosis", "Real-time Diagnosis")
)

def concatenate_audio_files(audio_files):
    combined_audio = AudioSegment.empty()

    for file in audio_files:
        audio = AudioSegment.from_wav(file)
        combined_audio += audio
    return combined_audio
def save_audio(filename, audio_bytes):
    with open(filename, 'wb') as f:
        f.write(audio_bytes)


def speech_to_text(start_prompt="Start recording", stop_prompt="Stop recording", just_once=False,
                   use_container_width=False, language='en', callback=None, args=(), kwargs={}, key=None):
    if not '_last_speech_to_text_transcript_id' in st.session_state:
        st.session_state._last_speech_to_text_transcript_id = 0
    if key and not key + '_output' in st.session_state:
        st.session_state[key + '_output'] = None
    audio = mic_recorder(start_prompt=start_prompt, stop_prompt=stop_prompt, just_once=just_once,
                         use_container_width=use_container_width, format="wav", key=key)
    new_output = False
    if audio is None:
        output = None
    else:
        id = audio['id']
        new_output = (id > st.session_state._last_speech_to_text_transcript_id)
        if new_output or not just_once:
            st.session_state._last_speech_to_text_transcript_id = id
            r = Recognizer()
            audio_data = AudioData(audio['bytes'], audio['sample_rate'], audio['sample_width'])
            # 保存录音数据
            temp_filename = f"D:\\Development of a Multimodal Mental Health Diagnosis Model\\speech_to_text(the speech)\\audio_{id}.wav"  # 使用 id 或其他唯一标识生成文件名
            save_audio(temp_filename, audio['bytes'])  # 保存音频数据为文件
            try:
                output = r.recognize_google(audio_data, language=language)
            except:
                output = None
    if key:
        st.session_state[key + '_output'] = output
    if new_output and callback:
        callback(*args, **kwargs)
    return output

# 读取本地 CSV 文件并准备为下载
def load_csv_for_download(file_path):
    # 读取文件
    with open(file_path, "rb") as file:
        return file.read()

def plot_mel_spectrogram(mel_spectrogram):
    # 画出 Mel 频谱图
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, sr=16000, hop_length=533, y_axis='mel', x_axis='time')

    # 设置坐标轴标签和标题
    plt.title('Mel Spectrogram')
    # plt.xlabel('Time (frames)')
    # plt.ylabel('Mel Frequency bins')
    # 添加 colorbar
    plt.colorbar(format='%+2.0f dB')
    # 显示图像
    st.pyplot(plt)

def plot_text_embedings(vectors):
    plt.figure(figsize=(10, 6))
    plt.imshow(vectors, aspect='auto', cmap='viridis')  # 绘制热力图
    plt.colorbar()  # 显示颜色条
    st.pyplot(plt)

def find_last_ckpts(path, key, date=None):
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Arguments:
        path: str, path to the checkpoint
        key: str, model type
        date: str, a specific date in string format 'YYYY-MM-DD'
    Returns:
        The path of the last checkpoint file
    """
    ckpts = list(sorted(os.listdir(path)))

    if date is not None:
        # match the date format
        date_format = "%Y-%m-%d"
        try:
            datetime.strptime(date, date_format)
            # print("This is the correct date string format.")
            matched = True
        except ValueError:
            # print("This is the incorrect date string format. It should be YYYY-MM-DD")
            matched = False
        assert matched, "The given date is the incorrect date string format. It should be YYYY-MM-DD"

        key = '{}_{}'.format(key, date)
    else:
        key = str(key)

    # filter the files
    ckpts = list(filter(lambda f: f.startswith(key), ckpts))
    # get whole file path
    last_ckpt = os.path.join(path, ckpts[-1])

    return last_ckpt

def get_audio_models(model_config, args, model_type=None, ckpt_path=None):
    """
    Get the Deep Audio Net as encoder backbone and the evaluator with parameters moved to GPU.
    """
    audio_net = ConvLSTM_Audio(
        input_dim=model_config['AUDIO_NET']['INPUT_DIM'],
        output_dim=model_config['AUDIO_NET']['OUTPUT_DIM'],
        conv_hidden=model_config['AUDIO_NET']['CONV_HIDDEN'],
        lstm_hidden=model_config['AUDIO_NET']['LSTM_HIDDEN'],
        num_layers=model_config['AUDIO_NET']['NUM_LAYERS'],
        activation=model_config['AUDIO_NET']['ACTIVATION'],
        norm=model_config['AUDIO_NET']['NORM'],
        dropout=model_config['AUDIO_NET']['DROPOUT']
    )

    evaluator = Evaluator(
        feature_dim=model_config['EVALUATOR']['INPUT_FEATURE_DIM'],
        output_dim=model_config['EVALUATOR']['CLASSES_RESOLUTION'],
        predict_type=model_config['EVALUATOR']['PREDICT_TYPE'],
        num_subscores=model_config['EVALUATOR']['N_SUBSCORES']
    )

    # Move models to device
    device = args.device
    audio_net = audio_net.to(device)
    evaluator = evaluator.to(device)

    # Determine weights path based on configuration
    if model_config['WEIGHTS']['TYPE'].lower() == 'last':
        assert ckpt_path is not None, "'ckpt_path' must be given for the function 'get_models'"
        weights_path = find_last_ckpts(path=ckpt_path, key=model_type, date=model_config['WEIGHTS']['DATE'])
    elif model_config['WEIGHTS']['TYPE'].lower() == 'absolute_path':
        weights_path = str(model_config['WEIGHTS']['CUSTOM_ABSOLUTE_PATH'])
    elif model_config['WEIGHTS']['TYPE'].lower() != 'new':
        weights_path = os.path.join(model_config['WEIGHTS']['PATH'], model_config['WEIGHTS']['NAME'])
    else:
        weights_path = None

    # Load model weights
    if weights_path is not None:
        model_config['WEIGHTS']['INCLUDED'] = [x.lower() for x in model_config['WEIGHTS']['INCLUDED']]
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

        # Remove 'module.' prefix if present
        def remove_module_prefix(state_dict):
            return {k.replace("module.", ""): v for k, v in state_dict.items()}

        if 'audio_net' in model_config['WEIGHTS']['INCLUDED']:
            print(f"Loading Deep Audio Net weights from {weights_path}")
            audio_net_state_dict = remove_module_prefix(checkpoint['audio_net'])
            audio_net.load_state_dict(audio_net_state_dict, strict=False)

        if 'evaluator' in model_config['WEIGHTS']['INCLUDED']:
            print(f"Loading MUSDL weights from {weights_path}")
            evaluator_state_dict = remove_module_prefix(checkpoint['evaluator'])
            evaluator.load_state_dict(evaluator_state_dict, strict=False)

    return audio_net, evaluator


def get_text_models(model_config, args, model_type=None, ckpt_path=None):
    """
    Get the Deep Text Net as encoder backbone and the evaluator with parameters moved to GPU.
    """
    text_net = ConvLSTM_Text(input_dim=model_config['TEXT_NET']['INPUT_DIM'],
                             output_dim=model_config['TEXT_NET']['OUTPUT_DIM'],
                             conv_hidden=model_config['TEXT_NET']['CONV_HIDDEN'],
                             lstm_hidden=model_config['TEXT_NET']['LSTM_HIDDEN'],
                             num_layers=model_config['TEXT_NET']['NUM_LAYERS'],
                             activation=model_config['TEXT_NET']['ACTIVATION'],
                             norm=model_config['TEXT_NET']['NORM'],
                             dropout=model_config['TEXT_NET']['DROPOUT'])

    evaluator = models.Text_ConvLSTM.models.evaluator.Evaluator(feature_dim=model_config['EVALUATOR']['INPUT_FEATURE_DIM'],
                          output_dim=model_config['EVALUATOR']['CLASSES_RESOLUTION'],
                          predict_type=model_config['EVALUATOR']['PREDICT_TYPE'],
                          num_subscores=model_config['EVALUATOR']['N_SUBSCORES'])

    if len(args.gpu.split(',')) > 1:
        text_net = nn.DataParallel(text_net)
        evaluator = nn.DataParallel(evaluator)

    # move to GPU
    text_net = text_net.to(args.device)
    evaluator = evaluator.to(args.device)

    # find the model weights
    if model_config['WEIGHTS']['TYPE'].lower() == 'last':
        assert ckpt_path is not None, \
            "'ckpt_path' must be given for the function 'get_models' "
        weights_path = find_last_ckpts(path=ckpt_path,
                                       key=model_type,
                                       date=model_config['WEIGHTS']['DATE'])

    elif model_config['WEIGHTS']['TYPE'].lower() == 'absolute_path':
        assert model_config['WEIGHTS']['CUSTOM_ABSOLUTE_PATH'] is not None, \
            "'CUSTOM_ABSOLUTE_PATH' (absolute path to wights file) in config file under 'WEIGHTS' must be given"
        assert os.path.isabs(model_config['WEIGHTS']['CUSTOM_ABSOLUTE_PATH']), \
            "The given 'CUSTOM_ABSOLUTE_PATH' is not an absolute path to wights file, please give an absolute"

        weights_path = str(model_config['WEIGHTS']['CUSTOM_ABSOLUTE_PATH'])

    elif model_config['WEIGHTS']['TYPE'].lower() != 'new':
        assert model_config['WEIGHTS']['NAME'] is not None, \
            "'NAME' (name of the wights file) in config file under 'WEIGHTS' must be given"
        weights_path = os.path.join(model_config['WEIGHTS']['PATH'], model_config['WEIGHTS']['NAME'])

    else:
        weights_path = None

    # load model weights
    if weights_path is not None:
        model_config['WEIGHTS']['INCLUDED'] = [x.lower() for x in model_config['WEIGHTS']['INCLUDED']]

        checkpoint = torch.load(weights_path,map_location=torch.device('cpu'))

        if 'text_net' in model_config['WEIGHTS']['INCLUDED']:
            print("Loading Deep Text Net weights from {}".format(weights_path))
            text_net.load_state_dict(checkpoint['text_net'])

        if 'evaluator' in model_config['WEIGHTS']['INCLUDED']:
            print("Loading MUSDL weights from {}".format(weights_path))
            evaluator.load_state_dict(checkpoint['evaluator'])

    return text_net, evaluator



def get_AT_models(model_config, args, model_type=None, ckpt_path=None):
    """
    Get the different deep model nets as encoder backbones and the evaluator with parameters moved to GPU.
    """
    # Initialize models
    audio_net = models.AT_ConvLSTM_Attention.models.convlstm.ConvLSTM_Audio(input_dim=model_config['AUDIO_NET']['INPUT_DIM'],
                               output_dim=model_config['AUDIO_NET']['OUTPUT_DIM'],
                               conv_hidden=model_config['AUDIO_NET']['CONV_HIDDEN'],
                               lstm_hidden=model_config['AUDIO_NET']['LSTM_HIDDEN'],
                               num_layers=model_config['AUDIO_NET']['NUM_LAYERS'],
                               activation=model_config['AUDIO_NET']['ACTIVATION'],
                               norm=model_config['AUDIO_NET']['NORM'],
                               dropout=model_config['AUDIO_NET']['DROPOUT'])

    text_net = models.AT_ConvLSTM_Attention.models.convlstm.ConvLSTM_Text(input_dim=model_config['TEXT_NET']['INPUT_DIM'],
                             output_dim=model_config['TEXT_NET']['OUTPUT_DIM'],
                             conv_hidden=model_config['TEXT_NET']['CONV_HIDDEN'],
                             lstm_hidden=model_config['TEXT_NET']['LSTM_HIDDEN'],
                             num_layers=model_config['TEXT_NET']['NUM_LAYERS'],
                             activation=model_config['TEXT_NET']['ACTIVATION'],
                             norm=model_config['TEXT_NET']['NORM'],
                             dropout=model_config['TEXT_NET']['DROPOUT'])

    fusion_net = models.AT_ConvLSTM_Attention.models.fusion1.Bottleneck1(inplanes=model_config['FUSION_NET']['INPUT_DIM'],
                            planes=model_config['FUSION_NET']['HIDDEN_DIM'],
                            base_width=model_config['FUSION_NET']['BASE_WIDTH'],
                            fuse_type=model_config['FUSION_NET']['FUSE_TYPE'])

    evaluator = models.AT_ConvLSTM_Attention.models.evaluator.Evaluator(feature_dim=model_config['EVALUATOR']['INPUT_FEATURE_DIM'],
                          output_dim=model_config['EVALUATOR']['CLASSES_RESOLUTION'],
                          predict_type=model_config['EVALUATOR']['PREDICT_TYPE'],
                          num_subscores=model_config['EVALUATOR']['N_SUBSCORES'])

    # Handle DataParallel for multiple GPUs
    if len(args.gpu.split(',')) > 1:
        audio_net = nn.DataParallel(audio_net)
        text_net = nn.DataParallel(text_net)
        fusion_net = nn.DataParallel(fusion_net)
        evaluator = nn.DataParallel(evaluator)

    # Move to GPU
    audio_net = audio_net.to(args.device)
    text_net = text_net.to(args.device)
    fusion_net = fusion_net.to(args.device)
    evaluator = evaluator.to(args.device)

    # Find model weights
    if model_config['WEIGHTS']['TYPE'].lower() == 'last':
        assert ckpt_path is not None, "'ckpt_path' must be given for the function 'get_models' "
        weights_path = find_last_ckpts(path=ckpt_path, key=model_type, date=model_config['WEIGHTS']['DATE'])
    elif model_config['WEIGHTS']['TYPE'].lower() == 'absolute_path':
        assert model_config['WEIGHTS']['CUSTOM_ABSOLUTE_PATH'] is not None, "'CUSTOM_ABSOLUTE_PATH' must be given"
        assert os.path.isabs(model_config['WEIGHTS']['CUSTOM_ABSOLUTE_PATH']), "The given 'CUSTOM_ABSOLUTE_PATH' is not an absolute path"
        weights_path = str(model_config['WEIGHTS']['CUSTOM_ABSOLUTE_PATH'])
    elif model_config['WEIGHTS']['TYPE'].lower() != 'new':
        assert model_config['WEIGHTS']['NAME'] is not None, "'NAME' (name of the weights file) must be given"
        weights_path = os.path.join(model_config['WEIGHTS']['PATH'], model_config['WEIGHTS']['NAME'])
    else:
        weights_path = None

    # Load model weights
    if weights_path is not None:
        model_config['WEIGHTS']['INCLUDED'] = [x.lower() for x in model_config['WEIGHTS']['INCLUDED']]
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

        # If the checkpoint has 'module.' prefix, remove it
        if 'module.' in list(checkpoint.keys())[0]:
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

        # Load weights into models
        if 'audio_net' in model_config['WEIGHTS']['INCLUDED']:
            print(f"Loading Deep Audio Net weights from {weights_path}")
            audio_net.load_state_dict(checkpoint.get('audio_net', {}), strict=False)

        if 'text_net' in model_config['WEIGHTS']['INCLUDED']:
            print(f"Loading Deep Text Net weights from {weights_path}")
            text_net.load_state_dict(checkpoint.get('text_net', {}), strict=False)

        if 'fusion_net' in model_config['WEIGHTS']['INCLUDED']:
            print(f"Loading Attention Fusion Layer weights from {weights_path}")
            fusion_net.load_state_dict(checkpoint.get('fusion_net', {}), strict=False)

        if 'evaluator' in model_config['WEIGHTS']['INCLUDED']:
            print(f"Loading MUSDL weights from {weights_path}")
            evaluator.load_state_dict(checkpoint.get('evaluator', {}), strict=False)

    return audio_net, text_net, fusion_net, evaluator


def load_audio_model(config_file, gpu_ids):
    # Load config
    with open(config_file, 'r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh)

    # Set up GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a dummy args object to pass GPU and device information
    class Args:
        def __init__(self, gpu, device):
            self.gpu = gpu
            self.device = device

    args = Args(gpu=gpu_ids, device=device)

    # Get models
    ckpt_path = os.path.join(config['CKPTS_DIR'], config['TYPE'])
    model_type = config['TYPE']
    audio_net, evaluator = get_audio_models(config['MODEL'], args, model_type, ckpt_path)
    audio_net.eval()  # Set model to evaluation mode
    evaluator.eval()

    return audio_net, evaluator, config

def load_text_model(config_file, gpu_ids):
    # Load config
    with open(config_file, 'r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh)

    # Set up GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a dummy args object to pass GPU and device information
    class Args:
        def __init__(self, gpu, device):
            self.gpu = gpu
            self.device = device

    args = Args(gpu=gpu_ids, device=device)

    # Get models
    ckpt_path = os.path.join(config['CKPTS_DIR'], config['TYPE'])
    model_type = config['TYPE']
    text_net, evaluator = get_text_models(config['MODEL'], args, model_type, ckpt_path)
    text_net.eval()  # Set model to evaluation mode
    evaluator.eval()

    return text_net, evaluator, config

def load_AT_model(config_file, gpu_ids):
    # Load config
    with open(config_file, 'r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh)

    # Set up GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a dummy args object to pass GPU and device information
    class Args:
        def __init__(self, gpu, device):
            self.gpu = gpu
            self.device = device

    args = Args(gpu=gpu_ids, device=device)

    # Get models
    ckpt_path = os.path.join(config['CKPTS_DIR'], config['TYPE'])
    model_type = config['TYPE']
    audio_net, text_net, fusion_net, evaluator = get_AT_models(config['MODEL'], args, model_type, ckpt_path)
    audio_net.eval()  # Set model to evaluation mode
    text_net.eval()  # Set model to evaluation mode
    fusion_net.eval()
    evaluator.eval()

    return audio_net, text_net, fusion_net, evaluator, config


def compute_score(probs, evaluator_config, args):
    if evaluator_config['PREDICT_TYPE'] == 'phq-subscores':
        # calculate expectation & denormalize & sort
        factor = evaluator_config['N_CLASSES'] / evaluator_config['CLASSES_RESOLUTION']
        subscores_pred = torch.stack([prob.argmax(dim=-1) * factor
                                      for prob in probs], dim=1).sort()[0].to(int).to(
            float)  # (number of batch, num_subscores)

        # sum the subscores to get
        score_pred = torch.sum(subscores_pred, dim=1)  # number of batch x 1

        return score_pred.to(args.device)

    else:
        factor = evaluator_config['N_CLASSES'] / evaluator_config['CLASSES_RESOLUTION']
        score_pred = (probs.argmax(dim=-1) * factor).to(int).to(float)

        return score_pred.to(args.device)



if input_option0 == "Offline Diagnosis":
    # 使用侧边栏来创建输入选项
    st.sidebar.header("Input Options")
    #
    # input_option = st.sidebar.selectbox(
    #     "Diagnosis Mode:",
    #     ("Offline Diagnosis", "Real-time Diagnosis")
    # )

    # 选择输入类型
    input_option1 = st.sidebar.selectbox(
        "Select Input Type:",
        ("CSV File for Text", "Audio Only", "CSV File and Audio")
    )

    input_option2 = st.sidebar.selectbox(
        "Select a Model:",
        ("Model: Conv1D-BiLSTM",)
    )

    # 根据选择展示不同的输入组件
    text_file = None
    audio_file = None

    if input_option1 == "CSV File for Text" or input_option1 == "CSV File and Audio":
        text_file = st.sidebar.file_uploader("Upload your text CSV file:", type=["csv"])

    if input_option1 == "Audio Only" or input_option1 == "CSV File and Audio":
        audio_file = st.sidebar.file_uploader("Upload an audio file:", type=["wav", "mp3"])
    # 本地文件路径
    csv_file_path = "D:\\Development of a Multimodal Mental Health Diagnosis Model\\queries.csv"

    # 在侧边栏添加 CSV 模板下载按钮
    st.sidebar.download_button(
        label="Download CSV Template",
        data=load_csv_for_download(csv_file_path),
        file_name="queries.csv",
        mime="text/csv"
    )
    # 确保创建音频存储文件夹
    os.makedirs("uploaded_audios", exist_ok=True)
    # 提交按钮
    if st.sidebar.button("Submit"):
        df = pd.read_csv(SAVE_PATH)
        new_row = {"Text": None, "Audio": None}

        # 处理CSV文本文件
        if text_file is not None and audio_file is None:
            text_df = pd.read_csv(text_file)
            text_df = text_df.iloc[:-2, :]  # Drop last two rows if they are not relevant
            sent2vec = SentenceTransformer('all-mpnet-base-v2')
            text_feature = sentence_embedding(text_df, model=sent2vec)
            print(text_feature.shape)
            window_size = 10
            overlap_size = 2
            num_frame, text_feature_all= sliding_window_text(text_feature, window_size, overlap_size)
            st.header("Pre-processing Modules")
            st.header('Text data pre-processing')
            plot_text_embedings(text_feature_all[0])
            # new_row["Text"] = text_feature.tolist()
            # 诊断结果
            config_file = 'D:\\Development of a Multimodal Mental Health Diagnosis Model\models\Text_ConvLSTM/config/config_inference1.yaml'  # Update with your config path
            gpu_ids = '2,3'  # Adjust according to your GPU setup
            st.header("Depression diagnosis results")
            text_net, evaluator, config = load_text_model(config_file, gpu_ids)
            score_pred_all=[]
            cnt=0
            for text_feature in text_feature_all:
                # 确保 mel_spectrogram 是一个 torch.Tensor
                if not isinstance(text_feature, torch.Tensor):
                    text_feature = torch.tensor(text_feature)
                # 添加通道维度
                text_feature = text_feature.unsqueeze(0).permute(0, 2, 1).contiguous()
                text_feature = text_feature.float().to(
                    torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                text_features = text_net(text_feature)
                probs = evaluator(text_features)
                subscores_pred = torch.stack([prob.argmax(dim=-1)/8
                                              for prob in probs], dim=1).sort()[0].to(int).to(
                    float)  # (number of batch, num_subscores)
                # sum the subscores to get
                score_pred = torch.sum(subscores_pred, dim=1).item()  # number of batch x 1
                st.write(f"Prediction result of clip_{cnt + 1}: {score_pred}")
                score_pred_all.append(score_pred)
                cnt += 1
            final_score = sum([1 if score >= 10 else 0 for score in score_pred_all])
            if final_score >= num_frame * 0.5:
                # st.write(f"Prediction result: Depression")
                # st.write("You may be suffering from depression, please contact a psychologist for treatment.")
                st.markdown(
                    "<h3 style='font-size:24px; color:red;'>Prediction result: Depression</h3>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<h3 style='font-size:24px; color:red;'>You may be suffering from depression, please contact a psychologist for treatment.</h3>",
                    unsafe_allow_html=True
                )
            else:
                # st.write(f"Prediction result: Not Depression")
                st.markdown(
                    "<h3 style='font-size:24px; color:green;'>Prediction result: Not Depression</h3>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<h3 style='font-size:24px; color:green;'>Your score indicates that you are not likely suffering from depression. It's great to see you feeling well!</h3>",
                    unsafe_allow_html=True
                )
                # st.write(
                #     "Your score indicates that you are not likely suffering from depression. It's great to see you feeling well!")

                # # sum the subscores to get
                # score_pred = torch.sum(subscores_pred, dim=1).item()  # number of batch x 1
                # st.write(f"Prediction result: {score_pred}")
                # if score_pred < 10:
                #     st.write(
                #         f"Your score indicates that you are not likely suffering from depression. It's great to see you feeling well!")
                # else:
                #     st.write(f"You may be suffering from depression, please contact a psychologist for treatment.")

        elif text_file is None and audio_file is not None:
            audio_file_name = audio_file.name
            with open(f"uploaded_audios/{audio_file_name}", "wb") as f:
                f.write(audio_file.getbuffer())
            audio, audio_sr = load_audio(f"uploaded_audios/{audio_file_name}")
            mel_spectrogram = normalize(
                convert_mel_spectrogram(audio, audio_sr, frame_size=2048, hop_size=533, num_mel_bands=80))
            # sent2vec = SentenceTransformer('all-mpnet-base-v2')
            # text_feature = sentence_embedding(text_df, model=sent2vec)
            window_size = 60  # 60s
            overlap_size = 10  # 10s
            num_frame, mel_spectrogram_all= sliding_window_audio(mel_spectrogram, window_size, overlap_size)
            st.header("Pre-processing Modules")
            st.header('Audio data pre-processing')
            plot_mel_spectrogram(mel_spectrogram_all[0])
            # 诊断结果
            # Load the model once
            config_file = 'D:\\Development of a Multimodal Mental Health Diagnosis Model\models\Audio_ConvLSTM/config/config_inference1.yaml'  # Update with your config path
            gpu_ids = '2,3'  # Adjust according to your GPU setup
            st.header("Depression diagnosis results")
            audio_net, evaluator, config = load_audio_model(config_file, gpu_ids)
            score_pred_all=[]
            cnt=0
            for mel_spectrogram in mel_spectrogram_all:
                # 确保 mel_spectrogram 是一个 torch.Tensor
                if not isinstance(mel_spectrogram, torch.Tensor):
                    mel_spectrogram = torch.tensor(mel_spectrogram)

                # 添加通道维度
                mel_spectrogram = mel_spectrogram.unsqueeze(0).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                audio_features = audio_net(mel_spectrogram)
                probs = evaluator(audio_features)
                subscores_pred = torch.stack([prob.argmax(dim=-1)
                                              for prob in probs], dim=1).sort()[0].to(int).to(
                    float)  # (number of batch, num_subscores)

                # sum the subscores to get
                score_pred = torch.sum(subscores_pred, dim=1).item()  # number of batch x 1
                st.write(f"Prediction result of clip_{cnt+1}: {score_pred}")
                score_pred_all.append(score_pred)
                cnt+=1
            final_score = sum([1 if score >= 10 else 0 for score in score_pred_all])
            if final_score >= num_frame * 0.5:
                # st.write(f"Prediction result: Depression")
                # st.write("You may be suffering from depression, please contact a psychologist for treatment.")
                st.markdown(
                    "<h3 style='font-size:24px; color:red;'>Prediction result: Depression</h3>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<h3 style='font-size:24px; color:red;'>You may be suffering from depression, please contact a psychologist for treatment.</h3>",
                    unsafe_allow_html=True
                )
            else:
                # st.write(f"Prediction result: Not Depression")
                st.markdown(
                    "<h3 style='font-size:24px; color:green;'>Prediction result: Not Depression</h3>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<h3 style='font-size:24px; color:green;'>Your score indicates that you are not likely suffering from depression. It's great to see you feeling well!</h3>",
                    unsafe_allow_html=True
                )
                # st.write(
                #     "Your score indicates that you are not likely suffering from depression. It's great to see you feeling well!")
            # st.write(f"Prediction result: {score_pred}")
            # if score_pred < 10:
            #     st.write(f"Your score indicates that you are not likely suffering from depression. It's great to see you feeling well!")
            # else:
            #     st.write(f"You may be suffering from depression, please contact a psychologist for treatment.")


        elif text_file is not None and audio_file is not None:
            audio_file_name = audio_file.name
            with open(f"uploaded_audios/{audio_file_name}", "wb") as f:
                f.write(audio_file.getbuffer())
            # 处理音频文件
            audio, audio_sr = load_audio(f"uploaded_audios/{audio_file_name}")
            text_df = pd.read_csv(text_file)
            clipped_audio = audio_clipping(audio, audio_sr, text_df, zero_padding=False)
            mel_spectrogram = normalize(
                convert_mel_spectrogram(clipped_audio, audio_sr, frame_size=2048, hop_size=533, num_mel_bands=80))
            text_df = text_df.iloc[:-2, :]  # Drop last two rows if they are not relevant
            sent2vec = SentenceTransformer('all-mpnet-base-v2')
            text_feature = sentence_embedding(text_df, model=sent2vec)
            window_size = 60  # 60s
            overlap_size = 10  # 10s
            num_frame, mel_spectrogram_all, text_feature_all= sliding_window(mel_spectrogram, text_feature,
                                       window_size, overlap_size)
            st.header("Pre-processing Modules")
            st.header('Audio data pre-processing')
            plot_mel_spectrogram(mel_spectrogram_all[0])
            st.header('Text data pre-processing')
            plot_text_embedings(text_feature_all[0])
            # 诊断结果
            # Load the model once
            config_file = 'D:\\Development of a Multimodal Mental Health Diagnosis Model\models\AT_ConvLSTM_Attention/config/config_inference2.yaml'  # Update with your config path
            gpu_ids = '2,3'  # Adjust according to your GPU setup
            st.header("Depression diagnosis results")
            audio_net, text_net, fusion_net, evaluator, config = load_AT_model(config_file, gpu_ids)
            score_pred_all=[]
            cnt=0
            for mel_spectrogram,text_feature in zip(mel_spectrogram_all,text_feature_all):
                # 确保 mel_spectrogram 是一个 torch.Tensor
                if not isinstance(mel_spectrogram, torch.Tensor):
                    mel_spectrogram = torch.tensor(mel_spectrogram)
                if not isinstance(text_feature, torch.Tensor):
                    text_feature = torch.tensor(text_feature)
                # 添加通道维度
                mel_spectrogram = mel_spectrogram.unsqueeze(0).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                text_feature = text_feature.unsqueeze(0).permute(0, 2, 1).contiguous()
                text_feature = text_feature.float().to(
                    torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                audio_features = audio_net(mel_spectrogram)
                text_features = text_net(text_feature)
                all_features = torch.stack([audio_features, text_features], dim=1)
                # fused_features = all_features.view(B, -1)  # shape: (B, num_modal x audio net output dim)
                fused_features = fusion_net(all_features)
                B, C, F = fused_features.shape
                fused_features = fused_features.view(B, -1)  # shape: (B, num_modal x audio net output dim)
                probs = evaluator(fused_features)
                print(probs)
                subscores_pred = torch.stack([prob.argmax(dim=-1)/8
                                              for prob in probs], dim=1).sort()[0].to(int).to(
                    float)  # (number of batch, num_subscores)
                print(subscores_pred)
                # sum the subscores to get
                score_pred = torch.sum(subscores_pred, dim=1).item()  # number of batch x 1
                st.write(f"Prediction result of clip_{cnt+1}: {score_pred}")
                score_pred_all.append(score_pred)
                cnt+=1
            final_score = sum([1 if score >= 10 else 0 for score in score_pred_all])
            if final_score >= num_frame * 0.5:
                # st.write(f"Prediction result: Depression")
                # st.write("You may be suffering from depression, please contact a psychologist for treatment.")
                st.markdown(
                    "<h3 style='font-size:24px; color:red;'>Prediction result: Depression</h3>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<h3 style='font-size:24px; color:red;'>You may be suffering from depression, please contact a psychologist for treatment.</h3>",
                    unsafe_allow_html=True
                )
            else:
                # st.write(f"Prediction result: Not Depression")
                st.markdown(
                    "<h3 style='font-size:24px; color:green;'>Prediction result: Not Depression</h3>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<h3 style='font-size:24px; color:green;'>Your score indicates that you are not likely suffering from depression. It's great to see you feeling well!</h3>",
                    unsafe_allow_html=True
                )
                # st.write(
                #     "Your score indicates that you are not likely suffering from depression. It's great to see you feeling well!")

elif input_option0 == "Real-time Diagnosis":
    # 使用侧边栏来创建输入选项
    st.sidebar.header("Input Options")
    #
    # 选择输入类型
    input_option1 = st.sidebar.selectbox(
        "Select Input Type:",
        ("Text Answer Questions", "Voice Answer Questions", "Multimodal Method")
    )

    input_option2 = st.sidebar.selectbox(
        "Select a Model:",
        ("Model: Conv1D-BiLSTM",)
    )

    # 根据选择展示不同的输入组件
    text_file = None
    audio_file = None

    os.makedirs("uploaded_audios", exist_ok=True)

    if input_option1 == "Text Answer Questions":
        @st.cache_data
        def load_questions(file_path):
            try:
                data = pd.read_csv(file_path)
                if 'Questions' in data.columns:
                    return data['Questions'].tolist()
                else:
                    st.error("The file does not contain a 'Questions' column.")
                    return []
            except FileNotFoundError:
                st.error("The file 'questionaire.csv' was not found.")
                return []


        st.write("The following is a short questionnaire, please enter your answer：")
        # 加载数据
        questions = load_questions("questionaire1.csv")

        # # 随机选取10个问题
        # random_questions = random.sample(questions, min(10, len(questions)))

        # 清除所有回答并刷新问题的按钮
        if st.sidebar.button("Clear All Answers and Refresh Questions"):
            # 清除回答并生成新的随机问题
            st.session_state.random_questions = random.sample(
                questions, min(15, len(questions))
            )
            st.session_state.answers = [""] * len(st.session_state.random_questions)
            st.sidebar.success("Questions and answers have been reset.")
        # 检查是否已经生成随机问题，如果没有则生成并存储到 session_state
        if "random_questions" not in st.session_state:
            st.session_state.random_questions = random.sample(
                questions, min(15, len(questions))
            )
            # 初始化回答列表
            st.session_state.answers = [""] * len(st.session_state.random_questions)

        # 从 session_state 中获取随机问题
        random_questions = st.session_state.random_questions
        answers = []  # 用于存储所有回答

        # 显示问题和回答框
        for i, question in enumerate(random_questions):
            st.write(f"**Question {i + 1}: {question}**")
            answer=st.text_area(f"Your answer(question {i + 1})", key=f"answer_{i}")
            answers.append(answer)  # 将每个回答加入列表
        # 拼接所有回答并保存为 CSV
        if st.button("Submit"):
            text_df = pd.DataFrame({"Questions": random_questions, "Text": answers})
            combined_text = " ".join(answers)  # 拼接所有回答
            text_df.to_csv("answers.csv", index=False)  # 保存 CSV
            # st.success("Your answers have been saved to 'answers.csv'.")

            # 显示拼接后的文本和 DataFrame
            st.write("**All Combined Answers:**")
            st.write(combined_text)
            # st.write("**DataFrame of Answers:**")
            # st.dataframe(text_df)
            text_df = text_df.iloc[:, :]  # Drop last two rows if they are not relevant
            sent2vec = SentenceTransformer('all-mpnet-base-v2')
            text_feature = sentence_embedding(text_df, model=sent2vec)
            print(text_feature.shape)
            window_size = 10
            overlap_size = 2
            num_frame, text_feature_all= sliding_window_text(text_feature, window_size, overlap_size)
            st.header("Pre-processing Modules")
            st.header('Text data pre-processing')
            plot_text_embedings(text_feature_all[0])
            # new_row["Text"] = text_feature.tolist()
            # 诊断结果
            config_file = 'D:\\Development of a Multimodal Mental Health Diagnosis Model\models\Text_ConvLSTM/config/config_inference1.yaml'  # Update with your config path
            gpu_ids = '2,3'  # Adjust according to your GPU setup
            st.header("Depression diagnosis results")
            text_net, evaluator, config = load_text_model(config_file, gpu_ids)
            score_pred_all=[]
            cnt=0
            for text_feature in text_feature_all:
                # 确保 mel_spectrogram 是一个 torch.Tensor
                if not isinstance(text_feature, torch.Tensor):
                    text_feature = torch.tensor(text_feature)
                # 添加通道维度
                text_feature = text_feature.unsqueeze(0).permute(0, 2, 1).contiguous()
                text_feature = text_feature.float().to(
                    torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                text_features = text_net(text_feature)
                probs = evaluator(text_features)
                subscores_pred = torch.stack([prob.argmax(dim=-1)/8
                                              for prob in probs], dim=1).sort()[0].to(int).to(
                    float)  # (number of batch, num_subscores)
                # sum the subscores to get
                score_pred = torch.sum(subscores_pred, dim=1).item()  # number of batch x 1
                st.write(f"Prediction result of clip_{cnt + 1}: {score_pred}")
                score_pred_all.append(score_pred)
                cnt += 1
            final_score = sum([1 if score >= 10 else 0 for score in score_pred_all])
            if final_score >= num_frame * 0.5:
                st.markdown(
                    "<h3 style='font-size:24px; color:red;'>Prediction result: Depression</h3>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<h3 style='font-size:24px; color:red;'>You may be suffering from depression, please contact a psychologist for treatment.</h3>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<h3 style='font-size:24px; color:green;'>Prediction result: Not Depression</h3>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<h3 style='font-size:24px; color:green;'>Your score indicates that you are not likely suffering from depression. It's great to see you feeling well!</h3>",
                    unsafe_allow_html=True
                )
                # st.write(
                #     "Your score indicates that you are not likely suffering from depression. It's great to see you feeling well!")

    if input_option1 == "Voice Answer Questions":
        def load_audio_as_array(audio_dict, target_sample_rate=16000):
            # 将音频字节保存为临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            with open(temp_file.name, "wb") as f:
                f.write(audio_dict['bytes'])

            # 使用 pydub 读取 WAV 文件并重新采样
            audio_segment = AudioSegment.from_wav(temp_file.name)
            audio_segment = audio_segment.set_frame_rate(target_sample_rate)

            # 保存重新采样的音频到临时文件
            resampled_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_segment.export(resampled_file.name, format="wav")

            # 使用 scipy.io.wavfile 读取重新采样后的音频文件
            sample_rate, audio_data = wav.read(resampled_file.name)
            return sample_rate, audio_data

        state = st.session_state

        @st.cache_data
        def load_questions(file_path):
            try:
                data = pd.read_csv(file_path)
                if 'Questions' in data.columns:
                    return data['Questions'].tolist()
                else:
                    st.error("The file does not contain a 'Questions' column.")
                    return []
            except FileNotFoundError:
                st.error("The file 'questionaire.csv' was not found.")
                return []


        st.write("The following is a short questionnaire, please speak your answer：")
        # 加载数据
        questions = load_questions("questionaire1.csv")
        # 检查是否已经生成随机问题，如果没有则生成并存储到 session_state
        if "random_questions" not in st.session_state:
            st.session_state.random_questions = random.sample(
                questions, min(15, len(questions))
            )

        # 从 session_state 中获取随机问题
        random_questions = st.session_state.random_questions
        answers = []  # 用于存储所有回答

        # 显示问题和回答框
        for i, question in enumerate(random_questions):
            st.write(f"**Question {i + 1}: {question}**")

        st.write("Record your voice, and play the recorded audio:")
        audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", key='recorder',format="wav")
        file_path = "record_audio.wav"

        if audio:
            with open(file_path, "wb") as f:
                f.write(audio['bytes'])
            # 加载音频数据并设置采样率为 16000 Hz
            audio_sr, audio_data = load_audio_as_array(audio, target_sample_rate=16000)
            st.write(f"Sample rate: {audio_sr}, Audio data shape: {audio_data.shape}")
            # 播放音频
            st.audio(audio['bytes'])
            if st.button("Submit"):
                audio_data = np.array(audio_data, dtype=np.float32)
                audio_data /= 32768.0  # 16-bit 音频最大值是 32767
                mel_spectrogram = normalize(
                    convert_mel_spectrogram(audio_data, audio_sr, frame_size=2048, hop_size=533, num_mel_bands=80))
                # sent2vec = SentenceTransformer('all-mpnet-base-v2')
                # text_feature = sentence_embedding(text_df, model=sent2vec)
                window_size = 20  # 60s
                overlap_size = 4  # 10s
                num_frame, mel_spectrogram_all = sliding_window_audio(mel_spectrogram, window_size, overlap_size)
                st.header("Pre-processing Modules")
                st.header('Audio data pre-processing')
                plot_mel_spectrogram(mel_spectrogram_all[0])
                # 诊断结果
                # Load the model once
                config_file = 'D:\\Development of a Multimodal Mental Health Diagnosis Model\models\Audio_ConvLSTM/config/config_inference1.yaml'  # Update with your config path
                gpu_ids = '2,3'  # Adjust according to your GPU setup
                st.header("Depression diagnosis results")
                audio_net, evaluator, config = load_audio_model(config_file, gpu_ids)
                score_pred_all = []
                cnt = 0
                for mel_spectrogram in mel_spectrogram_all:
                    # 确保 mel_spectrogram 是一个 torch.Tensor
                    if not isinstance(mel_spectrogram, torch.Tensor):
                        mel_spectrogram = torch.tensor(mel_spectrogram)

                    # 添加通道维度
                    mel_spectrogram = mel_spectrogram.unsqueeze(0).float().to(
                        torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                    audio_features = audio_net(mel_spectrogram)
                    probs = evaluator(audio_features)
                    subscores_pred = torch.stack([prob.argmax(dim=-1)
                                                  for prob in probs], dim=1).sort()[0].to(int).to(
                        float)  # (number of batch, num_subscores)

                    # sum the subscores to get
                    score_pred = torch.sum(subscores_pred, dim=1).item()  # number of batch x 1
                    st.write(f"Prediction result of clip_{cnt + 1}: {score_pred}")
                    score_pred_all.append(score_pred)
                    cnt += 1
                final_score = sum([1 if score >= 10 else 0 for score in score_pred_all])
                if final_score >= num_frame * 0.5:
                    # st.write(f"Prediction result: Depression")
                    # st.write("You may be suffering from depression, please contact a psychologist for treatment.")
                    st.markdown(
                        "<h3 style='font-size:24px; color:red;'>Prediction result: Depression</h3>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        "<h3 style='font-size:24px; color:red;'>You may be suffering from depression, please contact a psychologist for treatment.</h3>",
                        unsafe_allow_html=True
                    )
                else:
                    # st.write(f"Prediction result: Not Depression")
                    st.markdown(
                        "<h3 style='font-size:24px; color:green;'>Prediction result: Not Depression</h3>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        "<h3 style='font-size:24px; color:green;'>Your score indicates that you are not likely suffering from depression. It's great to see you feeling well!</h3>",
                        unsafe_allow_html=True
                    )
                    # st.write(
                    #     "Your score indicates that you are not likely suffering from depression. It's great to see you feeling well!")

    if input_option1 == "Multimodal Method":
        def load_audio_as_array(audio_dict, target_sample_rate=16000):
            # 将音频字节保存为临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            with open(temp_file.name, "wb") as f:
                f.write(audio_dict['bytes'])

            # 使用 pydub 读取 WAV 文件并重新采样
            audio_segment = AudioSegment.from_wav(temp_file.name)
            audio_segment = audio_segment.set_frame_rate(target_sample_rate)

            # 保存重新采样的音频到临时文件
            resampled_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_segment.export(resampled_file.name, format="wav")

            # 使用 scipy.io.wavfile 读取重新采样后的音频文件
            sample_rate, audio_data = wav.read(resampled_file.name)
            return sample_rate, audio_data


        import soundfile as sf
        def load_audio_file_as_array(audio_binary, target_sample_rate=16000):
            # Use `BytesIO` to simulate a file object from the binary content
            audio_stream = io.BytesIO(audio_binary)
            # Read the audio file
            data, samplerate = sf.read(audio_stream, dtype='float32')
            # Resample if needed
            if samplerate != target_sample_rate:
                import librosa
                data = librosa.resample(data, orig_sr=samplerate, target_sr=target_sample_rate)
                samplerate = target_sample_rate
            return samplerate, data
        @st.cache_data
        def load_questions(file_path):
            try:
                data = pd.read_csv(file_path)
                if 'Questions' in data.columns:
                    return data['Questions'].tolist()
                else:
                    st.error("The file does not contain a 'Questions' column.")
                    return []
            except FileNotFoundError:
                st.error("The file 'questionaire.csv' was not found.")
                return []


        st.write("The following is a short questionnaire, please enter your answer：")
        # 加载数据
        questions = load_questions("questionaire1.csv")

        # # 随机选取10个问题
        # random_questions = random.sample(questions, min(10, len(questions)))

        # 清除所有回答并刷新问题的按钮
        if st.sidebar.button("Clear All Answers and Refresh Questions"):
            # 清除回答并生成新的随机问题
            st.session_state.random_questions = random.sample(
                questions, min(15, len(questions))
            )
            st.session_state.answers = [""] * len(st.session_state.random_questions)
            st.sidebar.success("Questions and answers have been reset.")
        # 检查是否已经生成随机问题，如果没有则生成并存储到 session_state
        if "random_questions" not in st.session_state:
            st.session_state.random_questions = random.sample(
                questions, min(15, len(questions))
            )
            # 初始化回答列表
            st.session_state.answers = [""] * len(st.session_state.random_questions)

        # 从 session_state 中获取随机问题
        random_questions = st.session_state.random_questions
        answers = []  # 用于存储所有回答

        # 显示问题和回答框
        for i, question in enumerate(random_questions):
            st.write(f"**Question {i + 1}: {question}**")
            answer=st.text_area(f"Your answer(question {i + 1})", key=f"answer_{i}")
            answers.append(answer)  # 将每个回答加入列表

        state = st.session_state
        st.write("Record your voice, and play the recorded audio:")
        audio_file = st.sidebar.file_uploader("Upload an audio file:", type=["wav", "mp3"])
        if audio_file is not None:
            audio_bytes = audio_file.read()
            audio_sr, audio_data = load_audio_file_as_array(audio_bytes, target_sample_rate=16000)

        audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", key='recorder',format="wav")
        if audio:

            audio_sr, audio_data = load_audio_as_array(audio, target_sample_rate=16000)
            st.write(f"Sample rate: {audio_sr}, Audio data shape: {audio_data.shape}")

            st.audio(audio['bytes'])

        if st.button("Submit"):
            text_df = pd.DataFrame({"Questions": random_questions, "Text": answers})
            combined_text = " ".join(answers)
            text_df.to_csv("answers_multimodal.csv", index=False)


            st.write("**All Combined Answers:**")
            st.write(combined_text)
            text_df = text_df.iloc[:, :]  # Drop last two rows if they are not relevant
            sent2vec = SentenceTransformer('all-mpnet-base-v2')
            text_feature = sentence_embedding(text_df, model=sent2vec)
            print(text_feature.shape)

            audio_data = np.array(audio_data, dtype=np.float32)
            audio_data /= 32768.0
            mel_spectrogram = normalize(
                convert_mel_spectrogram(audio_data, audio_sr, frame_size=2048, hop_size=533, num_mel_bands=80))

            window_size = 60  # 60s
            overlap_size = 10  # 10s
            num_frame, mel_spectrogram_all, text_feature_all= sliding_window(mel_spectrogram, text_feature,
                                       window_size, overlap_size)
            st.header("Pre-processing Modules")
            st.header('Audio data pre-processing')
            plot_mel_spectrogram(mel_spectrogram_all[0])
            st.header('Text data pre-processing')
            plot_text_embedings(text_feature_all[0])

            # Load the model once
            config_file = 'D:\\Development of a Multimodal Mental Health Diagnosis Model\models\AT_ConvLSTM_Attention/config/config_inference2.yaml'  # Update with your config path
            gpu_ids = '2,3'  # Adjust according to your GPU setup
            st.header("Depression diagnosis results")
            audio_net, text_net, fusion_net, evaluator, config = load_AT_model(config_file, gpu_ids)
            score_pred_all=[]
            cnt=0
            for mel_spectrogram,text_feature in zip(mel_spectrogram_all,text_feature_all):

                if not isinstance(mel_spectrogram, torch.Tensor):
                    mel_spectrogram = torch.tensor(mel_spectrogram)
                if not isinstance(text_feature, torch.Tensor):
                    text_feature = torch.tensor(text_feature)

                mel_spectrogram = mel_spectrogram.unsqueeze(0).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                text_feature = text_feature.unsqueeze(0).permute(0, 2, 1).contiguous()
                text_feature = text_feature.float().to(
                    torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                audio_features = audio_net(mel_spectrogram)
                text_features = text_net(text_feature)
                all_features = torch.stack([audio_features, text_features], dim=1)
                # fused_features = all_features.view(B, -1)  # shape: (B, num_modal x audio net output dim)
                fused_features = fusion_net(all_features)
                B, C, F = fused_features.shape
                fused_features = fused_features.view(B, -1)  # shape: (B, num_modal x audio net output dim)
                probs = evaluator(fused_features)
                print(probs)
                subscores_pred = torch.stack([prob.argmax(dim=-1)/8
                                              for prob in probs], dim=1).sort()[0].to(int).to(
                    float)  # (number of batch, num_subscores)
                print(subscores_pred)
                # sum the subscores to get
                score_pred = torch.sum(subscores_pred, dim=1).item()  # number of batch x 1
                st.write(f"Prediction result of clip_{cnt+1}: {score_pred}")
                score_pred_all.append(score_pred)
                cnt+=1
            final_score = sum([1 if score >= 10 else 0 for score in score_pred_all])
            if final_score >= num_frame * 0.5:
                st.markdown(
                    "<h3 style='font-size:24px; color:red;'>Prediction result: Depression</h3>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<h3 style='font-size:24px; color:red;'>You may be suffering from depression, please contact a psychologist for treatment.</h3>",
                    unsafe_allow_html=True
                )
            else:
                # st.write(f"Prediction result: Not Depression")
                st.markdown(
                    "<h3 style='font-size:24px; color:green;'>Prediction result: Not Depression</h3>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<h3 style='font-size:24px; color:green;'>Your score indicates that you are not likely suffering from depression. It's great to see you feeling well!</h3>",
                    unsafe_allow_html=True
                )




