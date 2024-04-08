import os
import torch
import librosa
import gradio as gr
from scipy.io.wavfile import write
from transformers import WavLMModel

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from speaker_encoder.voice_encoder import SpeakerEncoder

'''
def get_wavlm():
    os.system('gdown https://drive.google.com/uc?id=12-cB34qCTvByWT-QtOcZaqwwO21FLSqU')
    shutil.move('WavLM-Large.pt', 'wavlm')
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

print("Loading FreeVC-s...")
hps = utils.get_hparams_from_file("configs/freevc-s.json")
freevc_s = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).to(device)
_ = freevc_s.eval()
_ = utils.load_checkpoint("checkpoints/freevc-s.pth", freevc_s, None)

print("Loading WavLM for content...")
cmodel = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)


from openai import OpenAI

import ffmpeg
import urllib.request
urllib.request.urlretrieve("https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/UVR-HP2.pth", "uvr5/uvr_model/UVR-HP2.pth")
urllib.request.urlretrieve("https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/UVR-HP5.pth", "uvr5/uvr_model/UVR-HP5.pth")
urllib.request.urlretrieve("https://modelscope.cn/api/v1/models/Kevin676/rvc/repo?Revision=master&FilePath=freevc-24.pth", "checkpoints/freevc-24.pth")
urllib.request.urlretrieve("https://modelscope.cn/api/v1/models/Kevin676/rvc/repo?Revision=master&FilePath=pretrained_bak_5805000.pt", "speaker_encoder/ckpt/pretrained_bak_5805000.pt")

from uvr5.vr import AudioPre
weight_uvr5_root = "uvr5/uvr_model"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

func = AudioPre

pre_fun_hp2 = func(
  agg=int(10),
  model_path=os.path.join(weight_uvr5_root, "UVR-HP2.pth"),
  device="cuda",
  is_half=True,
)
pre_fun_hp5 = func(
  agg=int(10),
  model_path=os.path.join(weight_uvr5_root, "UVR-HP5.pth"),
  device="cuda",
  is_half=True,
)


def convert(api_key, text, tgt, voice, save_path):
    model = "FreeVC (24kHz)"
    with torch.no_grad():
        # tgt
        wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
        if model == "FreeVC" or model == "FreeVC (24kHz)":
            g_tgt = smodel.embed_utterance(wav_tgt)
            g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(device)
        else:
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).to(device)
            mel_tgt = mel_spectrogram_torch(
                wav_tgt,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
        # src
        client = OpenAI(api_key=api_key)

        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text,
        )

        response.stream_to_file("output_openai.mp3")

        src = "output_openai.mp3"
        wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
        wav_src = torch.from_numpy(wav_src).unsqueeze(0).to(device)
        c = cmodel(wav_src).last_hidden_state.transpose(1, 2).to(device)
        # infer
        if model == "FreeVC":
            audio = freevc.infer(c, g=g_tgt)
        elif model == "FreeVC-s":
            audio = freevc_s.infer(c, mel=mel_tgt)
        else:
            audio = freevc_24.infer(c, g=g_tgt)
        audio = audio[0][0].data.cpu().float().numpy()
        if model == "FreeVC" or model == "FreeVC-s":
            write(f"output/{save_path}.wav", hps.data.sampling_rate, audio)
        else:
            write(f"output/{save_path}.wav", 24000, audio)
    return f"output/{save_path}.wav"


class subtitle:
    def __init__(self,index:int, start_time, end_time, text:str):
        self.index = int(index)
        self.start_time = start_time
        self.end_time = end_time
        self.text = text.strip()
    def normalize(self,ntype:str,fps=30):
         if ntype=="prcsv":
              h,m,s,fs=(self.start_time.replace(';',':')).split(":")#seconds
              self.start_time=int(h)*3600+int(m)*60+int(s)+round(int(fs)/fps,2)
              h,m,s,fs=(self.end_time.replace(';',':')).split(":")
              self.end_time=int(h)*3600+int(m)*60+int(s)+round(int(fs)/fps,2)
         elif ntype=="srt":
             h,m,s=self.start_time.split(":")
             s=s.replace(",",".")
             self.start_time=int(h)*3600+int(m)*60+round(float(s),2)
             h,m,s=self.end_time.split(":")
             s=s.replace(",",".")
             self.end_time=int(h)*3600+int(m)*60+round(float(s),2)
         else:
             raise ValueError
    def add_offset(self,offset=0):
        self.start_time+=offset
        if self.start_time<0:
            self.start_time=0
        self.end_time+=offset
        if self.end_time<0:
            self.end_time=0
    def __str__(self) -> str:
        return f'id:{self.index},start:{self.start_time},end:{self.end_time},text:{self.text}'

def read_srt(uploaded_file):
    offset=0
    with open(uploaded_file.name,"r",encoding="utf-8") as f:
        file=f.readlines()
    subtitle_list=[]
    indexlist=[]
    filelength=len(file)
    for i in range(0,filelength):
        if " --> " in file[i]:
            is_st=True
            for char in file[i-1].strip().replace("\ufeff",""):
                if char not in ['0','1','2','3','4','5','6','7','8','9']:
                    is_st=False
                    break
            if is_st:
                indexlist.append(i) #get line id
    listlength=len(indexlist)
    for i in range(0,listlength-1):
        st,et=file[indexlist[i]].split(" --> ")
        id=int(file[indexlist[i]-1].strip().replace("\ufeff",""))
        text=""
        for x in range(indexlist[i]+1,indexlist[i+1]-2):
            text+=file[x]
        st=subtitle(id,st,et,text)
        st.normalize(ntype="srt")
        st.add_offset(offset=offset)
        subtitle_list.append(st)
    st,et=file[indexlist[-1]].split(" --> ")
    id=file[indexlist[-1]-1]
    text=""
    for x in range(indexlist[-1]+1,filelength):
        text+=file[x]
    st=subtitle(id,st,et,text)
    st.normalize(ntype="srt")
    st.add_offset(offset=offset)
    subtitle_list.append(st)
    return subtitle_list

from pydub import AudioSegment

def trim_audio(intervals, input_file_path, output_file_path):
    # load the audio file
    audio = AudioSegment.from_file(input_file_path)

    # iterate over the list of time intervals
    for i, (start_time, end_time) in enumerate(intervals):
        # extract the segment of the audio
        segment = audio[start_time*1000:end_time*1000]

        # construct the output file path
        output_file_path_i = f"{output_file_path}_{i}.wav"

        # export the segment to a file
        segment.export(output_file_path_i, format='wav')

import re

def sort_key(file_name):
    """Extract the last number in the file name for sorting."""
    numbers = re.findall(r'\d+', file_name)
    if numbers:
        return int(numbers[-1])
    return -1  # In case there's no number, this ensures it goes to the start.


def merge_audios(folder_path):
    output_file = "AI配音版.wav"
    # Get all WAV files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    # Sort files based on the last digit in their names
    sorted_files = sorted(files, key=sort_key)
    
    # Initialize an empty audio segment
    merged_audio = AudioSegment.empty()
    
    # Loop through each file, in order, and concatenate them
    for file in sorted_files:
        audio = AudioSegment.from_wav(os.path.join(folder_path, file))
        merged_audio += audio
        print(f"Merged: {file}")
    
    # Export the merged audio to a new file
    merged_audio.export(output_file, format="wav")
    return "AI配音版.wav"

import shutil

def convert_from_srt(apikey, filename, video_full, voice, split_model, multilingual):
    subtitle_list = read_srt(filename)
    
    if os.path.exists("audio_full.wav"):
        os.remove("audio_full.wav")

    ffmpeg.input(video_full).output("audio_full.wav", ac=2, ar=44100).run()
    
    if split_model=="UVR-HP2":
        pre_fun = pre_fun_hp2
    else:
        pre_fun = pre_fun_hp5

    filename = "output"
    pre_fun._path_audio_("audio_full.wav", f"./denoised/{split_model}/{filename}/", f"./denoised/{split_model}/{filename}/", "wav")
    if os.path.isdir("output"):
        shutil.rmtree("output")
    if multilingual==False:
        for i in subtitle_list:
            os.makedirs("output", exist_ok=True)
            trim_audio([[i.start_time, i.end_time]], f"./denoised/{split_model}/{filename}/vocal_audio_full.wav_10.wav", f"sliced_audio_{i.index}")
            print(f"正在合成第{i.index}条语音")
            print(f"语音内容：{i.text}")
            convert(apikey, i.text, f"sliced_audio_{i.index}_0.wav", voice, i.text + " " + str(i.index))
    else:
        for i in subtitle_list:
            os.makedirs("output", exist_ok=True)
            trim_audio([[i.start_time, i.end_time]], f"./denoised/{split_model}/{filename}/vocal_audio_full.wav_10.wav", f"sliced_audio_{i.index}")
            print(f"正在合成第{i.index}条语音")
            print(f"语音内容：{i.text.splitlines()[1]}")
            convert(apikey, i.text.splitlines()[1], f"sliced_audio_{i.index}_0.wav", voice, i.text.splitlines()[1] + " " + str(i.index))
     
    return merge_audios("output")


with gr.Blocks() as app:
    gr.Markdown("# <center>🌊💕🎶 OpenAI TTS - SRT文件一键AI配音</center>")
    gr.Markdown("### <center>🌟 只需上传SRT文件和原版配音文件即可，每次一集视频AI自动配音！Developed by Kevin Wang </center>")
    with gr.Row():
        with gr.Column():
            inp0 = gr.Textbox(type='password', label='请输入您的OpenAI API Key')
            inp1 = gr.File(file_count="single", label="请上传一集视频对应的SRT文件")
            inp2 = gr.Video(label="请上传一集包含原声配音的视频", info="需要是.mp4视频文件")
            inp3 = gr.Dropdown(choices=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'], label='请选择一个说话人提供基础音色', info="试听音色链接：https://platform.openai.com/docs/guides/text-to-speech/voice-options", value='alloy')
            inp4 = gr.Dropdown(label="请选择用于分离伴奏的模型", info="UVR-HP5去除背景音乐效果更好，但会对人声造成一定的损伤", choices=["UVR-HP2", "UVR-HP5"], value="UVR-HP5")
            inp5 = gr.Checkbox(label="SRT文件是否为双语字幕", info="若为双语字幕，请打勾选择（SRT文件中需要先出现中文字幕，后英文字幕；中英字幕各占一行）")
            btn = gr.Button("一键开启AI配音吧💕", variant="primary")
        with gr.Column():
            out1 = gr.Audio(label="为您生成的AI完整配音", type="filepath")

        btn.click(convert_from_srt, [inp0, inp1, inp2, inp3, inp4, inp5], [out1])
        
    gr.Markdown("### <center>注意❗：请勿生成会对任何个人或组织造成侵害的内容，请尊重他人的著作权和知识产权。用户对此程序的任何使用行为与程序开发者无关。</center>")
    gr.HTML('''
        <div class="footer">
                    <p>🌊🏞️🎶 - 江水东流急，滔滔无尽声。 明·顾璘
                    </p>
        </div>
    ''')

app.launch(share=True, show_error=True)
