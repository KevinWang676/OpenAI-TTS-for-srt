import os
import torch
import librosa
import gradio as gr
from scipy.io.wavfile import write

import urllib.request
urllib.request.urlretrieve("https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/UVR-HP2.pth", "uvr5/uvr_model/UVR-HP2.pth")
urllib.request.urlretrieve("https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/UVR-HP5.pth", "uvr5/uvr_model/UVR-HP5.pth")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

import ffmpeg

def denoise(video_full, split_model):
    
    if os.path.exists("audio_full.wav"):
        os.remove("audio_full.wav")

    ffmpeg.input(video_full).output("audio_full.wav", ac=2, ar=44100).run()
    
    if split_model=="UVR-HP2":
        pre_fun = pre_fun_hp2
    else:
        pre_fun = pre_fun_hp5

    filename = "output"
    pre_fun._path_audio_("audio_full.wav", f"./{split_model}/", f"./{split_model}/", "mp3")
     
    return f"./{split_model}/vocal_audio_full.wav_10.mp3"


with gr.Blocks() as app:
    gr.Markdown("# <center>🌊💕🎶 OpenAI TTS - SRT文件一键AI配音</center>")
    gr.Markdown("### <center>🌟 只需上传SRT文件和原版配音文件即可，每次一集视频AI自动配音！Developed by Kevin Wang </center>")
    with gr.Row():
        with gr.Column():
            inp1 = gr.Video(label="请上传一集包含原声配音的视频", info="需要是.mp4视频文件")
            inp2 = gr.Dropdown(label="请选择用于分离伴奏的模型", info="UVR-HP5去除背景音乐效果更好，但会对人声造成一定的损伤", choices=["UVR-HP2", "UVR-HP5"], value="UVR-HP5")
            btn = gr.Button("一键去除背景音吧💕", variant="primary")
        with gr.Column():
            out1 = gr.Audio(label="为您合成的原声音频", type="filepath")

        btn.click(denoise, [inp1, inp2], [out1])
        
    gr.Markdown("### <center>注意❗：请勿生成会对任何个人或组织造成侵害的内容，请尊重他人的著作权和知识产权。用户对此程序的任何使用行为与程序开发者无关。</center>")
    gr.HTML('''
        <div class="footer">
                    <p>🌊🏞️🎶 - 江水东流急，滔滔无尽声。 明·顾璘
                    </p>
        </div>
    ''')

app.launch(show_error=True, share=True)
