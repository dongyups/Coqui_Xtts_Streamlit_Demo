# reference: https://huggingface.co/spaces/coqui/xtts/blob/main/app.py#L32

import streamlit as st
import warnings, os, re, random
from pydub import AudioSegment
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch, torchaudio
import numpy as np
warnings.simplefilter(action='ignore')

# 하이퍼파라미터 세팅
use_gpu = False # True: 0번 single-GPU 사용 & False: CPU 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_gpu else "-1"
seed_num = 777 # 수정가능, 결과값 고정을 위함
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed_num)
random.seed(seed_num)


# TTS 모델 캐싱
@st.cache_resource(show_spinner=True)
def Caching_XTTS_Model():
    config = XttsConfig()
    config.load_json("./tts_models--multilingual--multi-dataset--xtts_v2.0.2/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, 
        checkpoint_dir="./tts_models--multilingual--multi-dataset--xtts_v2.0.2", 
    )
    if use_gpu:
        model.cuda()
    return model
model = Caching_XTTS_Model()


# 제목
st.title("COQUI 개인 음성 TTS 합성 Demo")


# Part1: 개별 분류 코드 작성
supported_languages = [
    "  : -- Select Your Language -- :  ",
    "Arabic(아랍어) : ar", 
    "Brazilian Portuguese(포르투갈어) : pt", 
    "Mandarin Chinese(중국어) : zh-cn", 
    "Czech(체코어) : cs", 
    "Dutch(네덜란드어) : nl", 
    "English(영어) : en", 
    "French(프랑스어) : fr", 
    "German(독일어) : de", 
    "Italian(이탈리아어) : it", 
    "Polish(폴란드어) : pl", 
    "Russian(러시아어) : ru", 
    "Spanish(스페인어) : es", 
    "Turkish(터키어) : tr", 
    "Japanese(일본어) : ja", 
    "Korean(한국어) : ko", 
    "Hungarian(헝가리어) : hu", 
    "Hindi(힌디어) : hi"
]
chosen_lang = st.selectbox("언어를 선택해주세요.", supported_languages)
lang_code = chosen_lang.split(" : ")[-1]

name = st.text_input("성함을 영어로 작성해주세요. EX) 홍길동 -> gildong hong", value="")
button0 = st.button("Confirm")
if button0:
    if lang_code == ' ':
        st.warning("언어가 선택되지 않았습니다.", icon="⚠️")
    if name == '':
        st.warning("성함이 입력되지 않았습니다.", icon="⚠️")
    else:
        st.success(f"사용될 언어 코드와 성함: {lang_code} & {name}")
name = "_".join(name.lower().split())

st.markdown(f"선택한 '언어_이름' 으로 경로를 생성합니다. EX) ko_gildong_hong")
button1 = st.button("Submit")
personal_path_inputs = f"./voices/{lang_code}_{name}/inputs/"
personal_path_outputs = f"./voices/{lang_code}_{name}/outputs/"
if button1:
    # 경로설정
    os.makedirs(personal_path_inputs, exist_ok=True)  # 개별 음성파일 저장 경로
    os.makedirs(personal_path_outputs, exist_ok=True) # TTS 결과파일 저장 경로
    st.success('경로 생성됨')


# Part2: 개별 목소리 업로드/변환 및 로컬 저장
with st.form("upload-then-clear-form", clear_on_submit=True):
        file_list  = st.file_uploader(
            '음성파일을 업로드 하세요. 여러 파일을 한번에 업로드 하셔도 됩니다.', 
            type=['m4a','wav'], accept_multiple_files=True
        )
        button2 = st.form_submit_button("Convert")
        if button2:

            # 업로드 된 파일 로컬에 저장
            for file in file_list:
                with open(personal_path_inputs + file.name.lower(), 'wb') as f:
                    f.write(file.getbuffer())

            # 확장자 변환 및 trim
            for file in os.listdir(personal_path_inputs):
                # m4a 파일의 경우
                if len(file.split(".m4a")[0]) != len(file):
                    tobesaved = personal_path_inputs + file.split(".m4a")[0]+".wav"
                    audio = AudioSegment.from_file(personal_path_inputs + file, format="m4a")
                    audio.export(tobesaved, format="wav")
                    os.remove(personal_path_inputs + file) # m4a 파일 제거
                    audio = AudioSegment.from_wav(tobesaved)
                    audio = audio[:-200] # 윈도우 녹음기 사용시 마지막 노이즈 제거
                    audio.export(tobesaved, format="wav") # 덮어쓰기

                # wav 파일의 경우
                else:
                    tobesaved = personal_path_inputs + file
                    audio = AudioSegment.from_wav(tobesaved)
                    audio = audio[:-200] # 윈도우 녹음기 사용시 마지막 노이즈 제거
                    audio.export(tobesaved, format="wav") # 덮어쓰기

            del file_list
            st.success('변환 완료')


# Part3: 모델 인퍼런스
st.markdown("사용될 모델은 multilingual_xtts_v2.0.2 입니다.")
output_name = st.text_input(
    "TTS로 생성될 파일명을 입력하세요. \
    중복될 시 덮어씌워 집니다. \
    파일 확장자는 입력하지 않으셔도 됩니다.", 
    value=""
    )
tts_input = st.text_area("TTS로 변환할 텍스트를 입력하세요.")
prompt= re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)",r"\1 \2\2", tts_input)
button3 = st.button("Run")

if button3:
    # st.write(prompt)
    with st.spinner("변환 중..."):
        # 확인
        st.write("레퍼런스 파일: " + ", ".join(os.listdir(personal_path_inputs)))
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60,
            audio_path=[
                personal_path_inputs + x for x in os.listdir(personal_path_inputs)
            ]
        )
        out = model.inference(
            prompt,
            lang_code,
            gpt_cond_latent,
            speaker_embedding,
            repetition_penalty=5.0,
            temperature=0.75,
        )
        # HTML Display
        st.audio(np.expand_dims(np.array(out["wav"]), 0), sample_rate=24000)
        # 자동 저장
        torchaudio.save(personal_path_outputs+f"{output_name}.wav", 
                        torch.tensor(out["wav"]).unsqueeze(0), 24000)

        st.success('TTS 생성 및 저장 완료')