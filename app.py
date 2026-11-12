"""Qwen3-TTS 실험용 웹 UI: 목소리 디자인 / 프리셋 화자 + 톤 / 보이스 클로닝 / 캐릭터."""

import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "mps" else torch.float32
SIZE = os.environ.get("TTS_SIZE", "1.7B")  # CustomVoice/Base는 0.6B도 있음. VoiceDesign은 1.7B만 있음.

MODEL_IDS = {
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom": f"Qwen/Qwen3-TTS-12Hz-{SIZE}-CustomVoice",
    "clone": f"Qwen/Qwen3-TTS-12Hz-{SIZE}-Base",
}

LANGUAGES = ["Auto", "Korean", "English", "Japanese", "Chinese"]
SPEAKERS = ["Sohee", "Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna"]
EMOTIONS = ["기본", "기쁨", "슬픔", "분노", "당황", "두려움", "속삭임", "수줍음"]

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "outputs"
CHAR_DIR = ROOT / "characters"  # characters/<캐릭터>/<감정>.wav + <감정>.txt(참조 대본)
OUT_DIR.mkdir(exist_ok=True)
CHAR_DIR.mkdir(exist_ok=True)

_models: dict[str, Qwen3TTSModel] = {}
_clone_prompts: dict[tuple[str, float], list] = {}


def get_model(kind: str) -> Qwen3TTSModel:
    if kind not in _models:
        _models[kind] = Qwen3TTSModel.from_pretrained(MODEL_IDS[kind], device_map=DEVICE, dtype=DTYPE)
    return _models[kind]


def lang(value: str):
    return None if value == "Auto" else value


def save(kind: str, wavs, sr):
    path = OUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{kind}.wav"
    sf.write(path, wavs[0], sr)
    return str(path)


def timed(fn):
    def wrapper(*args):
        start = time.time()
        path = fn(*args)
        return path, f"완료: {time.time() - start:.1f}초, 저장 위치: {path}"

    return wrapper


@timed
def run_design(text, language, instruct):
    if not instruct.strip():
        raise gr.Error("목소리 설명을 입력해 주세요.")
    wavs, sr = get_model("design").generate_voice_design(text=text, instruct=instruct, language=lang(language))
    return save("design", wavs, sr)


@timed
def run_custom(text, language, speaker, instruct):
    wavs, sr = get_model("custom").generate_custom_voice(
        text=text, speaker=speaker, language=lang(language), instruct=instruct.strip() or None
    )
    return save(f"custom_{speaker}", wavs, sr)


@timed
def run_clone(text, language, ref_audio, ref_text):
    if ref_audio is None:
        raise gr.Error("참조 음성을 올려 주세요.")
    ref_text = ref_text.strip()
    wavs, sr = get_model("clone").generate_voice_clone(
        text=text,
        language=lang(language),
        ref_audio=ref_audio,
        ref_text=ref_text or None,
        x_vector_only_mode=not ref_text,  # 대본이 없으면 화자 임베딩만 사용 (품질은 다소 낮음)
    )
    return save("clone", wavs, sr)


# ---------- 캐릭터 ----------


def check_name(value, label):
    value = (value or "").strip()
    if not value or value.startswith(".") or "/" in value or "\\" in value:
        raise gr.Error(f"{label} 이름을 확인해 주세요. (빈 값, '/', '\\', '.'으로 시작하는 이름은 안 돼요)")
    return value


def list_characters():
    return sorted(p.name for p in CHAR_DIR.iterdir() if p.is_dir())


def list_emotions(character):
    if not character or not (CHAR_DIR / character).is_dir():
        return []
    return sorted(p.stem for p in (CHAR_DIR / character).glob("*.wav"))


def character_summary():
    rows = [f"- **{c}**: {', '.join(list_emotions(c)) or '(없음)'}" for c in list_characters()]
    return "\n".join(rows) or "저장된 캐릭터가 없어요. 다른 탭에서 마음에 드는 결과를 캐릭터로 저장해 보세요."


def emotion_update(character):
    emotions = list_emotions(character)
    return gr.update(choices=emotions, value=emotions[0] if emotions else None)


def save_to_character(audio_path, text, character, emotion, selected):
    if not audio_path:
        raise gr.Error("먼저 음성을 생성해 주세요.")
    if not text.strip():
        raise gr.Error("참조 대본(읽은 텍스트)이 비어 있어요.")
    character, emotion = check_name(character, "캐릭터"), check_name(emotion, "감정")
    folder = CHAR_DIR / character
    folder.mkdir(exist_ok=True)
    shutil.copy(audio_path, folder / f"{emotion}.wav")
    (folder / f"{emotion}.txt").write_text(text.strip(), encoding="utf-8")
    chars = gr.update(choices=list_characters())
    return (
        f"저장했어요: {character} / {emotion}",
        *[chars] * len(char_dropdowns),
        emotion_update(selected),
        character_summary(),
    )


def clone_prompt(character, emotion):
    wav = CHAR_DIR / character / f"{emotion}.wav"
    key = (str(wav), wav.stat().st_mtime)  # 같은 감정을 덮어쓰면 다시 만듦
    if key not in _clone_prompts:
        ref_text = (CHAR_DIR / character / f"{emotion}.txt").read_text(encoding="utf-8")
        _clone_prompts[key] = get_model("clone").create_voice_clone_prompt(ref_audio=str(wav), ref_text=ref_text)
    return _clone_prompts[key]


@timed
def run_character(text, language, character, emotion):
    if not character or not emotion:
        raise gr.Error("캐릭터와 감정을 골라 주세요.")
    wavs, sr = get_model("clone").generate_voice_clone(
        text=text, language=lang(language), voice_clone_prompt=clone_prompt(character, emotion)
    )
    return save(f"char_{character}_{emotion}", wavs, sr)


# ---------- UI ----------


def text_box(default):
    return gr.Textbox(label="읽을 텍스트", value=default, lines=3)


def lang_box():
    return gr.Dropdown(LANGUAGES, value="Auto", label="언어")


char_dropdowns = []  # 저장 후 선택지를 갱신할 캐릭터 드롭다운들
save_rows = []  # (결과 오디오, 읽은 텍스트, 캐릭터, 감정, 버튼, 상태)


def save_row(audio, text):
    with gr.Accordion("이 결과를 캐릭터로 저장", open=False):
        gr.Markdown("결과 음성과 '읽을 텍스트'가 참조 음성/대본으로 저장돼요. 같은 감정으로 다시 저장하면 덮어써요.")
        with gr.Row():
            character = gr.Dropdown(list_characters(), label="캐릭터 (새 이름 입력 가능)", allow_custom_value=True)
            emotion = gr.Dropdown(EMOTIONS, value="기본", label="감정 (직접 입력 가능)", allow_custom_value=True)
        btn = gr.Button("캐릭터로 저장")
        status = gr.Markdown()
    char_dropdowns.append(character)
    save_rows.append((audio, text, character, emotion, btn, status))


with gr.Blocks(title="TTS Lab") as demo:
    gr.Markdown(f"# TTS Lab\nQwen3-TTS · device: `{DEVICE}` · size: `{SIZE}` · 첫 실행 시 모델 다운로드/로드로 몇 분 걸릴 수 있어요.")

    with gr.Tab("목소리 디자인"):
        text = text_box("오늘 밤, 이 도시에서 가장 위험한 사람은 바로 나다.")
        language = lang_box()
        instruct = gr.Textbox(
            label="목소리 설명",
            value="낮고 거친 목소리의 30대 남성, 차분하지만 위협적인 말투, 느린 속도",
            lines=2,
        )
        btn = gr.Button("생성", variant="primary")
        audio, status = gr.Audio(label="결과", type="filepath"), gr.Markdown()
        btn.click(run_design, [text, language, instruct], [audio, status])
        save_row(audio, text)

    with gr.Tab("프리셋 화자 + 톤"):
        text = text_box("정말요? 저 진짜 합격한 거예요?")
        language = lang_box()
        speaker = gr.Dropdown(SPEAKERS, value="Sohee", label="화자 (Sohee: 한국어, Ryan/Aiden: 영어, Ono_Anna: 일본어)")
        instruct = gr.Textbox(label="감정·톤 지시 (선택)", value="믿기지 않는다는 듯 들뜨고 기쁜 목소리로", lines=2)
        btn = gr.Button("생성", variant="primary")
        audio, status = gr.Audio(label="결과", type="filepath"), gr.Markdown()
        btn.click(run_custom, [text, language, speaker, instruct], [audio, status])
        save_row(audio, text)

    with gr.Tab("보이스 클로닝"):
        gr.Markdown("본인 목소리나 사용 동의를 받은 목소리만 쓰세요. 3~10초 정도의 깨끗한 음성이 좋아요.")
        text = text_box("이건 제 목소리를 복제해서 만든 문장이에요.")
        language = lang_box()
        ref_audio = gr.Audio(label="참조 음성 (업로드 또는 녹음)", type="filepath", sources=["upload", "microphone"])
        ref_text = gr.Textbox(label="참조 음성의 대본 (입력하면 품질이 더 좋아요)", lines=2)
        btn = gr.Button("생성", variant="primary")
        audio, status = gr.Audio(label="결과", type="filepath"), gr.Markdown()
        btn.click(run_clone, [text, language, ref_audio, ref_text], [audio, status])
        save_row(audio, text)

    with gr.Tab("캐릭터"):
        gr.Markdown(
            "저장한 캐릭터의 감정별 참조 음성으로 클로닝해서, 같은 목소리를 유지하며 감정만 바꿔요. "
            "감정 표현은 참조 음성을 따라가므로 감정별로 참조 음성을 따로 저장해 두세요."
        )
        summary = gr.Markdown(character_summary())
        with gr.Row():
            characters = list_characters()
            char_select = gr.Dropdown(characters, value=characters[0] if characters else None, label="캐릭터")
            emotions = list_emotions(char_select.value)
            emotion_select = gr.Dropdown(emotions, value=emotions[0] if emotions else None, label="감정")
        text = text_box("에?! 잠깐만요, 잠깐만요! 이거, 제 거 아니에요!")
        language = lang_box()
        btn = gr.Button("생성", variant="primary")
        audio, status = gr.Audio(label="결과", type="filepath"), gr.Markdown()
        char_select.change(emotion_update, char_select, emotion_select)
        btn.click(run_character, [text, language, char_select, emotion_select], [audio, status])
        char_dropdowns.append(char_select)

    for audio, text, character, emotion, btn, status in save_rows:
        btn.click(
            save_to_character,
            [audio, text, character, emotion, char_select],
            [status, *char_dropdowns, emotion_select, summary],
        )
    # 새로고침 시 다른 탭에서 저장한 캐릭터도 보이도록
    demo.load(lambda: (gr.update(choices=list_characters()), character_summary()), None, [char_select, summary])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
