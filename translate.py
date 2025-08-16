import os
from dotenv import load_dotenv

from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from yt_dlp import YoutubeDL  # ✅ 꼭 필요

load_dotenv()

# API 키 확인
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("OPENAI_API_KEY"):
    raise ValueError("API 키 환경 변수가 설정되지 않았습니다. GOOGLE_API_KEY와 OPENAI_API_KEY를 설정해주세요.")

openai_client = OpenAI()

def download_audio_from_youtube(video_url: str, outdir: str = "downloads", basename: str = "%(title)s.%(ext)s"):
    """
    yt-dlp로 bestaudio를 다운받고 FFmpeg로 m4a로 추출합니다.
    ⚠️ 시스템에 ffmpeg 가 설치되어 있어야 합니다.
    """
    os.makedirs(outdir, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(outdir, basename),
        "retries": 5,
        "fragment_retries": 5,
        "skip_unavailable_fragments": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",
            "preferredquality": "192",
        }],
        "nocheckcertificate": True,
        "http_headers": {"User-Agent": "Mozilla/5.0"},
        "quiet": True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            audio_path = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".m4a"
            return audio_path
    except Exception as e:
        raise RuntimeError(f"오디오 다운로드 실패: {e}")

def transcribe_audio_with_whisper(audio_path: str) -> str | None:
    """
    Whisper로 음성을 텍스트로 변환.
    - openai-python v1 계열 기준 예시
    """
    try:
        with open(audio_path, "rb") as f:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"  # 문자열 반환
            )
        # 일부 환경에선 객체가 올 수도 있으므로 방어적으로 문자열화
        if hasattr(transcript, "text"):
            return transcript.text
        return str(transcript)
    except Exception as e:
        print(f"오류: Whisper 변환 실패. {e}")
        return None

def analyze_video_with_gemini(video_url: str):
    """
    1) 유튜브에서 오디오 다운로드
    2) Whisper로 텍스트 변환
    3) Gemini로 요약/분석
    """
    print("1. YouTube 영상에서 오디오 다운로드 중...")
    audio_path = download_audio_from_youtube(video_url)
    if not audio_path:
        return "영상을 분석할 수 없습니다."

    print("2. 오디오를 텍스트로 변환 중 (Whisper 모델 사용)...")
    transcript = transcribe_audio_with_whisper(audio_path)

    # 임시 오디오 파일 정리
    try:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception:
        pass

    if not transcript or not transcript.strip():
        return "영상을 분석할 수 없습니다. 음성 변환에 실패했습니다."

    print("3. Gemini 모델로 분석 요청 중...")
    # 최신 모델 권장: "gemini-1.5-pro" / "gemini-1.5-flash"
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    prompt_template = """
        다음은 YouTube 동영상의 음성을 텍스트로 변환한 내용입니다.
        다음은 노인 운동 유튜브 영상의 제목, 설명, 스크립트(내용)입니다. 제공된 정보를 기반으로 아래에 정의된 분류 기준에 맞춰 영상을 분석하고, 적절한 분류 항목을 JSON 형식으로 추출해 주세요.

        [분류 기준]
        1. 운동 종류: 근력 운동, 유산소 운동, 유연성 운동, 균형 운동
        2. 운동 부위: 전신, 팔/어깨, 다리/하체, 허리/복부, 손목/발목
        3. 도구 사용: 맨몸 운동, 밴드 사용, 덤벨/물통 사용, 수건 사용, 기타 소도구 사용
        4. 운동 난이도: 초급, 중급, 상급
        5. 운동 시간: 10분 미만, 10-20분, 20분 초과
        6. 특정 대상/목적: 특정 질환/증상(예: 무릎 관절염, 허리 통증, 치매 예방), 재활, 일상생활 기능 향상, 정서적 안정/스트레스 해소, 와상 환자용, 휠체어 이용자용, 보행 가능자용
        7. 운동 환경: 실내, 야외, 의자 사용
        8. 강사 스타일: 활기찬, 차분한, 설명 위주, 기타

        [분석할 영상 정보]
        "{transcript}"

        분석 결과:
        """
        
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=prompt_template,
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(transcript=transcript)
    return response

# if __name__ == "__main__":
#     youtube_video_url = "https://www.youtube.com/watch?v=5ypSJRSq-t8"
#     analysis_result = analyze_video_with_gemini(youtube_video_url)
#     print("\n--- YouTube 영상 분석 결과 ---")
#     print(analysis_result)
