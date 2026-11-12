"""R1: 문서 없이 LLM에게 질문 3개를 그대로 물어본다 (RAG 없음)."""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 이 파일 옆 .env 를 우선 읽고, 없으면 상위 폴더에서 찾는다
load_dotenv(Path(__file__).parent / ".env")
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.environ["OPENAI_MODEL"]

QUESTIONS = [
    "클래스로 마켓 환불 규정은? 수업 3일 전에 취소하면 얼마나 돌려받아?",
    "클래스로 마켓에서 사진 리뷰를 쓰면 포인트를 얼마나 줘?",
    "클래스로 마켓 고객센터 전화번호 알려 줘.",
]

for i, q in enumerate(QUESTIONS, 1):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": q}],  # 시스템 프롬프트·문서 없음
    )
    print(f"[Q{i}] {q}")
    print(f"[A{i}] {res.choices[0].message.content}\n")
