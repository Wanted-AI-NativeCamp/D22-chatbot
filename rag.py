import json
import os
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

DOCS_DIR = Path(__file__).parent / "docs"
INDEX_PATH = Path(__file__).parent / "index.json"
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def load_docs(docs_dir=DOCS_DIR):
    """docs 폴더의 .md 를 모두 읽는다. 출처 = 파일 이름."""
    return [
        {"source": p.name, "text": p.read_text(encoding="utf-8")}
        for p in sorted(docs_dir.glob("*.md"))
    ]


def split_docs(docs, chunk_size, chunk_overlap=0):
    """문서들을 chunk_size 글자 단위 조각으로 자른다. 조각마다 출처를 붙인다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = []
    for d in docs:
        for i, text in enumerate(splitter.split_text(d["text"])):
            chunks.append({"source": d["source"], "index": i, "text": text})
    return chunks


def count_tokens(texts):
    """대략 토큰 수. 임베딩 모델이 쓰는 cl100k_base 방식으로 센다."""
    enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(t)) for t in texts)


def embed_texts(texts, batch_size=100):
    """글자 목록을 숫자 벡터 목록으로 바꾼다."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    vectors = []
    for i in range(0, len(texts), batch_size):
        res = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch_size])
        vectors.extend(d.embedding for d in res.data)
    return vectors


def load_index(path=INDEX_PATH):
    """index.json 을 불러온다."""
    return json.loads(path.read_text(encoding="utf-8"))


def cosine(a, b):
    """코사인 유사도. 1에 가까울수록 비슷하고, 0이면 무관, 음수면 반대 방향."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


def search(question, index, top_k=3):
    """질문과 가장 비슷한 조각 top_k 개를 (점수, 조각) 목록으로 돌려준다."""
    q_vec = embed_texts([question])[0]
    scored = [(cosine(q_vec, c["embedding"]), c) for c in index["chunks"]]
    scored.sort(key=lambda s: s[0], reverse=True)
    return scored[:top_k]


SYSTEM_PROMPT = """너는 '클래스로 마켓' 고객센터 안내 담당이야.
아래 [자료]만 보고 질문에 답해.

규칙:
1. [자료] 안에 있는 내용만으로 답해. 원래 알던 지식은 쓰지 마.
2. [자료]에 답이 없으면 다른 말 없이 이 문장만 써: "문서에서 찾지 못했어요."
3. 숫자, 기간, 조건은 [자료]에 적힌 그대로 써. 바꾸거나 줄이거나 추측하지 마.
4. 출처(파일 이름)는 쓰지 마. 출처는 프로그램이 따로 붙여."""

NOT_FOUND = "문서에서 찾지 못했어요."


def build_messages(question, results):
    """검색 결과(점수, 조각)로 모델에 보낼 메시지를 만든다."""
    material = "\n\n---\n\n".join(chunk["text"] for _, chunk in results)
    user = f"[자료]\n{material}\n\n[질문]\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def ask_model(messages):
    """모델을 불러 답을 받는다. (이 모델은 temperature 0을 지원하지 않아 기본값 사용)"""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    res = client.chat.completions.create(
        model=os.environ["OPENAI_MODEL"], messages=messages
    )
    return res.choices[0].message.content


def add_sources(answer, results):
    """답 끝에 출처를 코드가 붙인다. 모델이 쓴 '출처:' 줄은 지운다.
    '문서에서 찾지 못했어요.' 이면 출처를 붙이지 않는다."""
    if NOT_FOUND in answer:
        return NOT_FOUND
    lines = [ln for ln in answer.strip().splitlines() if not ln.strip().startswith("출처")]
    sources = list(dict.fromkeys(chunk["source"] for _, chunk in results))  # 순위 순, 중복 제거
    return "\n".join(lines).rstrip() + "\n\n출처: " + ", ".join(sources)


def save_index(chunks, vectors, chunk_size, chunk_overlap, path=INDEX_PATH):
    """조각 + 벡터를 index.json 으로 저장한다."""
    data = {
        "model": EMBED_MODEL,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunks": [{**c, "embedding": v} for c, v in zip(chunks, vectors)],
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
