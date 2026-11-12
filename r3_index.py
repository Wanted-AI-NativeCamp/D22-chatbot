from rag import (EMBED_MODEL, INDEX_PATH, count_tokens, embed_texts,
                 load_docs, save_index, split_docs)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 0

docs = load_docs()
chunks = split_docs(docs, CHUNK_SIZE, CHUNK_OVERLAP)
texts = [c["text"] for c in chunks]

# 1) 먼저 규모를 보여 준다
print(f"전체 조각 수      : {len(chunks)}개")
print(f"숫자로 바꿀 글자 수: {sum(len(t) for t in texts):,}자")
print(f"대략 토큰 수      : {count_tokens(texts):,}개")
print(f"임베딩 모델       : {EMBED_MODEL}\n")

# 2) 임베딩 → 저장
vectors = embed_texts(texts)
save_index(chunks, vectors, CHUNK_SIZE, CHUNK_OVERLAP)
print(f"벡터 차원         : {len(vectors[0])}")
print(f"저장 완료         : {INDEX_PATH.name} ({INDEX_PATH.stat().st_size / 1024:,.0f}KB)")
