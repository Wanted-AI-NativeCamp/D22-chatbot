from rag import load_docs, split_docs

KEYWORD = "72시간 전까지"
SIZES = [100, 400, 1500]

docs = load_docs()

# 1) 문서별 글자 수
print("== 문서별 글자 수 ==")
for d in docs:
    print(f"{d['source']:<28} {len(d['text']):>6}자")
print(f"{'전체':<28} {sum(len(d['text']) for d in docs):>6}자\n")

# 2) 조각 크기별 조각 수 + 키워드가 든 조각
results = {}
for size in SIZES:
    results[size] = split_docs(docs, size)

print("== 조각 크기별 조각 수 ==")
for size in SIZES:
    print(f"chunk_size={size:<5} -> {len(results[size])}개")

for size in SIZES:
    print(f"\n== chunk_size={size}: '{KEYWORD}' 가 든 조각 ==")
    hits = [c for c in results[size] if KEYWORD in c["text"]]
    if not hits:
        print("(없음 - 경계에서 잘려 한 조각에 못 들어갔어요)")
    for c in hits:
        print(f"[{c['source']} #{c['index']}] {len(c['text'])}자")
        print(c["text"])
        print("-" * 40)
