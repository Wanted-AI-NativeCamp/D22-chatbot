import argparse

from rag import load_index, search

parser = argparse.ArgumentParser(description="index.json 에서 질문과 비슷한 조각 검색 (답 생성 없음)")
parser.add_argument("question", nargs="+", help='질문 (따옴표 있어도 없어도 됨)')
args = parser.parse_args()
question = " ".join(args.question)

index = load_index()
results = search(question, index, top_k=3)

print(f"질문: {question}\n")
for rank, (score, chunk) in enumerate(results, 1):
    preview = chunk["text"][:100].replace("\n", " ")
    print(f"{rank}위  점수 {score:.2f}  {chunk['source']}")
    print(f"    {preview}\n")
