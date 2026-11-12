import argparse

from rag import add_sources, ask_model, build_messages, load_index, search

parser = argparse.ArgumentParser(description="검색 -> 프롬프트 -> 모델 호출로 답하기")
parser.add_argument("question", nargs="+", help="질문 (따옴표 있어도 없어도 됨)")
question = " ".join(parser.parse_args().question)

print(f"질문: {question}\n")

# 1단계: 검색
print("=" * 50)
print("1단계: 검색")
print("=" * 50)
results = search(question, load_index(), top_k=3)
for rank, (score, chunk) in enumerate(results, 1):
    print(f"{rank}위  점수 {score:.2f}  {chunk['source']}")

# 2단계: 프롬프트 만들기
print("\n" + "=" * 50)
print("2단계: 프롬프트 만들기 (모델에 보내는 내용 전문)")
print("=" * 50)
messages = build_messages(question, results)
for m in messages:
    print(f"--- {m['role']} ---")
    print(m["content"])
    print()

# 3단계: 모델 부르기
print("=" * 50)
print("3단계: 모델 부르기")
print("=" * 50)
answer = add_sources(ask_model(messages), results)  # 출처는 코드가 붙임

print("\n답:")
print(answer)
print("\n찾아온 조각:")
for rank, (score, chunk) in enumerate(results, 1):
    print(f"  {rank}위  점수 {score:.2f}  {chunk['source']}")
