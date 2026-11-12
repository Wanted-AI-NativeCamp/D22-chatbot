from flask import Flask, render_template, request

from rag import add_sources, ask_model, build_messages, load_index, search

MAX_QUESTION_LEN = 300
PREVIEW_LEN = 150

app = Flask(__name__)
INDEX = load_index()  # 앱이 켜질 때 한 번만 불러온다 (요청마다 다시 읽지 않음)


@app.route("/", methods=["GET", "POST"])
def ask():
    if request.method == "GET":
        return render_template("ask.html", question="")

    question = request.form.get("question", "").strip()

    if not question:
        return render_template("ask.html", question="", error="질문을 입력해 주세요.")
    if len(question) > MAX_QUESTION_LEN:
        error = (f"질문이 너무 길어요. {MAX_QUESTION_LEN}자 이내로 줄여 주세요. "
                 f"(지금 {len(question)}자)")
        return render_template("ask.html", question=question, error=error)

    try:
        results = search(question, INDEX, top_k=3)
        answer = add_sources(ask_model(build_messages(question, results)), results)
    except Exception:
        app.logger.exception("답 만들기 실패")
        return render_template(
            "ask.html", question=question,
            error="답을 가져오지 못했어요. 잠시 후 다시 시도해 주세요.")

    chunks = [
        {
            "rank": i,
            "source": chunk["source"],
            "score": f"{score:.2f}",
            "preview": chunk["text"][:PREVIEW_LEN].replace("\n", " "),
            "more": len(chunk["text"]) > PREVIEW_LEN,
        }
        for i, (score, chunk) in enumerate(results, 1)
    ]
    return render_template("ask.html", question=question, answer=answer, chunks=chunks)


if __name__ == "__main__":
    # debug 끄기: 켜면 재시작기가 앱을 두 번 띄워 JSON을 두 번 읽는다
    app.run(host="127.0.0.1", port=5000, debug=False)
