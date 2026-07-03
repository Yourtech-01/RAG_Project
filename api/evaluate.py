"""
evaluate.py
Evaluates RAG quality using Gemini as the judge (free, no OpenAI needed).

Metrics measured:
  - faithfulness:     does the answer stick to retrieved context?
  - answer_relevancy: is the answer relevant to the question?
  - context_quality:  are retrieved chunks useful?

Usage:
    python evaluate.py

Outputs:
    eval_results/ragas_scores.json
    eval_results/ragas_report.html
"""

import json
import pathlib
import requests
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
judge = genai.GenerativeModel("gemini-flash-latest")

# ── Test set — update these after ingesting your own documents ────────────────
TEST_QUESTIONS = [
    "What is the main purpose described in the document?",
    "What are the key conclusions or findings?",
    "What methodology was used?",
    "What are the limitations mentioned?",
    "What recommendations are made?",
]

OUTPUT_DIR = pathlib.Path("eval_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")


def score_faithfulness(question: str, answer: str, context: str) -> float:
    """Ask Gemini to judge if the answer is grounded in the context."""
    prompt = f"""You are an evaluation judge. Score the FAITHFULNESS of the answer.
Faithfulness = does the answer ONLY use information from the context, with no hallucination?

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Score from 0.0 (completely hallucinated) to 1.0 (perfectly faithful to context).
Reply with ONLY a number like 0.85"""
    try:
        resp = judge.generate_content(prompt)
        return float(resp.text.strip())
    except Exception:
        return 0.0


def score_relevancy(question: str, answer: str) -> float:
    """Ask Gemini to judge if the answer is relevant to the question."""
    prompt = f"""You are an evaluation judge. Score the RELEVANCY of the answer.
Relevancy = does the answer directly address what was asked?

QUESTION: {question}

ANSWER: {answer}

Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).
Reply with ONLY a number like 0.85"""
    try:
        resp = judge.generate_content(prompt)
        return float(resp.text.strip())
    except Exception:
        return 0.0


def run_evaluation():
    print("[eval] Starting RAG evaluation...")
    print(f"[eval] Evaluating {len(TEST_QUESTIONS)} questions")
    print(f"[eval] API: {API_URL}\n")

    results = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"  [{i}/{len(TEST_QUESTIONS)}] {question[:60]}...")
        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "top_k": 5},
                timeout=30)
            data     = resp.json()
            answer   = data.get("answer", "")
            sources  = data.get("sources", [])
            context  = "\n\n".join(s["excerpt"] for s in sources)

            faith = score_faithfulness(question, answer, context)
            relev = score_relevancy(question, answer)

            results.append({
                "question":    question,
                "answer":      answer[:300],
                "faithfulness": faith,
                "relevancy":   relev,
                "n_sources":   len(sources),
            })
            print(f"     faithfulness={faith:.2f}  relevancy={relev:.2f}  sources={len(sources)}")

        except Exception as e:
            print(f"     ERROR: {e}")
            results.append({
                "question": question, "answer": "",
                "faithfulness": 0.0, "relevancy": 0.0, "n_sources": 0})

    # Aggregate scores
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_relev = sum(r["relevancy"]    for r in results) / len(results)

    summary = {
        "avg_faithfulness": round(avg_faith, 4),
        "avg_relevancy":    round(avg_relev, 4),
        "n_questions":      len(results),
        "details":          results,
    }

    print(f"\n=== Evaluation Results ===")
    print(f"  Avg Faithfulness : {avg_faith:.4f}")
    print(f"  Avg Relevancy    : {avg_relev:.4f}")
    print(f"  Questions tested : {len(results)}")

    # Save JSON
    out_json = OUTPUT_DIR / "ragas_scores.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval] Scores saved to {out_json}")

    # Save HTML report
    rows = ""
    for r in results:
        rows += f"""<tr>
            <td>{r['question'][:80]}</td>
            <td>{r['faithfulness']:.2f}</td>
            <td>{r['relevancy']:.2f}</td>
            <td>{r['n_sources']}</td>
            <td>{r['answer'][:150]}...</td>
        </tr>"""

    html = f"""<!DOCTYPE html><html><head><title>RAG Evaluation</title>
<style>
  body{{font-family:sans-serif;padding:20px;}}
  table{{border-collapse:collapse;width:100%;}}
  th,td{{border:1px solid #ddd;padding:8px;text-align:left;font-size:13px;}}
  th{{background:#4361ee;color:white;}}
  tr:nth-child(even){{background:#f8f9fa;}}
  .summary{{background:#e8f4fd;padding:16px;border-radius:8px;margin-bottom:20px;}}
</style></head><body>
<h1>RAG Evaluation Report</h1>
<div class="summary">
  <strong>Avg Faithfulness:</strong> {avg_faith:.4f} &nbsp;|&nbsp;
  <strong>Avg Relevancy:</strong> {avg_relev:.4f} &nbsp;|&nbsp;
  <strong>Questions:</strong> {len(results)}
</div>
<table><tr>
  <th>Question</th><th>Faithfulness</th><th>Relevancy</th>
  <th>Sources</th><th>Answer Preview</th>
</tr>{rows}</table>
</body></html>"""

    out_html = OUTPUT_DIR / "ragas_report.html"
    with open(out_html, "w") as f:
        f.write(html)
    print(f"[eval] HTML report saved to {out_html}")

    return summary


if __name__ == "__main__":
    run_evaluation()
