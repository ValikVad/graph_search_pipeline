"""
run_llm_eval.py
---------------
Эксперимент: может ли LLM правильно локализовать код в репозитории
(а) без инструментов — только по памяти,
(б) с инструментом GraphSearch (tool use через Anthropic API).

Метрика: доля вопросов, где LLM назвал правильный файл в ответе.

Использование:
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_llm_eval.py --repo karpathy_nanochat
    python run_llm_eval.py --repo vllm-project_vllm --split hard --limit 20
    python run_llm_eval.py --all-repos --split hard
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic

# Добавляем корень проекта в путь для импорта pipeline
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import CodeSearchPipeline, PipelineConfig

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------
AST_CACHE_DIR      = Path("ast_cache")
QUESTIONS_DIR      = Path("questions")
QUESTIONS_HARD_DIR = Path("questions_hard")
OUT_DIR            = Path("llm_eval_reports")

MODEL = "claude-sonnet-4-5"   # можно заменить на claude-opus-4-5

# Описание инструментов для tool use
TOOLS = [
    {
        "name": "search_code",
        "description": (
            "Search for relevant code snippets in a repository by developer intent. "
            "Returns the top matching functions with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Developer intent in natural language.",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name, e.g. 'vllm-project_vllm'.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results (default 5).",
                    "default": 5,
                },
            },
            "required": ["query", "repo"],
        },
    }
]

# ---------------------------------------------------------------------------
# Pipeline (ленивая загрузка)
# ---------------------------------------------------------------------------
_pipeline_cache: dict[str, CodeSearchPipeline] = {}


def _get_pipeline(repo: str) -> CodeSearchPipeline:
    if repo not in _pipeline_cache:
        cfg = PipelineConfig(top_n=20, hop_depth=1, top_k=5)
        p = CodeSearchPipeline(cfg, AST_CACHE_DIR)
        p.load([repo])
        _pipeline_cache[repo] = p
    return _pipeline_cache[repo]


def call_search_tool(query: str, repo: str, top_k: int = 5) -> str:
    """Выполняет реальный поиск и возвращает JSON-строку для LLM."""
    pipeline = _get_pipeline(repo)

    original_k = pipeline.config.top_k
    pipeline.config.top_k = max(1, min(top_k, 10))
    try:
        results = pipeline.search(query)
    finally:
        pipeline.config.top_k = original_k

    formatted = [
        {
            "rank":           i + 1,
            "file":           s["file"],
            "function":       s["qualified_name"],
            "lines":          f"{s['start_line']}-{s['end_line']}",
            "code_preview":   s["context"][:600],
        }
        for i, (s, _) in enumerate(results)
    ]
    return json.dumps({"results": formatted}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade(response_text: str, gt_file: str) -> bool:
    """
    Ответ засчитывается как правильный если LLM упомянул правильный файл.
    Проверяем несколько форматов: полный путь, basename, без расширения.
    """
    text_lower = response_text.lower()
    file_lower = gt_file.lower()

    checks = [
        file_lower,                            # полный путь
        Path(file_lower).name,                 # только имя файла
        Path(file_lower).stem,                 # без расширения
        file_lower.replace("/", ".").rstrip(".py"),  # пакетная нотация
    ]
    return any(c in text_lower for c in checks)


# ---------------------------------------------------------------------------
# LLM без инструментов
# ---------------------------------------------------------------------------

def ask_without_tools(client: anthropic.Anthropic,
                       question: str, repo: str) -> str:
    prompt = (
        f"You are a software engineer analyzing the '{repo}' repository.\n\n"
        f"Question: {question}\n\n"
        "Answer concisely. Specify the exact file path and function name "
        "where this is implemented."
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# LLM с инструментом GraphSearch
# ---------------------------------------------------------------------------

def ask_with_tools(client: anthropic.Anthropic,
                   question: str, repo: str) -> tuple[str, int]:
    """
    Запускает agentic loop: LLM может вызывать search_code сколько угодно раз.
    Возвращает (финальный текст ответа, кол-во вызовов инструмента).
    """
    system = (
        "You are a software engineer with access to a code search tool. "
        f"You are analyzing the '{repo}' repository. "
        "Use the search_code tool to find the relevant code, then answer "
        "with the exact file path and function name."
    )
    messages = [{"role": "user", "content": question}]
    tool_calls = 0

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        # Если LLM хочет вызвать инструмент
        if response.stop_reason == "tool_use":
            # Добавляем ответ ассистента в историю
            messages.append({"role": "assistant", "content": response.content})

            # Обрабатываем каждый tool_use блок
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_calls += 1
                args = block.input

                if block.name == "search_code":
                    result = call_search_tool(
                        query=args["query"],
                        repo=args.get("repo", repo),
                        top_k=args.get("top_k", 5),
                    )
                else:
                    result = json.dumps({"error": f"Unknown tool: {block.name}"})

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                })

            messages.append({"role": "user", "content": tool_results})

        else:
            # LLM закончил — извлекаем финальный текстовый ответ
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            return final_text, tool_calls


# ---------------------------------------------------------------------------
# Основная функция оценки
# ---------------------------------------------------------------------------

def evaluate_repo(client: anthropic.Anthropic,
                  repo: str,
                  split: str,
                  limit: int | None) -> dict:
    qdir  = QUESTIONS_HARD_DIR if split == "hard" else QUESTIONS_DIR
    qpath = qdir / f"{repo}.json"

    if not qpath.exists():
        print(f"  [skip] {qpath} not found")
        return {}

    questions = json.loads(qpath.read_text())
    if limit:
        questions = questions[:limit]

    print(f"\n{'─'*60}")
    print(f"  Repo: {repo}  |  Split: {split}  |  N: {len(questions)}")
    print(f"{'─'*60}")

    records = []
    correct_no_tool = 0
    correct_with_tool = 0

    for i, q in enumerate(questions):
        gt_file = q["file"]
        question = q["question"]
        print(f"\n  [{i+1}/{len(questions)}] {question[:70]}…")
        print(f"    GT: {gt_file}  L{q['line_start']}-{q['line_end']}")

        # ── Без инструмента ────────────────────────────────────────────────
        try:
            ans_no_tool = ask_without_tools(client, question, repo)
            ok_no_tool  = grade(ans_no_tool, gt_file)
        except Exception as e:
            ans_no_tool = f"ERROR: {e}"
            ok_no_tool  = False
        time.sleep(0.5)   # rate limit

        # ── С инструментом ─────────────────────────────────────────────────
        try:
            ans_with_tool, n_tool_calls = ask_with_tools(client, question, repo)
            ok_with_tool = grade(ans_with_tool, gt_file)
        except Exception as e:
            ans_with_tool  = f"ERROR: {e}"
            n_tool_calls   = 0
            ok_with_tool   = False
        time.sleep(0.5)

        status_no   = "✓" if ok_no_tool   else "✗"
        status_with = "✓" if ok_with_tool else "✗"
        print(f"    no-tool  {status_no}  |  with-tool {status_with}"
              f"  (search called {n_tool_calls}x)")

        if ok_no_tool:
            correct_no_tool += 1
        if ok_with_tool:
            correct_with_tool += 1

        records.append({
            "idx":            i,
            "question":       question,
            "gt_file":        gt_file,
            "gt_start":       q["line_start"],
            "gt_end":         q["line_end"],
            "ans_no_tool":    ans_no_tool,
            "ans_with_tool":  ans_with_tool,
            "n_tool_calls":   n_tool_calls,
            "ok_no_tool":     ok_no_tool,
            "ok_with_tool":   ok_with_tool,
        })

    n = len(questions)
    result = {
        "repo":          repo,
        "split":         split,
        "n":             n,
        "no_tool_acc":   round(correct_no_tool  / n, 4),
        "with_tool_acc": round(correct_with_tool / n, 4),
        "delta":         round((correct_with_tool - correct_no_tool) / n, 4),
        "records":       records,
    }

    print(f"\n  ── Result ───────────────────────────────────────")
    print(f"  Without tool : {correct_no_tool}/{n} = {result['no_tool_acc']:.1%}")
    print(f"  With tool    : {correct_with_tool}/{n} = {result['with_tool_acc']:.1%}")
    print(f"  Delta        : {result['delta']:+.1%}")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo",      help="Single repo name")
    ap.add_argument("--all-repos", action="store_true",
                    help="Run on all available repos")
    ap.add_argument("--split",     default="hard",
                    choices=["easy", "hard"],
                    help="Question split (default: hard)")
    ap.add_argument("--limit",     type=int, default=None,
                    help="Max questions per repo (default: all)")
    ap.add_argument("--model",     default=MODEL,
                    help=f"Claude model (default: {MODEL})")
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    global MODEL
    MODEL = args.model

    client = anthropic.Anthropic(api_key=api_key)
    OUT_DIR.mkdir(exist_ok=True)

    # Определяем список репо
    if args.all_repos:
        repos = sorted(
            d.name for d in AST_CACHE_DIR.iterdir()
            if d.is_dir() and (d / "snippets.json").exists()
        )
    elif args.repo:
        repos = [args.repo]
    else:
        print("ERROR: specify --repo REPO_NAME or --all-repos")
        sys.exit(1)

    all_results = []
    for repo in repos:
        result = evaluate_repo(client, repo, args.split, args.limit)
        if not result:
            continue
        all_results.append(result)

        # Сохраняем сразу после каждого репо
        out_path = OUT_DIR / f"{repo}_{args.split}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"  Saved → {out_path}")

    # Итоговая таблица
    if all_results:
        print(f"\n{'═'*60}")
        print("  FINAL SUMMARY")
        print(f"{'═'*60}")
        print(f"  {'Repo':<35} {'No tool':>9} {'With tool':>10} {'Delta':>7}")
        print(f"  {'─'*35} {'─'*9} {'─'*10} {'─'*7}")
        for r in all_results:
            print(f"  {r['repo']:<35} "
                  f"{r['no_tool_acc']:>8.1%}  "
                  f"{r['with_tool_acc']:>9.1%}  "
                  f"{r['delta']:>+7.1%}")

        # Сохраняем общий summary
        summary_path = OUT_DIR / f"summary_{args.split}.json"
        summary_path.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False)
        )
        print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
