from dotenv import load_dotenv

load_dotenv()

from core.pipeline import run_pipeline
from core.rag_engine import ask_question


def _print_stage(stage: str, state: str) -> None:
    print(f"[{state.upper()}] {stage}")


if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"

    try:
        result = run_pipeline(source, language, callback=_print_stage)
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")

    if result.get("errors"):
        print("\nPartial pipeline warnings:")
        for stage, error in result["errors"].items():
            print(f"- {stage}: {error}")
    print("=" * 60)

    rag_chain = result.get("rag_chain")
    if rag_chain is None:
        print("\nRAG chat is unavailable for this run.")
        raise SystemExit(0)

    print("\nChat with your meeting (type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\nAssistant: {answer}\n")
