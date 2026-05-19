import os

from dotenv import load_dotenv

load_dotenv()

from core.pipeline import run_pipeline


def print_stage(stage: str, state: str) -> None:
    print(f"[{state.upper()}] {stage}")


source = os.getenv("SMOKE_TEST_SOURCE") or input("Enter YouTube URL or local file path: ").strip()
language = os.getenv("SMOKE_TEST_LANGUAGE", "english")

result = run_pipeline(source, language=language, callback=print_stage)

print("\n" + "=" * 60)
print("TRANSCRIPT")
print("=" * 60)
transcript = result["transcript"]
print(transcript[:500] + "..." if len(transcript) > 500 else transcript)

print("\n" + "=" * 60)
print(f"TITLE: {result['title']}")
print("=" * 60)
print("\nSUMMARY")
print("-" * 60)
print(result["summary"])

print("\n" + "=" * 60)
print("ACTION ITEMS")
print("=" * 60)
print(result["action_items"])

print("\n" + "=" * 60)
print("KEY DECISIONS")
print("=" * 60)
print(result["key_decisions"])

print("\n" + "=" * 60)
print("OPEN QUESTIONS")
print("=" * 60)
print(result["open_questions"])

if result.get("errors"):
    print("\nWARNINGS")
    print("=" * 60)
    for stage, error in result["errors"].items():
        print(f"- {stage}: {error}")
