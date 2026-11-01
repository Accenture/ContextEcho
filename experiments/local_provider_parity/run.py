"""Compare the adapter with direct SDK calls against a real local endpoint."""

import argparse
import importlib.metadata
import json
from pathlib import Path

from harness.clients import TargetClient
from harness.clients_local import make_local_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter = TargetClient("local", args.model)
    direct = make_local_client()
    rows = []
    for case in json.loads(args.cases.read_text()):
        messages = [{"role": "system", "content": case["system"]}] + [
            {"role": message["role"], "content": "\n".join(block["text"] for block in message["content"])}
            for message in case["messages"]
        ]
        for budget in args.budgets:
            actual = adapter.step(case["system"], case["messages"], [], max_tokens=budget)
            reference = direct.chat.completions.create(
                model=args.model, messages=messages, max_tokens=budget, temperature=0.0,
            )
            observed = {"text": actual.text, "finish_reason": actual.stop_reason, **actual.raw_usage}
            expected = {"text": reference.choices[0].message.content or "",
                        "finish_reason": reference.choices[0].finish_reason,
                        "input_tokens": reference.usage.prompt_tokens,
                        "output_tokens": reference.usage.completion_tokens}
            rows.append({"case": case["id"], "max_tokens": budget, "adapter": observed,
                         "direct": expected, "equal": observed == expected,
                         "within_budget": observed["output_tokens"] <= budget})
            args.output.write_text(json.dumps({
                "model": args.model, "temperature": 0.0,
                "openai_version": importlib.metadata.version("openai"), "comparisons": rows,
            }, indent=2, ensure_ascii=False) + "\n")
            print(case["id"], budget, rows[-1]["equal"], flush=True)
    assert all(row["equal"] and row["within_budget"] for row in rows)


if __name__ == "__main__":
    main()
