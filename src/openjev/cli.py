"""Command line entry points: ``openjev serve``, ``openjev ask``, ``openjev calibrate``, ``openjev eval``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .backends import load_backend
from .calibration import Calibration
from .engine import SystemOneEngine
from .prompting import PromptConfig
from .schema import SystemOneRequest


def _backend_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if getattr(args, "device", None):
        kwargs["device"] = args.device
    if getattr(args, "dtype", None):
        kwargs["dtype"] = args.dtype
    if getattr(args, "device_map", None):
        kwargs["device_map"] = args.device_map
    if getattr(args, "trust_remote_code", False):
        kwargs["trust_remote_code"] = True
    if getattr(args, "max_batch_tokens", None):
        kwargs["max_batch_tokens"] = args.max_batch_tokens
    if getattr(args, "no_share_prefix", False):
        kwargs["share_prefix"] = False
    if getattr(args, "adapter", None):
        kwargs["adapter"] = args.adapter
    return kwargs


def build_engine(args: argparse.Namespace) -> SystemOneEngine:
    backend = load_backend(args.model, **_backend_kwargs(args))
    calibration = Calibration.load(args.calibration) if getattr(args, "calibration", None) else Calibration()
    prompt_config = PromptConfig(system_prompt=Path(args.system_prompt).read_text(encoding="utf-8")) if getattr(args, "system_prompt", None) else PromptConfig()
    return SystemOneEngine(
        backend=backend,
        model_name=args.name or f"openjev/{Path(args.model).name}",
        calibration=calibration,
        prompt_config=prompt_config,
    )


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", "-m", default="mock", help="HF model id / local path, or 'mock'")
    p.add_argument("--name", default=None, help="model name reported in responses")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, help="e.g. bfloat16, float16, float32")
    p.add_argument("--device-map", default=None, help="e.g. auto (multi-GPU)")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-batch-tokens", type=int, default=None)
    p.add_argument("--no-share-prefix", action="store_true", help="disable KV-cache prefix sharing")
    p.add_argument("--adapter", default=None, help="path to a LoRA adapter to merge into the model")
    p.add_argument("--calibration", default=None, help="path to calibration JSON")
    p.add_argument("--system-prompt", default=None, help="path to a custom system prompt")


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    engine = build_engine(args)
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _load_json_arg(value: str) -> Any:
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def cmd_ask(args: argparse.Namespace) -> int:
    engine = build_engine(args)
    if args.request:
        request = SystemOneRequest.model_validate(_load_json_arg(args.request))
    else:
        questions: dict[str, Any] = {}
        for spec in args.noul or []:
            name, _, text = spec.partition("=")
            questions[name] = {"type": "noul", "instructions": text}
        for spec in args.choice or []:
            name, _, rest = spec.partition("=")
            text, _, opts = rest.partition("|")
            questions[name] = {"type": "choice", "instructions": text, "criteria": {o.strip(): None for o in opts.split(",")}}
        for spec in args.score or []:
            name, _, rest = spec.partition("=")
            text, _, levels = rest.partition("|")
            questions[name] = {"type": "score", "instructions": text, "criteria": [lv.strip() for lv in levels.split(",")]}
        if not questions:
            print("no questions given (use --noul/--choice/--score or --request)", file=sys.stderr)
            return 2
        request = SystemOneRequest(state=_load_json_arg(args.state), questions=questions)
    response = engine.evaluate(request)
    print(response.model_dump_json(indent=2))
    return 0


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _gold_index(kind: str, outcomes: list[str], gold: Any) -> int:
    if kind == "noul":
        return 0 if bool(gold) else 1
    return outcomes.index(str(gold))


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit per-type temperatures on a JSONL of {state, questions, gold: {name: answer}} rows."""
    from .calibration import fit_temperature

    engine = build_engine(args)
    sets: dict[str, list[list[float]]] = {"noul": [], "choice": [], "score": []}
    gold: dict[str, list[int]] = {"noul": [], "choice": [], "score": []}
    for row in _iter_jsonl(args.data):
        request = SystemOneRequest.model_validate({"state": row["state"], "questions": row["questions"]})
        raws, _ = engine.evaluate_raw(request)
        for raw in raws:
            if raw.decision.name not in row.get("gold", {}):
                continue
            kind = raw.decision.kind
            sets[kind].append(raw.logprobs)
            gold[kind].append(_gold_index(kind, raw.decision.outcomes, row["gold"][raw.decision.name]))
    calibration = Calibration()
    for kind in sets:
        if sets[kind]:
            calibration.temperatures[kind] = fit_temperature(sets[kind], gold[kind])
            print(f"{kind}: n={len(sets[kind])} temperature={calibration.temperatures[kind]:.3f}")
    calibration.save(args.output)
    print(f"wrote {args.output}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Accuracy / NLL / ECE on a labelled JSONL (same format as calibrate)."""
    import math

    from .calibration import expected_calibration_error

    engine = build_engine(args)
    n = correct = 0
    nll = 0.0
    confs: list[float] = []
    oks: list[bool] = []
    for row in _iter_jsonl(args.data):
        request = SystemOneRequest.model_validate({"state": row["state"], "questions": row["questions"]})
        raws, _ = engine.evaluate_raw(request)
        for raw in raws:
            if raw.decision.name not in row.get("gold", {}):
                continue
            g = _gold_index(raw.decision.kind, raw.decision.outcomes, row["gold"][raw.decision.name])
            probs = raw.probabilities
            pred = max(range(len(probs)), key=probs.__getitem__)
            n += 1
            correct += int(pred == g)
            nll -= math.log(max(probs[g], 1e-12))
            confs.append(probs[pred])
            oks.append(pred == g)
    if n == 0:
        print("no labelled questions found")
        return 1
    print(json.dumps({
        "n": n,
        "accuracy": correct / n,
        "nll": nll / n,
        "ece": expected_calibration_error(confs, oks),
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openjev", description="Open-weight System One decision engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="run the /v1/systemone HTTP server")
    _add_model_args(p_serve)
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--log-level", default="info")
    p_serve.set_defaults(func=cmd_serve)

    p_ask = sub.add_parser("ask", help="answer questions about a state from the command line")
    _add_model_args(p_ask)
    p_ask.add_argument("--state", "-s", default="", help="state text, JSON, or @file.json")
    p_ask.add_argument("--noul", action="append", help="name=question")
    p_ask.add_argument("--choice", action="append", help="name=question|opt1,opt2,...")
    p_ask.add_argument("--score", action="append", help="name=question|level0,level1,...")
    p_ask.add_argument("--request", default=None, help="full request JSON or @file.json")
    p_ask.set_defaults(func=cmd_ask)

    p_cal = sub.add_parser("calibrate", help="fit temperatures on labelled JSONL")
    _add_model_args(p_cal)
    p_cal.add_argument("--data", required=True)
    p_cal.add_argument("--output", "-o", default="calibration.json")
    p_cal.set_defaults(func=cmd_calibrate)

    p_eval = sub.add_parser("eval", help="report accuracy / NLL / ECE on labelled JSONL")
    _add_model_args(p_eval)
    p_eval.add_argument("--data", required=True)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
