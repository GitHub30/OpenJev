"""Calibrated-decision fine-tuning (LoRA) for OpenJev.

TypeSafe trains Jev with "Reinforcement Learning for Calibrated Decisions". The
open-weight recipe here is the supervised core of that idea: because every
answer is a distribution over a closed candidate set, we can optimise a *proper
scoring rule* (negative log-likelihood of the gold outcome, optionally mixed
with the Brier score) directly on the candidate distribution. Proper scoring
rules are minimised only by honest probabilities, so training on them pushes
the model toward both accuracy and calibration at once, with no text generation
anywhere in the loop.

Data format (JSONL), one request per line:

    {"state": "...", "questions": {"team": {"type": "choice", "criteria": {...}}, ...},
     "gold": {"team": "billing", "urgent": true, "severity": 2}}

``gold`` values are: the option name (choice), the level index (score) or a
boolean (noul). Questions without gold are ignored.

Example:

    python scripts/train_calibrated.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --data examples/triage.jsonl --output checkpoints/qwen-openjev-lora --epochs 2

Then serve with ``openjev serve --model Qwen/Qwen2.5-1.5B-Instruct --adapter checkpoints/qwen-openjev-lora``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from openjev.backends.hf import HFPromptRenderer
from openjev.prompting import PromptConfig, compile_request
from openjev.schema import SystemOneRequest


@dataclass
class Example:
    prefix: list[int]
    candidates: list[list[int]]
    gold: int
    kind: str


def gold_index(kind: str, outcomes: list[str], gold) -> int:
    if kind == "noul":
        return 0 if bool(gold) else 1
    return outcomes.index(str(gold))


def load_examples(path: str, renderer: HFPromptRenderer, prompt_config: PromptConfig, max_len: int) -> list[Example]:
    examples: list[Example] = []
    skipped = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            request = SystemOneRequest.model_validate({"state": row["state"], "questions": row["questions"]})
            for decision in compile_request(request.state, request.questions, prompt_config):
                if decision.name not in row.get("gold", {}):
                    continue
                prefix = renderer.render_prompt(decision.prompt)
                cands = renderer.candidate_ids(decision.candidates)
                if len(prefix) + max(len(c) for c in cands) > max_len:
                    skipped += 1
                    continue
                examples.append(Example(
                    prefix=prefix,
                    candidates=cands,
                    gold=gold_index(decision.kind, decision.outcomes, row["gold"][decision.name]),
                    kind=decision.kind,
                ))
    if skipped:
        print(f"skipped {skipped} examples longer than --max-len {max_len}")
    return examples


def candidate_logprobs(model, example: Example, pad_id: int, device) -> torch.Tensor:
    """Sum of token log-probs for every candidate, computed in one batched forward."""
    rows = [example.prefix + c for c in example.candidates]
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    ids, mask = ids.to(device), mask.to(device)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    logp = logits.log_softmax(-1)
    start = len(example.prefix) - 1
    out = []
    for i, c in enumerate(example.candidates):
        positions = torch.arange(start, start + len(c), device=device)
        targets = torch.tensor(c, device=device)
        out.append(logp[i, positions, targets].sum())
    return torch.stack(out)


def decision_loss(cand_logprobs: torch.Tensor, gold: int, brier_weight: float, label_smoothing: float) -> torch.Tensor:
    """NLL over the closed candidate set, optionally mixed with the Brier score."""
    log_dist = cand_logprobs.log_softmax(-1)
    k = log_dist.numel()
    target = torch.full((k,), label_smoothing / k, device=log_dist.device)
    target[gold] += 1.0 - label_smoothing
    nll = -(target * log_dist).sum()
    if brier_weight <= 0:
        return nll
    onehot = F.one_hot(torch.tensor(gold, device=log_dist.device), k).float()
    brier = ((log_dist.exp() - onehot) ** 2).sum()
    return nll + brier_weight * brier


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-data", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--brier-weight", type=float, default=0.0, help="add w * Brier score to the NLL")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps (smoke tests)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.bfloat16 if device.type == "cuda" else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=args.trust_remote_code)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code)
    model.to(device)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    renderer = HFPromptRenderer(tokenizer)
    prompt_config = PromptConfig()
    train = load_examples(args.data, renderer, prompt_config, args.max_len)
    print(f"{len(train)} training decisions")
    evals = load_examples(args.eval_data, renderer, prompt_config, args.max_len) if args.eval_data else []

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / args.grad_accum)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: max(0.0, 1.0 - s / max(total_steps, 1)))

    step = 0
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(train)
        running = 0.0
        for i, ex in enumerate(train):
            lp = candidate_logprobs(model, ex, tokenizer.pad_token_id, device)
            loss = decision_loss(lp, ex.gold, args.brier_weight, args.label_smoothing) / args.grad_accum
            loss.backward()
            running += loss.item()
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == total_steps:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {running / min(10, step):.4f}")
                    running = 0.0
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break

    if evals:
        model.eval()
        correct = 0
        nll = 0.0
        with torch.inference_mode():
            for ex in evals:
                lp = candidate_logprobs(model, ex, tokenizer.pad_token_id, device).log_softmax(-1)
                correct += int(lp.argmax().item() == ex.gold)
                nll -= lp[ex.gold].item()
        print(f"eval: n={len(evals)} accuracy={correct / len(evals):.3f} nll={nll / len(evals):.4f}")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
