# OpenJev

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GitHub30/OpenJev/blob/main/notebooks/OpenJev_Quickstart.ipynb)

オープンウェイト LLM で動く **System One モデル** の実装です。TypeSafe AI の
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) と同じ考え方
—「テキストを生成するのではなく、型付きの質問に対して確率分布を返す」— を、
Hugging Face 上の任意の instruct モデル (Qwen, Llama, Gemma, SmolLM など) で再現します。

- **生成しない**: 各質問は閉じた候補集合 (Yes/No、選択肢番号、レベル番号) に対する
  1 回の forward pass で答えます。トークンを 1 つもサンプリングしないので、スキーマ外の
  値が出ることがなく、幻覚的な「文字列」も生まれません。
- **質問数に対してほぼ定数時間**: 1 リクエスト内の全質問 × 全候補を 1 バッチで採点し、
  共通する `state` 部分は KV キャッシュを 1 度だけ計算してバッチ全体に共有します。
- **校正された確率**: `probabilities` は候補集合上の softmax、`confidence` はその形状から
  計算した統計量。温度スケーリング (`openjev calibrate`) と proper scoring rule による
  LoRA 微調整 (`scripts/train_calibrated.py`) で校正を改善できます。
- **ワイヤ互換**: `POST /v1/systemone` / `GET /v1/models` のリクエスト・レスポンス形式は
  TypeSafe の OpenAPI と同じなので、公式 `typesafe-sdk` (Python/JS) が無改造で動きます。

## インストール

```bash
git clone https://github.com/GitHub30/OpenJev.git
cd OpenJev
uv venv && uv pip install -e ".[hf,server,dev]"
# GPU なら PyTorch は CUDA 版を先に入れてください
```

## 使い方

### Google Colab で試す

上の **Open In Colab** バッジ ([notebooks/OpenJev_Quickstart.ipynb](notebooks/OpenJev_Quickstart.ipynb)) を開き、GPU ランタイムで上から実行してください
(A100/L4 なら Qwen2.5-7B、T4 なら 1.5B を自動選択。推論 → 評価/校正 → API サーバー + 公式 SDK → LoRA 微調整まで一通り動きます)。

### Python から

```python
from openjev import SystemOneEngine, load_backend

engine = SystemOneEngine(backend=load_backend("Qwen/Qwen2.5-1.5B-Instruct"))

response = engine.system_one(
    state="Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. I'm losing sales. Please help ASAP.",
    questions={
        "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "Charges, refunds, invoices", "integrations": "Connecting Stripe or Slack", "other": None},
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the customer's problem?",
            "criteria": ["Cosmetic issue", "Broken feature with a workaround", "Blocking issue with no workaround"],
        },
    },
)
print(response.answers["urgency"].noul)          # 0.98
print(response.answers["team"].choice, response.answers["team"].confidence)
print(response.answers["severity"].score, response.answers["severity"].probabilities)
```

`load_backend("mock")` を渡すとモデル無しの決定的なバックエンドになります (テスト用)。

### サーバーとして (公式 SDK 互換)

```bash
openjev serve --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/systemone \
  -H "Authorization: Bearer anything" -H "Content-Type: application/json" \
  -d '{"state": "I was charged twice. Please help ASAP.", "model": "jev-latest",
       "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}}}'
```

公式 SDK からは環境変数を向けるだけです:

```bash
pip install typesafe-sdk
TYPESAFE_API_KEY=anything TYPESAFE_BASE_URL=http://localhost:8000 python -c "
from typesafe_sdk import Noul, TypeSafeClient
with TypeSafeClient() as c:
    r = c.system_one('I was charged twice.', {'billing': Noul(instructions='Is this about billing?')})
print(r.nouls['billing'].noul)"
```

`OPENJEV_API_KEY` を設定すると Bearer トークンを検証します。

### CLI

```bash
openjev ask -m Qwen/Qwen2.5-1.5B-Instruct -s "I was charged twice, refund ASAP" \
  --noul "billing=Is this about billing?" \
  --choice "team=Which team?|billing,integrations,other" \
  --score "urgency=How urgent?|can wait,this week,today"
```

## 質問の型 (Jev と同じ 3 プリミティブ)

| type | リクエスト | レスポンス |
|---|---|---|
| `noul` | `instructions`, 任意の `criteria: {true, false}` | `noul` (Yes の確率 0–1) |
| `choice` | `instructions`, `criteria: {name: 説明 or null}` (最大 255) | `choice`, `probabilities`, `confidence` |
| `score` | `instructions`, `criteria: [level0, level1, ...]` (2–10) | `score` (期待値), `probabilities`, `legend`, `confidence` |

`state`, `instructions`, `criteria` の値は文字列でも JSON オブジェクト/配列でも構いません。

## 仕組み

1. **コンパイル** ([prompting.py](src/openjev/prompting.py)) — 各質問を「state + 質問 + 番号付き選択肢」
   のプロンプトと候補ラベル (`Yes`/`No`, `1..N`, `0..K-1`) に変換します。
   同じリクエスト内の全プロンプトは state 部分を共有します。
2. **採点** ([backends/hf.py](src/openjev/backends/hf.py)) — 全質問の全候補を 1 バッチにまとめ、
   共通トークン接頭辞を 1 回だけ forward して KV キャッシュをバッチに複製、残りの
   短い接尾辞だけを並列に処理します。LM head は必要な位置にだけ適用するため、
   語彙サイズに関係なくメモリは平坦です。ラベルが互いの接頭辞になる場合
   (`1` と `12`) は end-of-turn トークンを付加して曖昧さを除きます。
3. **確率化** ([engine.py](src/openjev/engine.py)) — 候補の対数尤度を (校正温度付きで) softmax し、
   `noul` / `choice` / `score` に整形します。`score` は確率加重平均なので `1.3` のような
   小数になります。
4. **confidence** ([confidence.py](src/openjev/confidence.py)) — choice は上位 2 候補の差 (margin)、
   score は順序を考慮した分散ベース (dispersion) が既定。`ConfidenceConfig` で
   `top1` / `entropy` にも切り替えられます。

## 校正 (Calibrated Decisions)

Jev は "Reinforcement Learning for Calibrated Decisions" で訓練されています。
OpenJev では答えが常に閉じた候補集合上の分布なので、**proper scoring rule** を
直接最適化できます (正直な確率でしか最小化されない損失 = 校正の学習)。

ラベル付きデータは 1 行 1 リクエストの JSONL です ([examples/triage.jsonl](examples/triage.jsonl)):

```json
{"state": "...", "questions": {"team": {"type": "choice", "criteria": {...}}, "urgent": {"type": "noul", ...}},
 "gold": {"team": "billing", "urgent": true, "severity": 2}}
```

```bash
# 精度 / NLL / ECE を測る
openjev eval -m Qwen/Qwen2.5-1.5B-Instruct --data examples/triage.jsonl

# 型ごとの温度を当てはめる (モデルは変えない)
openjev calibrate -m Qwen/Qwen2.5-1.5B-Instruct --data examples/triage.jsonl -o calibration.json
openjev serve -m Qwen/Qwen2.5-1.5B-Instruct --calibration calibration.json

# LoRA で候補分布の NLL (+ 任意で Brier) を最小化する
uv pip install -e ".[train]"
python scripts/train_calibrated.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --data train.jsonl --eval-data dev.jsonl --output checkpoints/openjev-lora --epochs 2 --brier-weight 0.5
openjev serve -m Qwen/Qwen2.5-1.5B-Instruct --adapter checkpoints/openjev-lora
```

## テスト

```bash
pytest                                                       # モックバックエンド
OPENJEV_TEST_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct pytest tests/test_hf_backend.py   # 実モデル
```

## 制限

- 文字列生成はしません (Jev と同じ設計上のトレードオフ)。
- 品質はベースモデル次第です。1.5B クラスでも英語の分類は実用的ですが、
  微妙な選択肢の区別や日本語などは、`criteria` の説明を具体的にするか校正/微調整で補ってください。
- Jev のような RL ベースの校正訓練そのものは含みません。含まれるのは温度スケーリングと、
  proper scoring rule による教師あり LoRA 微調整です。

## ライセンス

Apache-2.0
