---
title: JevのオープンウェイトモデルOpenJevを開発した話
tags: LLM Python transformers 機械学習 FastAPI
---

## TL;DR

- TypeSafe AI が発表した「System One モデル」[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) は、テキストを生成せずに **型付きの質問 (Yes/No・選択・スコア) へ校正された確率分布を返す** モデルです。ただし重みも学習法の詳細も非公開です。
- そこで、Hugging Face 上の任意の instruct モデル (Qwen, Llama, Gemma など) を使って同じ振る舞いを再現する **OpenJev** を作りました。
- 公式 API `POST /v1/systemone` とワイヤ互換なので、**公式 `typesafe-sdk` が無改造で動きます**。
- リポジトリ: https://github.com/GitHub30/OpenJev
- Colab で試す: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GitHub30/OpenJev/blob/main/notebooks/OpenJev_Quickstart.ipynb)

## Jev とは何か

TypeSafe AI のブログでは、Jev を「ソフトウェアが直接使える、速くて構造化された判断を下すための新しいクラスのフロンティアモデル」と位置づけています。要点は次の通りです。

- **トークンを逐次生成しない**。答えはあらかじめ定義された型 (スキーマ) の中の値で、並列に決まる。
- 応答は 70〜500ms 程度 (対して LLM は数秒〜数百秒)。
- 出力トークンは無料、入力は $0.042 / 1M トークン。
- スキーマ外の値が出ないので「幻覚率 0%」と主張。
- "Reinforcement Learning for Calibrated Decisions (RLCD)" という手法で、**認識論的に正直な確率** を出すよう訓練。
- 文字列生成はできない。選択肢は最大 255。

ドキュメント ([docs.typesafe.ai](https://docs.typesafe.ai)) によると、API は「`state` (文脈) に対して複数の型付き `questions` を投げると、それぞれの答えが返る」というものです。質問の型は 3 つだけ。

| type | 入力 | 出力 |
|---|---|---|
| `noul` | Yes/No の質問 | `noul`: Yes の確率 (0〜1) |
| `choice` | 選択肢の名前と説明 | `choice`, `probabilities`, `confidence` |
| `score` | 順序付きのレベル説明 (2〜10) | `score` (確率加重平均), `probabilities`, `legend`, `confidence` |

```json
{
  "state": "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. I'm losing sales. Please help ASAP.",
  "model": "jev-latest",
  "questions": {
    "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges and invoices", "integrations": "Connecting Stripe or Slack", "other": null}}
  }
}
```

「質問を増やしても応答時間はほぼ変わらない」「各質問は独立に評価されるので context rot が起きない」というのも特徴として挙げられています。

## なぜ作ったか

ブログには重み・アーキテクチャ・学習データのいずれも書かれていません。一方で、**API の形だけ見れば LLM の logits から実装できる** ことに気づきました。

- Yes/No の確率 → `Yes` と `No` の次トークン確率を正規化すればよい
- 選択肢の確率分布 → 選択肢番号の確率を正規化すればよい
- スコア → レベル番号の分布の期待値

つまり「生成しない」「スキーマ外の値が出ない」「確率が返る」は、**閉じた候補集合に対する 1 回の forward pass の採点** として自然に実現できます。Jev の本質はここにあると解釈して、その部分をオープンに実装したのが OpenJev です。

## 設計

### 全体像

```
request (state + questions)
   │
   ▼  prompting.py   質問ごとに「state + 質問 + 番号付き選択肢」のプロンプトと候補ラベルを作る
   │
   ▼  backends/hf.py 全質問 × 全候補を 1 バッチで採点 (log P(候補 | プロンプト))
   │
   ▼  engine.py      温度付き softmax → noul / choice / score に整形、confidence を計算
   │
   ▼  server.py      FastAPI で /v1/systemone として公開
```

### 1. 質問を「候補集合」にコンパイルする

各質問はプロンプトと候補ラベルのペアになります。

- `noul` → 候補 `Yes` / `No`
- `choice` → 選択肢を `1. billing: ...`, `2. integrations: ...` と番号付きで列挙し、候補 `1..N`
- `score` → レベルを `0. Cosmetic issue`, `1. ...` と列挙し、候補 `0..K-1`

プロンプトは chat template を通し、assistant ターンの先頭に候補ラベルが来る形にします。

```
## State
Hi, I've been trying to connect my Stripe account ...

## Question
Which team should handle this ticket?

## Options
1. billing: Charges, refunds and invoices
2. integrations: Connecting third-party services such as Stripe or Slack
3. other

Reply with the option number only.
```

ポイントは、**同じリクエスト内の全プロンプトが `state` 部分を共有する** ようにしていることです。これが次の高速化の前提になります。

### 2. 1 回の forward で全部採点する

`HFBackend.score()` は、リクエスト内の全質問・全候補を `_Seq(prefix, cand_ids)` に展開し、まとめて処理します。

- **共通接頭辞の KV キャッシュ共有**: 全シーケンスの共通トークン接頭辞 (system prompt + state) を 1 回だけ forward し、その KV キャッシュをバッチ分に複製 (`DynamicCache.batch_repeat_interleave`)。残りの短い接尾辞 (質問 + 選択肢 + 候補) だけをバッチで処理します。
- **LM head は必要な位置だけ**: `model.get_decoder()` で hidden state を取り、候補トークンの位置だけ `lm_head` を適用します。`batch × seq × vocab` の logits を作らないので、語彙 15 万のモデルでもメモリが平坦です。
- **ラベルの曖昧性**: 選択肢が 10 個以上あると `1` が `12` の接頭辞になり、短いラベルが常に有利になります。この場合は各候補に end-of-turn トークン (`<|im_end|>` など) を付加して、`1<|im_end|>` と `12<|im_end|>` を比較します。chat template を実際に適用してトークンを探索するので、モデルを問わず動きます。

```python
@torch.inference_mode()
def score(self, tasks):
    seqs = [...]                                   # (task, cand, prefix_ids, cand_ids)
    shared = self._shared_prefix_len(seqs)         # 全シーケンス共通のトークン数
    logprobs = self._score_with_shared_prefix(seqs, shared)  # 失敗時は full sequence にフォールバック
    ...
```

共有あり/なしで対数尤度を比較したところ差は 2e-5 程度で、CPU の Qwen2.5-1.5B で約 1.8 倍高速でした (state が長いほど効きます)。

### 3. 確率と confidence

候補の対数尤度を softmax して分布にします。`confidence` は公式でも「分布の形状から計算した統計量」とだけ説明されているので、切替可能にしました。

- `choice`: 既定は **margin** (1 位と 2 位の確率差)
- `score`: 既定は **dispersion** (レベル番号の標準偏差を最大値で正規化して 1 から引く)。隣接レベルに割れている方が、両端に割れているより「自信あり」と扱う順序尺度向けの定義です
- ほかに `top1`、`entropy` (1 − 正規化エントロピー)

### 4. 校正 (Calibrated Decisions)

Jev の RLCD そのものは再現できませんが、答えが常に閉じた候補集合上の分布なので、**proper scoring rule を直接最適化できる** のが System One の良いところです。

- **温度スケーリング** (`openjev calibrate`): 型ごと (noul/choice/score) に温度を黄金分割探索で当てはめる。答えの順位は変えず、鋭さだけ直す。
- **LoRA 微調整** (`scripts/train_calibrated.py`): 候補分布の NLL (+任意で Brier score) を損失に、peft で LoRA を学習。テキスト生成は学習ループにも一切登場しません。

```python
def decision_loss(cand_logprobs, gold, brier_weight, label_smoothing):
    log_dist = cand_logprobs.log_softmax(-1)      # 候補集合上の分布
    nll = -log_dist[gold]
    if brier_weight > 0:
        onehot = F.one_hot(torch.tensor(gold), log_dist.numel()).float()
        nll = nll + brier_weight * ((log_dist.exp() - onehot) ** 2).sum()
    return nll
```

学習データは 1 行 1 リクエストの JSONL で、`gold` に正解を書くだけです。

```json
{"state": "...", "questions": {"team": {"type": "choice", "criteria": {...}}},
 "gold": {"team": "integrations", "urgent": true, "severity": 2}}
```

### 5. 公式 SDK 互換のサーバー

公式 Python SDK ([typesafe-sdk-python](https://github.com/typesafe-ai/typesafe-sdk-python)) には OpenAPI から生成された pydantic モデルが同梱されているので、それをそのまま鏡写しにしました。結果、環境変数を向けるだけで公式 SDK が動きます。

```bash
openjev serve --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

```python
import os
os.environ["TYPESAFE_API_KEY"] = "anything"
os.environ["TYPESAFE_BASE_URL"] = "http://localhost:8000"

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient() as client:
    r = client.system_one(
        "I was charged twice. Please help ASAP.",
        {
            "billing": Noul(instructions="Is this about billing?"),
            "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
            "urgency": Score(instructions="How urgent is this?", criteria=["low", "medium", "high"]),
        },
    )
print(r.nouls["billing"].noul, r.choices["tone"].choice, r.scores["urgency"].score)
```

## 動かしてみる

### 応答時間 (Colab A100 / Qwen2.5-7B-Instruct, bf16)

冒頭の Stripe の state に対して noul 質問の数を増やしたときの応答時間です。

| 質問数 | 応答時間 | 1 質問あたり |
|---|---|---|
| 1 | 74 ms | 73.6 ms |
| 4 | 89 ms | 22.2 ms |
| 16 | 165 ms | 10.3 ms |
| 64 | 503 ms | 7.9 ms |

Jev のブログが挙げる「70〜500ms」と同じレンジに、7B のオープンモデルで収まっています。state の KV キャッシュを共有しているので、質問を 64 倍にしても時間は 7 倍弱にしかなりません。これが "speculative fan-out" (使うかどうか分からない質問も一緒に投げておく) を成立させる性質です。

### 答えの中身

7B では「Stripe の接続に失敗」を `integrations` (confidence 1.00) と正しく振り分け、深刻度も `2` (0.9999) でした。CPU で試した 1.5B では `billing` 0.74 / `integrations` 0.26 に割れていたので、この種の微妙な区別はモデルサイズが効きます。

公式 SDK 経由でも同じ結果が取れます。

```
noul  : 0.98              # "Is this about billing?" (二重課金の問い合わせ)
choice: angry 0.08        # トーンは calm/angry で割れている → confidence が低い
score : 0.95  {0: 0.30, 1: 0.46, 2: 0.24}
```

confidence が低い答えはそのまま「人に回す」判断材料になります。

### 校正の効果

同梱の 16 件のサポートチケット (48 決定) で評価した結果です。

| | accuracy | NLL | ECE |
|---|---|---|---|
| 素の 7B | 0.833 | 1.555 | 0.182 |
| 温度スケーリング後 | 0.833 | **0.473** | **0.120** |

当てはまった温度は noul 3.7 / choice 6.0 / score 5.9 と、**素の instruct モデルはかなり過信している** ことが分かります (0.9999 のような確率を頻繁に出す)。順位を変えずに NLL が 1/3 になったので、確率をしきい値に使うなら校正は必須です。Jev が校正を訓練目標の中心に置いている理由が実感できました。

LoRA 微調整も同じノートブックで動作確認しています (16 件・6 ステップなので精度向上は期待しない動作確認です。7B で学習対象パラメータ 20M、学習後にアダプタをマージして推論まで通ります)。

## ハマったところ

**Colab で `pip install -e .` すると `import` できない。** editable install は `.pth` で `src/` をパスに足しますが、`.pth` は Python 起動時にしか読まれません。動作中のカーネルでは `ModuleNotFoundError` になるので、通常インストールに変更しました。

**Colab 同梱の torchao が古くて peft が落ちる。** `torchao 0.10.0` が入っており、peft の LoRA ディスパッチャが `>= 0.16.0` を要求して `ImportError` を投げます。OpenJev は torchao を使わないので `pip uninstall -y torchao` で回避しました。

**transformers のバージョン差で KV キャッシュ API が変わる。** `batch_repeat_interleave` → `to_legacy_cache` → tuple の順に試し、全部だめなら full sequence にフォールバックするようにしています。

**「1」と「12」問題。** 前述の通り、候補ラベルが互いの接頭辞になると短い方が常に有利になります。end-of-turn トークンの付加で解決しましたが、最初は 13 択で `lang0` ばかり選ばれて気づきました。

## Jev との違い・限界

正直に書いておきます。

- **RL ベースの校正訓練は含みません。** 含まれるのは温度スケーリングと、proper scoring rule による教師あり LoRA です。
- **品質はベースモデル次第です。** Jev のベンチマーク数値を再現したわけではありません。
- **`confidence` の定義は推測です。** 公式の計算式は非公開なので、切替可能な統計量として実装しています。
- 文字列生成はしません (これは Jev と同じ設計上のトレードオフです)。

それでも「型付きの質問に確率分布で答える」「質問を増やしてもコストがほぼ増えない」「スキーマ外の値が絶対に出ない」という System One の使い勝手は、手元の GPU と好きなオープンモデルで再現できます。

## 今後

- 汎用 System One モデル向けの学習データ構築 (boolq / banking77 / clinc_oos / sst5 / JGLUE などを 3 プリミティブに変換して混合)
- QLoRA 対応 (14B 以上を Colab で学習するため)
- llama.cpp / vLLM バックエンド

コードは MIT ライセンスで公開しています。Issue / PR 歓迎です。

https://github.com/GitHub30/OpenJev
