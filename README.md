# Diffusion Transformer Decoder for Quantum Error Correction

連続拡散モデル（Continuous Diffusion Model）を量子誤り訂正のデコーダに応用した研究です。
画像生成 AI で用いられる **拡散トランスフォーマー（Diffusion Transformer, DiT）** をシンドローム復号に適用し、
従来手法が必要としていた反復ステップ数を大幅に削減しながら、同等以上の論理誤り率を達成することを目指します。

---

## 研究概要

量子コンピュータの実用化における最大の障壁の一つが、量子ビットのノイズに起因するエラーです。
これを解決する量子誤り訂正では、エラーを高速かつ高精度に特定する **デコーダ** が不可欠ですが、
従来手法では 100 ステップ以上の反復計算を要し、リアルタイム処理への適用が困難でした。

本研究では、画像生成 AI で広く使われる連続拡散モデルを量子誤り訂正デコーダに応用し、
拡散トランスフォーマーと呼ばれる新アーキテクチャを構築しました。
従来モデルが最低でも 100 ステップ以上（最大 200 ステップ）を要していたのに対し、
本手法では **わずか 30 ステップで同等以上の精度** を達成し、計算コストを大幅に削減しながら
高い誤り訂正性能を維持できることを実証しました。

限られたハードウェア資源のもとで高性能な量子コンピュータを実現するための技術的ステップであり、
将来的にはより大規模な量子システムにおける実用的なリアルタイムデコーダへの展開が期待されます。

### 対象符号

- **Bivariate Bicycle (BB) 符号**：`[[72, 12, 6]]`、`[[144, 12, 12]]` などで実験
- ノイズモデル：`DEPOLARIZE1`（脱分極ノイズ）／既定の物理エラー率 `p = 0.06`

---

## 手法のポイント

| 要素 | 内容 |
|------|------|
| アーキテクチャ | Diffusion Transformer（DiT） |
| 予測対象 | `x_0` 予測（pred_x0 パラメータ化） |
| ノイズスケジュール | cosine schedule |
| 時間 | 連続時間サンプリング（`t ∈ [0, 1]`） |
| Self-Conditioning | 直前ステップの予測 `prev_x0` をモデルに再入力し精度を向上 |
| 条件付け | シンドローム（`label_y`）を `in_context` / `cross_attn` の2方式で注入 |
| 損失関数 | Weighted MSE（エラービットに重み 10 倍。希少なエラーへの感度を確保） |
| データ生成 | Stim + qLDPC によるオンザフライ生成（学習中に逐次サンプリング） |

### デコード（サンプリング）方式

本リポジトリには 3 種類のデコードスクリプトを用意しています。

| スクリプト | 手法 | 特徴 |
|------------|------|------|
| `decode_dit_DDIM.py` | DDIM | 決定論的サンプリング。少ステップでの高速復号に有効 |
| `decode_dit_DDPM.py` | DDPM | 事後分散を含む確率的サンプリング |
| `decode_dit_greedy.py` | Greedy | 事後平均（論文 Eq. A17）を取り、確率分布の最大値（平均値）で復号 |

各スクリプトはサンプリングステップ数を `[1, 10, 20, …, 100, 200]` でスイープし、
**LER（論理誤り率）/ BER（ビット誤り率）/ スループット** を CSV とグラフに出力します。

---

## ディレクトリ構成

```
.
├── train_dit.py            # 拡散トランスフォーマーの学習スクリプト
├── decode_dit_DDIM.py      # デコード：DDIM の数式を利用
├── decode_dit_DDPM.py      # デコード：DDPM の数式を利用
├── decode_dit_greedy.py    # デコード：事後平均（最大値）でデコード
│
├── diffusion/              # 拡散過程（q_sample, α・βスケジュール等）の処理
├── model/                  # モデル定義
│   ├── model_dit_*.py      #   dit: 拡散トランスフォーマー本体
│   ├── *ffn*.py            #   ffn: feed-forward NN（手法の有効性を素早く検証するための簡易版）
│   └── data_loader.py      #   データローダ・config 読み込み
│
├── generate/               # シミュレータ動作確認用（生成器が正しく動くかの検証）
├── results/                # 学習結果・損失グラフの格納先
│   ├── 72_12_6/            #   [[72,12,6]] 符号の実験結果
│   ├── 144_12_12/          #   [[144,12,12]] 符号の実験結果
│   └── heatmap/            #   ランダム値が徐々に2値へ分類されていく様子を可視化した図
├── test/                   # ffn 用のデコード・学習スクリプト
└── config.yaml             # 各種ハイパーパラメータ
```

---

## セットアップ

### 必要なライブラリ

```bash
pip install torch numpy matplotlib tqdm pyyaml
pip install stim sympy qldpc
```

CUDA 対応 GPU を推奨します（AMP / `torch.compile` による高速化に対応）。

### config.yaml（例）

```yaml
DISTANCE: 6              # 符号距離 L
SAMPLES_PER_EPOCH: 16384 # 1エポックあたりの生成サンプル数
BATCH_SIZE: 512
TIMESTEPS: 200           # 拡散の最大ステップ数
HIDDEN_SIZE: 256
DEPTH: 6
NUM_HEADS: 4
DATASET_DIR: ./data      # テスト用 npz の格納先
```

---

## 使い方

### 学習

オンザフライでデータを生成しながら学習します（最大 20000 エポック、Self-Conditioning 付き）。

```bash
python train_dit.py \
    --config config.yaml \
    --results_dir ./result_dit_selfcond \
    --mode in_context
```

主な引数：

- `--mode` … `in_context`（既定）/ `cross_attn`
- `--prob` … 学習時の物理エラー率を上書き（既定 0.06）
- `--lr` … 学習率を上書き（既定 1e-4）
- `--pretrained_path` … 指定した重みから再開（ファインチューニング）

チェックポイントは 50 エポックごと、ベストモデルは検証損失更新時に自動保存され、
中断時も `checkpoint_interrupted_epoch_*.pth` として保存されます。

### デコード（評価）

学習済みモデルを使い、ステップ数をスイープして性能を評価します。

```bash
# DDIM
python decode_dit_DDIM.py --model_path path/to/best_model.pth --batch_size 1024

# DDPM
python decode_dit_DDPM.py --model_path path/to/best_model.pth --batch_size 1024

# Greedy（事後平均）
python decode_dit_greedy.py --model_path path/to/best_model.pth --batch_size 1024
```

主な引数：

- `--model_path` …（必須）学習済みモデル `.pth`
- `--npz` … テスト用 npz を直接指定（未指定なら `DATASET_DIR` から探索）
- `--output_csv` / `--output_graph` … 出力ファイル名
- `--seed` … 乱数シード（再現性確保、既定 42）

出力：

- `*.csv` … `Steps, LER, BER, Throughput` の表
- `*.png` … サンプリングステップ数 対 LER のグラフ（対数軸）
- `decode_dit_DDIM.py` のみ `graph_dit/` に `x_t` の軌跡プロット（2値へ収束する過程）も保存

---

## 結果

`results/` 以下に符号ごとの学習結果・損失グラフを格納しています。
`heatmap/` には、初期のランダムな連続値が逆拡散の過程で徐々に 0/1 の 2 値へ分類されていく様子を
可視化した図を収録しています。

> 主な知見：従来手法が必要とした 100〜200 ステップに対し、本手法は約 30 ステップで
> 同等以上の論理誤り率を達成。

---

## 技術スタック

- **PyTorch**（AMP, `torch.compile`, cuDNN benchmark による高速化）
- **Stim** … 量子回路シミュレーション・検出器サンプリング
- **qLDPC** … BB 符号の構成・論理演算子の取得
- **AdamW + ReduceLROnPlateau**, 勾配クリッピング, 重み付き MSE 損失
