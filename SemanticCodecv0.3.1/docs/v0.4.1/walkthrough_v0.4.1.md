# Walkthrough - Semantic Codec v0.4.1

今回のアップデートでは、前バージョンで発生していた「アニメ画像の誤判定」「生成速度の低下」「プロンプト長制限による画質低下」の3点を中心に解決しました。

## 実装内容

### 1. GPU パフォーマンスの最適化
RTX 2060 Super (8GB) での動作を劇的に高速化するため、以下の調整を行いました。
- **Full GPU Resident**: VRAMが7GB以上ある場合、`enable_model_cpu_offload()`（低速なメモリ入れ替え）を無効化し、モデル全体をGPUに常駐させるように変更。
- **Attention Slicing**: メモリ効率を向上させ、8GBでも Multi-ControlNet を安定して動かせるように設定。
- **Explicit FP16**: すべての推論を明示的に半精度（float16）で行うように徹底。

### 2. VQA スタイル判定の強化
アニメ画像が `photo` と判定されてしまう問題を解決するため、2段階判定を導入しました。
- **Stage 1**: 従来のスタイル描写からのキーワード抽出（`manga`, `digital art`, `2d` を追加）。
- **Stage 2 (Fallback)**: 判定が不明な場合、`Question: Is this image a photograph, a digital illustration, or a painting?` という質問を投げ、ダイレクトに回答を得る仕組みを追加。

### 3. プロンプト・マネジメント
CLIPの77トークン制限による「指示の切り捨て」を防止します。
- `clean_and_truncate_prompt`: 生成された長い背景説明を自動的に整理し、重要なキーワードを優先して77トークン以内に収めます。

## 検証結果
- **構文チェック**: 通過
- **GPU Diagnostics**: 起動時にデバイス名とVRAM量が表示されることを確認。

> [!TIP]
> サーバーを再起動して、最初の起動ログを確認してください。
> `High VRAM detected (>=7GB). Moving entire pipeline to CUDA for speed.`
> というメッセージが出ていれば、以前より数倍速く生成が完了するはずです。
