# Semantic Codec v0.4.1 - 仕様書

## 1. コアシステム要件
- OS: Windows 10/11
- 対象ハードウェア: NVIDIA GeForce RTX 2060 Super (8GB VRAM) 等のミドルクラスGPU
- 必須パッケージ: PyTorch (CUDA), Diffusers, Transformers, Rembg 等

## 2. VQA スタイル判定仕様
背景生成時のプロンプト汚染や誤判定を防ぐため、以下の2段階判定を行う。

### Stage 1: キーワード抽出
BLIPに `The artistic style of this image is best described as a` のPrefixを与え、出力文字列から以下のカテゴリを判定する。
- **photo**: `photo`, `photograph`, `real`, `dslr`, `natural` 等
- **illustration**: `illust`, `anime`, `2d`, `digital art`, `manga` 等
- **painting**: `paint`, `oil`, `watercolor`, `canvas` 等
- **pattern**: `pattern`, `texture`, `flat` 等

### Stage 2: ダイレクトQ&A (フォールバック)
Stage 1 で判定不能な場合、`Question: Is this image a photograph, a digital illustration, or a painting? Answer:` を使用し、回答から強制的に振り分ける。その際、BLIPの出張による質問文のオウム返し（echo）を検出し、これを判定前に除去する処理を行う。

## 3. プロンプト長制限 (Prompt Truncation) 仕様
CLIPの仕様である「77トークン制限」による意図しないキーワード切り捨てを防ぐため、以下のロジックで自動クリーニングを行う。
1. プロンプト全体をカンマ `,` で分割し意味のチャンクとする。
2. 日本語/英語の単語数をカウントし、「単語数 × 1.5 + 1」を推定トークン数として計算。
3. 推定トークン数が70トークン（バッファ含む）を超えた時点でそれ以降のチャンクを切り捨てる。

## 4. パフォーマンス・メモリ最適化 (GPU)
8GB VRAM で Multi-ControlNet + Stable Diffusion Inpainting を処理する場合のWDDMスワップ（メインメモリへの溢れ）を防ぐため、以下の最適化を適応。
- **モデル配置**: パイプライン全体には `enable_model_cpu_offload()` を適用し、使用するサブモジュールのみをGPUに送るシーケンシャル構成とする。
- **Attention**: Xformersによる `enable_xformers_memory_efficient_attention()` を適用。
- **FP16徹底**: UNet, VAE, ControlNet 全てにおいて `torch.float16` を明示的に指定。
- **デコード上限**: Inference 時の最大解像度を `768x768` (0.59 MP) でクリップし、超過する場合は縮小スケーリングして生成後に元サイズへリサイズする。
