# Semantic Codec v0.4.0 仕様書

---

## 1. アーキテクチャ概要 — Multi-ControlNet パイプライン

### 1-1. 構造ガイドの2軸化: Depth + Segmentation
v0.3.x では Depth マップ単体であった構造ガイドを、**Multi-ControlNet** に拡張。
Depth（奥行き）と Segmentation（意味的領域分類）の2つの視点で背景を制御する。

| ControlNet | モデル | 役割 |
|-----------|--------|------|
| Depth | `lllyasviel/control_v11f1p_sd15_depth` | 空間配置（手前/奥/地平線）|
| **Seg (NEW)** | `lllyasviel/control_v11p_sd15_seg` | **物体カテゴリ境界**（道路/空/建物/植物 etc.）|

### 1-2. Segmentation マップ生成
`nvidia/segformer-b0-finetuned-ade-512-512` (ADE20K 150クラス) を使用。
出力は ADE20K カラーパレット（クラスごとに一意な色）に変換した RGB 画像。

---

## 2. 背景スタイル分類 + 動的プロンプト注入

### 2-1. VQA ベースのスタイル検出
BLIP に対し「Is this a photograph?」「Is this an illustration?」等の質問を行い、
画像を 4カテゴリに自動分類:

| カテゴリ | 例 |
|---------|-----|
| `photo` | 写真・カメラ撮影画像 |
| `illustration` | デジタルイラスト・アニメ風 |
| `painting` | 油絵・水彩・古典絵画 |
| `pattern` | テクスチャ・模様・抽象 |

### 2-2. スタイル別プロンプト
分類結果に応じて、専用のポジティブ・ネガティブプロンプトを自動注入。
例: `photo` → 「photorealistic, sharp focus, deep depth of field」
例: `illustration` → 「anime illustration, clean lineart, vibrant colors」

---

## 3. `.semic.jpg` フォーマット (v0.4.0)

```text
image.semic.jpg (Polyglot: JPEG preview + ZIP payload)
  ├── manifest.json         # version, prompt, negative_prompt, image_style, seed, etc.
  ├── fg_color.webp          # 前景 RGB (Lossy WebP, quality=85)
  ├── fg_alpha.webp          # 前景 Alpha (Lossless WebP)
  ├── thumb.jpg              # 256x256 (Quality=8) サムネイル
  ├── depth_map.webp         # Depth Anything v2 深度マップ  
  └── seg_map.webp (NEW)     # Segformer ADE20K セグメンテーションマップ
```

### manifest.json 追加フィールド (v0.4.0)
- `"version": "0.4.0"`
- `"image_style"`: `"photo"` / `"illustration"` / `"painting"` / `"pattern"`
- `"negative_prompt"`: スタイル別ネガティブプロンプト
- `"model_version": "stable-diffusion-inpainting-1.5_multi-controlnet-depth-seg"`

---

## 4. 被写界深度 (DOF) ポストプロセス

Decode 完了後にオプションで適用可能な DOF ブラー:
1. Depth マップを反転（遠景 = 高値）
2. 出力画像にガウシアンブラー適用（radius = ユーザー指定の強度）
3. `Image.composite()` で深度マスクに基づきブレンド

| パラメータ | Advanced Settings UI | CLI Flag |
|-----------|---------------------|----------|
| DOF Strength | `0～15` スライダー | `--dof` |

---

## 5. リアルタイム進捗表示

### Backend: SSE (Server-Sent Events)
- SD Pipeline の `callback_on_step_end` で各ステップの完了時刻を記録
- グローバル `decode_progress` dict に session_id 単位で保存
- `/api/progress/<session_id>` エンドポイントが 0.3秒間隔で SSE ストリーム

### Frontend
- ステップ数 / 合計ステップ数の表示
- ETA (推定残り時間) の計算・表示
- アニメーション付きプログレスバー（shimmer エフェクト）

---

## 6. Before/After 比較スライダー

Decode 完了時に、入力 `.semic.jpg` のプレビュー画像（Before）と
AI 再構成後の画像（After）を `clip-path: inset()` スライダーで比較可能。

---

## 7. CLI モード (`cli.py`)

```
python cli.py encode <image> [--mode Standard|HighFi|Lite] [-o output.semic.jpg]
python cli.py decode <semic>  [--mode Standard|HighFi|Lite] [-o output.jpg]
                               [--cn-scale 0.5] [--seg-scale 0.4]
                               [--strength 1.0] [--guidance 7.5]
                               [--steps 20] [--dof 0]
```

ASCII プログレスバー付き。ETA 表示あり。

---

## 8. 技術スタック

| コンポーネント | モデル / ライブラリ |
|--------------|-------------------|
| 前景分離 | `rembg` (U2Net) |
| キャプション | `Salesforce/blip-image-captioning-large` |
| 深度推定 | `depth-anything/Depth-Anything-V2-Small-hf` |
| **セグメンテーション** | **`nvidia/segformer-b0-finetuned-ade-512-512`** |
| 背景再構成 | `runwayml/stable-diffusion-inpainting` |
| 構造ガイド (Depth) | `lllyasviel/control_v11f1p_sd15_depth` |
| **構造ガイド (Seg)** | **`lllyasviel/control_v11p_sd15_seg`** |
| メタデータ | `piexif` |
| Web フレームワーク | Flask |

---

## 9. Advanced Settings パネル (v0.4.0)

| パラメータ | 範囲 | デフォルト | 効果 |
|-----------|------|----------|------|
| ControlNet Scale | 0.0～1.0 | 0.5 | Depth マップの影響力 |
| **Segmentation Scale** | **0.0～1.0** | **0.4** | **Seg マップの影響力** |
| Denoise Strength | 0.5～1.0 | 1.0 | 再生成の自由度 |
| Guidance Scale | 1.0～20.0 | 7.5 | プロンプト追従度 |
| Inference Steps | 10～100 | Auto | 推論ステップ数 |
| **DOF Strength** | **0～15** | **0 (Off)** | **被写界深度ブラー強度** |
