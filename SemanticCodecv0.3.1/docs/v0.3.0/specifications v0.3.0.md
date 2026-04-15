# Semantic Codec v0.3.0 仕様書 (Final)

本ドキュメントは、v0.3.0 で完了したアーキテクチャの抜本的アップデートを定義します。

---

## 1. パラダイムシフト: Canny → Depth 構造ガイドの移行

### 1-1. 背景
v0.2.0 で HighFi モード（ControlNet + Canny）のデコード時に、前景被写体のテクスチャが背景全体に増殖する暴走バグが発生。原因は Canny エッジ検出（1986年の古典的手法）がピクセルの明暗差を無差別に線として拾い、前景の輪郭線までもが背景再構成のガイドとして ControlNet に強制されていたこと。

### 1-2. 解決策
構造ガイドを **Depth（深度マップ）** に全面移行。

| 項目 | v0.2.0 (Canny) | v0.3.0 (Depth) |
|------|----------------|----------------|
| 抽出モデル | `cv2.Canny`（古典CV） | **Depth Anything v2 Small**（学習済みAI） |
| ControlNet | `lllyasviel/sd-controlnet-canny` | **`lllyasviel/control_v11f1p_sd15_depth`** |
| 記録データ | 1bit 白黒 PNG（線画） | 8bit グレースケール PNG（距離濃淡） |
| 保存ファイル名 | `edge_canny.png` | **`depth_map.png`** |
| デフォルト CN スケール | 1.0 | **0.5**（Standard: 0.3） |

### 1-3. Depth マップの仕組み
各ピクセルが「カメラからの距離」をグレースケール（0=近い, 255=遠い）で表現。線画ではなく**滑らかな距離のグラデーション**。テクスチャや前景輪郭の情報を含まないため、暴走の原因が構造的に排除される。

### 1-4. 背景再構築の3要素協調
```
① thumb.jpg    → 色のガイド（GaussianBlur で色グラデーション化）
② depth_map    → 空間配置ガイド（手前/奥/地平線）
③ VQA プロンプト → 意味内容ガイド（多角的テキスト記述）
```

---

## 2. VQA 多角的プロンプト抽出の完全改修

### 2-1. 前景マスキング
BLIP に入力する画像から `fg_alpha` マスクで被写体部分を黒塗りし、純粋に背景のみの画像を入力。これにより「馬」「人」等の前景要素がプロンプトに混入することを防止。

### 2-2. マルチプレフィックス・キャプショニング
既存の `BlipForConditionalGeneration` を活用し、4 つの視点からプレフィックス付きキャプション生成を実行:

| # | プレフィックス | 抽出する情報 |
|---|--------------|-------------|
| 1 | `"the background scene shows"` | 背景全体の描写 |
| 2 | `"the weather and lighting is"` | 天候・光源・雰囲気 |
| 3 | `"the ground or surface is"` | 地面・質感・素材 |
| 4 | `"in the distance there is"` | 遠景要素（山、空、建物等） |

### 2-3. プロンプト最終構成
```
[prefix_1の結果], [prefix_2の結果], [prefix_3の結果], [prefix_4の結果],
masterpiece, photorealistic, highly detailed, 8k resolution, sharp focus
```

### 2-4. ネガティブプロンプト
```
blurry, painting, abstract, artificial, low quality, deformed
```

---

## 3. `.semic.jpg` フォーマット (v0.3.0)

ポリグロット（JPEG + ZIP 結合バイナリ）アーキテクチャは v0.2.0 から変更なし。

### 内部構造
```text
image.semic.jpg (Polyglot: JPEG preview + ZIP payload)
  ├── manifest.json      # version, guide_type, width, height, prompt, seed, lora 等
  ├── fg_color.webp       # 前景 RGB (Lossy WebP, quality=85)
  ├── fg_alpha.webp       # 前景 Alpha (Lossless WebP)
  ├── thumb.jpg           # 256x256 (Quality=8) 超低品質サムネイル
  └── depth_map.png       # Depth Anything v2 による深度マップ (8bit Gray PNG)
```

### manifest.json 追加フィールド
- `"version": "0.3.0"` — フォーマットバージョン
- `"guide_type": "depth"` — 構造ガイドの種別

### 後方互換性
v0.3.0 は **Depth 専用パイプライン**。旧バージョン（Canny）の `.semic` ファイルのデコードには非対応（プロトタイプ段階のため切り捨て）。

---

## 4. マスク膨張処理 (バグB 修正)

Decoder にて、`bg_mask`（Inpaint 対象マスク）を `cv2.dilate()` で膨張処理。

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
bg_mask_np = cv2.dilate(bg_mask_np, kernel, iterations=2)
```

前景輪郭から約 7〜14 ピクセル外側まで Inpaint 領域を拡張し、サムネイル由来のモザイク・ブラー残りを AI による再描画で完全にカバー。

---

## 5. UX 改善

### 5-1. 圧縮率統計表示
Encode 完了時に以下を JSON レスポンスとして返却し、UI にカード表示:
- 元画像サイズ / `.semic.jpg` サイズ
- 圧縮率（%）/ 圧縮比（X:1）
- コンポーネント別の容量内訳（fg_color, fg_alpha, depth_map, thumb）

Encode API はファイルを一時保存し、2段階フロー（JSON → ダウンロードリンク）で提供。

### 5-2. Advanced Settings パネル（Decode UI）
以下のパラメータを UI スライダーから動的に制御可能:

| パラメータ | 範囲 | デフォルト | 効果 |
|-----------|------|----------|------|
| ControlNet Scale | 0.0〜1.0 | 0.5 | Depth マップの影響力 |
| Denoise Strength | 0.5〜1.0 | 1.0 | 再生成の自由度 |
| Guidance Scale | 1.0〜20.0 | 7.5 | プロンプト追従度 |
| Inference Steps | 10〜100 | Auto (mode依存) | 推論ステップ数 |

---

## 6. 技術スタック

| コンポーネント | モデル / ライブラリ |
|--------------|-------------------|
| 前景分離 | `rembg` (U2Net) |
| キャプション | `Salesforce/blip-image-captioning-large` (VQA プレフィックス方式) |
| 深度推定 | `depth-anything/Depth-Anything-V2-Small-hf` |
| 背景再構成 | `runwayml/stable-diffusion-inpainting` |
| 構造ガイド | `lllyasviel/control_v11f1p_sd15_depth` (ControlNet Depth) |
| メタデータ | `piexif` (EXIF 埋め込み) |
| Web フレームワーク | Flask |
