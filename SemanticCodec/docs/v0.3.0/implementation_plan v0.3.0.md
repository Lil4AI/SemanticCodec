# Semantic Codec v0.3.0 実装計画書

## 概要
v0.2.0 で発見された ControlNet Canny 暴走バグと境界モザイク問題を根本的に解決するため、構造ガイドを **Canny → Depth** へ全面移行する。同時に、VQA的アプローチによるプロンプト抽出の完全改修、マスク膨張処理、圧縮率統計表示、UIパラメータ微調整パネルを実装する。

旧バージョン（Canny）との後方互換は **完全に切り捨て**、Depth 専用のクリーンなパイプラインとして構築する。

---

## Phase 1: Depth 移行（構造ガイドのパラダイムシフト）

### 1-1. Depth Anything v2 による深度マップ生成（Encoder）

**変更対象**: `app.py` — Encoder パイプライン

- `cv2.Canny` / `generate_canny_edge()` 関数を**完全に削除**。
- `transformers.pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")` を導入。
- Encode 時に元画像から深度マップ（8bit グレースケール PNG）を生成。
- `.semic` ZIP 内に `depth_map.png` として格納（旧 `edge_canny.png` を置換）。
- `manifest.json` に `"guide_type": "depth"` フィールドを追加。

### 1-2. ControlNet Depth への切り替え（Decoder）

**変更対象**: `app.py` — モデル初期化 / Decoder パイプライン

- `lllyasviel/sd-controlnet-canny` → `lllyasviel/control_v11f1p_sd15_depth` に変更。
- Decode 時に `depth_map.png` を読み込み、ControlNet Depth のガイドとして渡す。
- `controlnet_conditioning_scale` のデフォルトを `0.5` に設定（Depth は穏やかに効かせる）。
- 旧 Canny 関連コード（`edge_canny.png` 読み込み等）を完全削除。

---

## Phase 2: VQA 多角的プロンプト抽出の完全改修（Encoder）

**変更対象**: `app.py` — `generate_caption()` 関数

### 2-1. 前景マスキング
- BLIP に入力する画像から前景を除去（`fg_alpha` で被写体部分を黒塗り）。
- `Image.composite(original_img, Image.new("RGB", size, (0,0,0)), bg_mask)` で純粋な背景のみの画像を生成。

### 2-2. VQA 的プロンプトプレフィックス強制
- 複数の視点からプレフィックス付きキャプション生成を実行:
  1. `"the background scene shows"` → 背景全体の描写
  2. `"the weather and lighting is"` → 天候・光源
  3. `"the ground or surface is"` → 地面・質感
  4. `"in the distance there is"` → 遠景要素
- 各結果をカンマ区切りで結合。

### 2-3. 高画質化プロンプトの自動付与
- 結合テキスト末尾に `masterpiece, photorealistic, highly detailed, 8k resolution, sharp focus` を付与。

---

## Phase 3: マスク膨張処理（バグB 修正）

**変更対象**: `app.py` — Decoder パイプライン

- `bg_mask` に `cv2.dilate()` を適用し、Inpaint 対象領域を前景の外側に数ピクセル拡張。
- カーネル: `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))`、`iterations=2`。
- 画像解像度に応じた動的調整は v0.4.0 以降の課題とし、固定値で実装。

---

## Phase 4: UX 改善

### 4-1. 圧縮率統計表示（B2）

**変更対象**: `app.py`, `templates/index.html`, `static/app.js`, `static/style.css`

- Encode API のレスポンスを変更: ファイルダウンロードに加え、JSON で圧縮統計を返す。
  - 元画像サイズ、`.semic.jpg` サイズ、圧縮率（%）、各コンポーネント内訳。
- Encode を2段階に: (1) `/api/encode` で JSON 統計を返す → (2) `/api/encode/download` でファイルDL。
- フロントエンドに圧縮統計カードを表示。

### 4-2. UI パラメータ微調整パネル（B4）

**変更対象**: `templates/index.html`, `static/app.js`, `static/style.css`

- Decode 画面に「Advanced Settings」トグルパネルを追加。
- 調整可能パラメータ:
  - `controlnet_conditioning_scale` (0.0 〜 1.0, default 0.5)
  - `strength` (0.5 〜 1.0, default 1.0)
  - `guidance_scale` (1.0 〜 20.0, default 7.5)
  - `num_inference_steps` (10 〜 100, default は mode 依存)
- Decode API にこれらのパラメータを追加送信。

---

## Phase 5: ドキュメント・仕上げ

- `requirements.txt` の更新（`depth-anything` 依存追加は不要: `transformers` 経由）。
- `docs/v0.3.0/specifications.md` の作成。
- `docs/v0.3.0/walkthrough.md` の作成（実装・検証完了後）。

---

## ファイル変更一覧

| 操作 | ファイル | 概要 |
|------|---------|------|
| MODIFY | `app.py` | Depth移行、VQAプロンプト改修、マスク膨張、圧縮統計API |
| MODIFY | `templates/index.html` | 圧縮統計表示、Advanced Settings パネル |
| MODIFY | `static/app.js` | 統計表示ハンドリング、パラメータ送信 |
| MODIFY | `static/style.css` | 統計カード、パラメータパネルのスタイル |
| MODIFY | `requirements.txt` | 必要に応じて依存追加 |
| NEW | `docs/v0.3.0/implementation_plan.md` | 本ドキュメント |
| NEW | `docs/v0.3.0/task.md` | タスクチェックリスト |
| NEW | `docs/v0.3.0/specifications.md` | 最終仕様書（完了後） |
| NEW | `docs/v0.3.0/walkthrough.md` | 作業振り返り（完了後） |
