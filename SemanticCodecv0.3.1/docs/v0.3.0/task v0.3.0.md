# Semantic Codec v0.3.0 — タスクチェックリスト

- `[ ]` uncompleted tasks
- `[/]` in progress tasks
- `[x]` completed tasks

## Phase 1: Depth 移行（構造ガイドのパラダイムシフト）
- `[x]` Encoder: `generate_canny_edge()` 関数の削除
- `[x]` Encoder: Depth Anything v2 Small の導入と深度マップ生成処理
- `[x]` Encoder: `.semic` 内 `edge_canny.png` → `depth_map.png` への置換
- `[x]` Encoder: `manifest.json` へ `guide_type: "depth"` の追加
- `[x]` Decoder: ControlNet モデルを `sd-controlnet-canny` → `control_v11f1p_sd15_depth` に変更
- `[x]` Decoder: `depth_map.png` 読み込み・ControlNet への適用処理

## Phase 2: VQA 多角的プロンプト抽出の完全改修
- `[x]` 前景マスキング: BLIP 入力画像から `fg_alpha` で被写体を黒塗り除去
- `[x]` VQA 的プレフィックス強制: 4つの視点からのキャプション生成と結合
- `[x]` 高画質化プロンプトの自動付与（末尾に固定テキスト結合）
- `[x]` ネガティブプロンプトの維持

## Phase 3: マスク膨張処理（バグB 修正）
- `[x]` `cv2.dilate()` による `bg_mask` の膨張処理の追加
- `[x]` カーネルサイズ・iterations の設定 (ELLIPSE 7x7, iterations=2)

## Phase 4: UX 改善
- `[x]` 圧縮率統計 API レスポンスの実装（`app.py`）
- `[x]` 圧縮率統計カードの HTML/CSS 実装
- `[x]` 圧縮率統計の JS ハンドリング
- `[x]` Advanced Settings パネルの HTML/CSS 実装
- `[x]` パラメータ送信・API 受け取りの JS/Python 実装

## Phase 5: ドキュメント・仕上げ
- `[x]` `docs/v0.3.0/implementation_plan.md`
- `[x]` `docs/v0.3.0/task.md`
- `[x]` `docs/v0.3.0/specifications.md`
- `[x]` `docs/v0.3.0/walkthrough.md`
