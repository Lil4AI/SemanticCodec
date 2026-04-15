# Semantic Codec v0.3.1 — タスクチェックリスト

- `[ ]` uncompleted tasks
- `[/]` in progress tasks
- `[x]` completed tasks

## Phase 1: Optimization (Depth Map)
- `[ ]` Encoder: 深度マップの 1024px 以下へのリサイズ処理追加
- `[ ]` Encoder: `PNG` -> `WEBP` (Lossless) への形式変更
- `[ ]` Decoder: `depth_map.webp` の読み込み・解凍ロジック対応
- `[ ]` Decoder: デコード時の深度マップリサイズ処理の整理

## Phase 2: Polishing
- `[ ]` Advanced Settings のデフォルトパラメータの再検討
- `[ ]` 統計カードの文言微調整（もし必要なら）

## Phase 3: Documentation
- `[ ]` `docs/v0.3.0/v0.3.1/walkthrough.md` の作成
- `[ ]` Version を 0.3.1 に更新
