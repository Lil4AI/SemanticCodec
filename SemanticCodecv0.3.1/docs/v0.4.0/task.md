# Semantic Codec v0.4.0 — タスクチェックリスト

- `[ ]` uncompleted tasks
- `[/]` in progress tasks
- `[x]` completed tasks

## Phase 1: Background & Pipeline Fixes (DOF / Style Bug)
- `[x]` VQA: 事前定義された4つのカテゴリ（写真/イラスト/絵画/模様等）に基づくスタイル分類枠組みの実装
- `[x]` Prompt: 分類結果に応じた専用のパンフォーカス向けポジティブ・ネガティブプロンプトの注入処理
- `[x]` Decoder: 後処理としてのカスタム「被写界深度ぼかし（DOF）」関数の実装

## Phase 2: Multi-ControlNet (Segmentation)
- `[x]` `init_models`: Segformer モデルおよび `control_v11p_sd15_seg` の読み込み追加
- `[x]` Encoder: 入力画像からの Segmentation マップ生成機能の追加
- `[x]` Encoder: `.semic` に `seg_map.webp` を含める処理の追加
- `[x]` Decoder: Multi-ControlNet として `Depth + Seg` をパイプラインに渡す処理

## Phase 3: UX & Progress Streaming
- `[x]` Backend: SD Pipeline のコールバック実装と進捗状態のメモリ/Redis共有
- `[x]` Backend: Flask SSE (`/api/progress`) エンドポイントの作成
- `[x]` Frontend: 進捗バー・推定残り時間 (ETA) を表示する UI の実装
- `[x]` Frontend: Before/After 比較スライダーコンポーネントの実装

## Phase 4: CLI Mode & Settings
- `[x]` `cli.py` として独立した CLI コマンドモードの実装
- `[x]` Advanced Settings: ユーザーが「背景ぼかし（DOF）」の強さを調整できる UI の追加

## Phase 5: Documentation
- `[x]` `v0.4.0/walkthrough.md`, `specifications.md` の作成（終了時）
