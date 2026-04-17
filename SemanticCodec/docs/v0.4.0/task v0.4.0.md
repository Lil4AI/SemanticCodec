# Semantic Codec v0.4.0 — タスクチェックリスト

- `[ ]` uncompleted tasks
- `[/]` in progress tasks
- `[x]` completed tasks

## Phase 1: Background & Pipeline Fixes (DOF / Style Bug)
- `[/]` VQA: 事前定義された4つのカテゴリ（写真/イラスト/絵画/模様等）に基づくスタイル分類枠組みの実装
- `[/]` Prompt: 分類結果に応じた専用のパンフォーカス向けポジティブ・ネガティブプロンプトの注入処理
- `[/]` Decoder: 後処理としてのカスタム「被写界深度ぼかし（DOF）」関数の実装

## Phase 2: Multi-ControlNet (Segmentation)
- `[ ]` `init_models`: Segformer モデルおよび `control_v11p_sd15_seg` の読み込み追加
- `[ ]` Encoder: 入力画像からの Segmentation マップ生成機能の追加
- `[ ]` Encoder: `.semic` に `seg_map.webp` を含める処理の追加
- `[ ]` Decoder: Multi-ControlNet として `Depth + Seg` をパイプラインに渡す処理

## Phase 3: UX & Progress Streaming
- `[ ]` Backend: SD Pipeline のコールバック実装と進捗状態のメモリ/Redis共有
- `[ ]` Backend: Flask SSE (`/api/progress`) エンドポイントの作成
- `[ ]` Frontend: 進捗バー・推定残り時間 (ETA) を表示する UI の実装
- `[ ]` Frontend: Before/After 比較スライダーコンポーネントの実装

## Phase 4: CLI Mode & Settings
- `[ ]` `app.py`: `argparse` による CLI コマンドモードの追加
- `[ ]` Advanced Settings: ユーザーが「背景ぼかし（DOF）」の強さを調整できる UI の追加

## Phase 5: Documentation
- `[ ]` `v0.4.0/walkthrough.md`, `specifications.md` の作成（終了時）
