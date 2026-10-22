# Semantic Codec v1.0.0 実装タスクチェックリスト

- [x] **Step 1: 依存関係と環境設定の更新**
  - [x] `requirements.txt` に `peft>=0.7.0` を追加 (LCM-LoRA 対応)
  - [x] v0.4.1 のローカルバックアップを `Backup/SemanticCodec pre-v1.0.0 local Backup` へ安全退避

- [x] **Step 2: コアエンジン `codec_engine.py` の新規作成**
  - [x] VRAM 排他制御マネージャ `ModelManager` (Encoder と Decoder のメモリ完全分離)
  - [x] 前景 ROI Coding (Tight-BBox クロップ + 背景ゼロクリア + Morphological Edge Bleed)
  - [x] 8-bit インデックスパレット PNG (`mode="P"`) による Seg マップ超圧縮
  - [x] エンコードプロファイル階層化 (`Lite`, `Standard`, `HighFi`)
  - [x] Turbo デコードモード (LCM-LoRA 4-step / UniPC 8-step フォールバック)
  - [x] Reinhard LAB 色調和アルゴリズム (`harmonize_color_lab`)
  - [x] 前景インペイント付き ピラミッドボケ (`apply_layered_dof`)
  - [x] 評価指標計算モジュール (`bpp`, `compression_ratio`, `PSNR`, `SSIM`)

- [x] **Step 3: コンテナ・スキーマ (`manifest.json`) の刷新**
  - [x] `fg_bbox`, `bpp`, `analysis` メタデータフィールド追加
  - [x] 旧バージョン (v0.1.0〜v0.4.1) の後方互換ローダー実装

- [x] **Step 4: `app.py` の薄層化と新エンドポイント整備**
  - [x] 重複ロジックを `codec_engine.py` に委譲
  - [x] 1時間経過した一時ファイルの自動クリーンアップ
  - [x] `POST /api/inspect` (デコード前軽量マニフェスト取得・Lite プレビュー生成)
  - [x] `POST /api/decode` (Turboモード, プロンプト上書き, 色調和トグル対応)
  - [x] SSE 進捗配信のタイムアウトおよびエラー通知ハンドリング

- [x] **Step 5: Web UI の完成 (`index.html`, `app.js`, `style.css`)**
  - [x] v1.0.0 デザイン・ヘッダー更新
  - [x] bpp / 前景占有率 / AI 抽出プロンプトの可視化カード
  - [x] 4択デコードモード (`Lite`, `Turbo`, `Standard`, `HighFi`)
  - [x] Before / After インタラクティブ比較スライダー実装

- [x] **Step 6: `cli.py` の刷新**
  - [x] 4大サブコマンド (`encode`, `decode`, `inspect`, `benchmark`) の実装
  - [x] Windows コントロール互換性対応 (UnicodeEncodeError 回避)

- [x] **Step 7: ドキュメントの整備**
  - [x] `docs/v1.0.0/` 4点セットの作成 (`implementation_plan`, `specifications`, `task`, `walkthrough`)
  - [x] ルート `README.md` の新規作成 (アーキテクチャ図、先行研究比較表、使用例)
