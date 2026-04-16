# Semantic Codec v0.4.0 実装計画書

## 概要
空間上の物体認識の強化（Segmentationの追加）、UXの大幅な向上（リアルタイム進捗、比較スライダー）、CLI対応、そして「被写界深度や絵画調の背景になってしまう合成違和感の一掃」を目標とするメジャーパッチ。

## Proposed Changes

### [Component: Background Generation & DOF Fix (Core Bug Fix)]
「被写界深度が強くかかった絵画調の背景になる問題」および「前景との合成違和感」を根本解決するためのアプローチ：
1. **VQAによる強固なスタイル分類枠組み（新規）**
   - 漠然とBLIPにキャプションを出させるのではなく、事前に「写真・イラスト・絵画・模様」という明確な分類カテゴリを定義します。
   - BLIPに `The artistic style of this image is` のような特定のプレフィックスで答えさせ、その出力結果から**事前に用意したカテゴリのどれに属するかを判定**するゼロショット的な枠組みを作ります。
   - 判定結果に応じて、背景生成時のSDプロンプト（ポジティブ/ネガティブ）をそれぞれ専用のものに切り替えます。
     - *分類例：「写真（Photo）」判定時* -> 強制パンフォーカスのプロンプト付与、ネガティブに「油絵、ぼやけ、被写界深度」を追加。
2. **前景の文脈を考慮した合成（Inpaint Mask）**
   - 現在の「完全に前景を消す」のではなく、サムネイルのブラー画像（`gen_bg`）において、前景のシルエットや色味をわずかに残し、AIに「ここに被写体があって、それに光が当たっている」という環境光のヒントを与える。
3. **後処理での人工的 DOF (被写界深度) 適用**
   - 生成されたパンフォーカスな背景に対し、ユーザーUIから「背景のぼかし強さ」を任意で設定できるようにする。デコード完了後に `depth_map` としきい値を元にして背景側をブラー処理し、前景を合成するロジックに変更。

### [Component: Encoder & AI Models (Multi-ControlNet)]
- **Segmentation モデルの追加**: `Transformers` から `Segformer` (例: `nvidia/segformer-b0-finetuned-ade-512-512`) 等を導入。
- **Encoder**: 深度（Depth）と同様に、セグメンテーションマップ（色分けされた領域画像）を生成し、`seg_map.webp` として保存・圧縮。
- **Decoder**: `ControlNetModel` をリスト化 `[controlnet_depth, controlnet_seg]` して `Multi-ControlNet` を構築。生成時にパレット画像に基づく空間意味情報をAIに強制する。

### [Component: UI / UX]
- **SSE 進捗ストリーミング**:
  - Flask 側に `/api/progress` エンドポイントを作成し、SSE でストリーミング。
  - SD Pipeline に `callback_on_step_end` を仕込み、現在ステップ、総ステップ、および時間計測から算出した「完了予測時間 (ETA)」をフロントエンドに送信。
- **Before/After 比較スライダー**:
  - デコード完了表示画面に、元画像（またはエンコード前プレビュー）とデコード後の画像を重ね合わせたCSS比較スライダー（`clip-path`を使用）を実装。

### [Component: CLI Mode]
- `app.py` 末尾で `argparse` を導入。
- 実行フラグによる分岐（例: `python app.py --mode cli --action encode --input image.jpg`）。サーバーを立ち上げずに直接処理を行い、ファイルを保存・終了する。

---

## Verification Plan

### User Target Validation
ユーザーに設計思想やアプローチの妥当性を確認してもらい、承認を得てから実装に着手します。

### Manual / Automated Tests
- CLIコマンドでのエンコード/デコード処理の完走確認。
- ブラウザ上でのプログレスバー進行と時間が正確に表示されるかの検証。
- 生成された画像の背景がぼけ過ぎておらず、スライダー等で自然な構成になるかの目視検証。
