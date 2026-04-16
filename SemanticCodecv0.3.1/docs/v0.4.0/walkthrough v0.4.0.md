# Semantic Codec v0.4.0 — Walkthrough

## 実装完了サマリー

v0.4.0 では以下の4フェーズを完了。

---

### Phase 1: 背景品質バグ修正 + スタイル分類

**問題**: 背景が油絵調になったり、被写界深度が過度にかかる現象。
**原因**: プロンプトがすべての画像に対して「photorealistic」を一律適用。

**解決策**:
- `classify_image_style()` — BLIP VQA による4カテゴリ自動分類
- `get_style_prompts()` — 分類結果に応じた専用プロンプト注入
- `apply_dof_blur()` — Depth マップベースの DOF ポストプロセス（任意適用）

**変更ファイル**: `app.py` (エンコーダ/デコーダ), `index.html` (DOF スライダー), `app.js`

---

### Phase 2: Multi-ControlNet (Segmentation)

**追加モデル**:
- `nvidia/segformer-b0-finetuned-ade-512-512` — ADE20K 150クラス
- `lllyasviel/control_v11p_sd15_seg` — ControlNet Seg

**変更**:
- `init_models()` に Segformer + ControlNet Seg の読み込みを追加
- `generate_seg_map()` — ADE20K パレット変換付きセグメンテーション生成
- エンコーダに `seg_map.webp` 保存処理を追加
- デコーダの `sd_pipeline()` が `controlnet=[depth, seg]` と `controlnet_conditioning_scale=[depth_scale, seg_scale]` を受け取るように変更

---

### Phase 3: UX — 進捗表示 + 比較スライダー

**進捗表示**:
- `callback_on_step_end` で各ステップの経過時間を記録
- `decode_progress` グローバル dict に session_id 単位で保存
- `/api/progress/<session_id>` SSE エンドポイント (300ms ポーリング)
- フロントエンド: shimmer アニメーション付きプログレスバー + ETA 表示


---

### Phase 4: CLI モード

`cli.py` を新規作成:
```
python cli.py encode photo.jpg --mode HighFi -o output.semic.jpg
python cli.py decode output.semic.jpg --steps 50 --dof 5
```

ASCII プログレスバー付き: `[████████░░░░░░] 8/20 | ETA: 12.3s`

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `app.py` | Segformer, Multi-ControlNet, スタイル分類, DOF, SSE 進捗 |
| `cli.py` | **新規** — CLI エンコード/デコード |
| `templates/index.html` | v0.4.0 バッジ, seg_map 統計, Seg Scale スライダー, DOF スライダー, 進捗バー |
| `static/app.js` | seg_map 統計計算, seg_scale/dof_strength 送信, SSE 購読 |
| `static/style.css` | seg-fill 色, プログレスバー |
| `docs/v0.4.0/specifications v0.4.0.md` | **新規** — 仕様書 |
