# Semantic Codec v0.3.1 (Optimization Patch) 実装計画書

## 概要
v0.3.0 で確認された「深度マップ（Depth Map）が元解像度のまま保存され、ファイルサイズを圧迫している」問題を修正し、セマンティック圧縮器としての効率を最適化する。

## Proposed Changes

### [Component: Encoder]

#### [MODIFY] [app.py](file:///c:/Users/Lil4C/.gemini/antigravity/scratch/SemanticCodec/app.py)
- **深度マップのリサイズ**: `encode_image` 内で深度マップを `width` / `height` の 1024px 以下（アスペクト比維持）に縮小してから保存する。
- **保存形式の変更**: `PNG` → `WEBP` (Lossless) に変更。ファイル名を `depth_map.webp` とし、ZIP 内の容量を削減する。

### [Component: Decoder]

#### [MODIFY] [app.py](file:///c:/Users/Lil4C/.gemini/antigravity/scratch/SemanticCodec/app.py)
- **深度マップの読み込み**: `depth_map.webp` の読み込みに対応。
- **リサイズ処理の整理**: すでにエンコード時に縮小されていることを考慮し、デコーダ側でのアスペクト比維持とリサイズをより堅牢にする。

---

## Verification Plan

### Automated Tests
- Encode を実行し、生成された `.semic.jpg` を解凍して `depth_map.webp` が 1024px 以下であることを確認。
- Decode を実行し、背景の構造が正しく再構築されることを確認。

### Manual Verification
- 統計カードで `depth_map` コンポーネントの容量が大幅に減っていることを確認。
