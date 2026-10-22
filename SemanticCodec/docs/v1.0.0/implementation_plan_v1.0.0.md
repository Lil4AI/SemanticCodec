# SemanticCodec v1.0.0 (Final Research Prototype) 実装設計書・作業指示書

> [!IMPORTANT]
> **本ドキュメントの位置づけ（実行担当エージェント Opus 5.5 への指示）**
> 本書は `SemanticCodec`（現在 `v0.4.1`）を、研究プロトタイプおよびポートフォリオの集大成となる **`v1.0.0 (Final Research Prototype Edition)`** へ引き上げるための完全な実装ブループリントである。
> 既存のコード構造や内部仕様は後方互換性を維持しつつ大胆に刷新してよい。本書のステップ1〜ステップ7を順番通りに実装・検証せよ。

---

## 0. 現状（v0.4.1）の課題と v1.0.0 で導入するコア技術

### 解決すべき6つの致命的課題
1. **逆圧縮（ファイル肥大化）問題**:
   - `fg_color.webp` が背景領域をゼロクリアせず元画像全体サイズで保存されており、`seg_map.webp` も24bit RGB Lossless（最大1024px）で保存されているため、入力JPEGより `.semic.jpg` の方が大きくなるケースがある。
   - エンコード時のモード選択（`Lite` / `Standard` / `HighFi`）が `manifest.json` に文字列を書くだけのダミーになっている。
2. **Segmentation リサイズ時の色補間バグ**:
   - `ADE20K_PALETTE` でRGB化した `seg_map` を `Image.LANCZOS` でリサイズしているため、クラス境界にパレット外の中間色が発生し ControlNet Seg の認識精度を破壊している。
3. **色・照明コンテキストの喪失（`strength=1.0` 問題）**:
   - デコード時のデフォルト `strength` が `1.0` のため、せっかく保存した `thumb.jpg` の低周波色情報が初期潜在変数（Latent）から100%消去され、元画像と全く異なる色調・明るさの背景が生成される。
4. **前景境界のハロー（色フリンジ）と WebP リンギング**:
   - `fg_color` の半透明エッジに元背景の色が残ったまま新背景にペーストされるため、境界に不自然な縁取りが出る。
5. **8GB VRAM におけるモデル常駐の無駄と推論の遅さ（20〜50 step）**:
   - Encode専用モデル（BLIP, Depth-Anything, SegFormer）が Decode 時も GPU VRAM を占有している。また Multi-ControlNet で 20〜50 ステップ回すため展開に数十秒かかる。
6. **コード重複と未完成UI**:
   - `cli.py` が `app.py` のロジックを丸ごとコピーしており保守性が低い。
   - `app.js` に残骸がある Before/After 比較スライダーが `index.html` に未実装。また `README.md` が存在しない。

---

### v1.0.0 で新たに採用する技術一覧（先行研究からの知見導入）

| 技術カテゴリ | 採用技術・アルゴリズム | 効果 |
| :--- | :--- | :--- |
| **1. 空間ビット配分 (ROI Coding)** | **Foreground Tight-BBox Cropping + Alpha Zeroing**<br>前景の非ゼロAlpha領域の最小外接矩形（`fg_bbox`）のみを切り出して保存。背景画素は完全に `(0,0,0)` 化。 | 前景データ容量を **40〜80%削減** |
| **2. 境界品質向上** | **Morphological Color Decontamination (Edge Bleed)**<br>前景マスク境界の外側へ前景内部色を数ピクセル膨張させてからWebP圧縮し、デコード時はソフトアルファ合成。 | 前景エッジの黒ずみ（WebPリンギング）と元背景ハローを完全除去 |
| **3. 構造マップ超圧縮** | **8-bit Indexed Palette PNG (`mode="P"`) + Nearest Resize**<br>SegマップをRGB画像ではなく 0〜149 のクラスインデックス（パレットPNG）として保存し、`Image.NEAREST` を徹底。 | `seg_map` 容量を **約1/10（2〜6 KB）** に削減＆クラス境界の劣化ゼロ |
| **4. 圧縮プロファイル階層化** | **Bitrate-Aware Encode Profiles (`Lite` / `Standard` / `HighFi`)**<br>エンコードモードに応じて解像度・量子化品質・格納マップを動的に切り替え。 | **bpp（Bits Per Pixel）** の明確な制御を実現 |
| **5. 超高速デコード (2025-2026 SOTA)** | **LCM-LoRA (`latent-consistency/lcm-lora-sdv1-5`) + `LCMScheduler`**<br>4〜6ステップで高品質な背景を再構成する `Turbo (LCM 4-Step)` デコードモードを追加。 | デコード速度を **5〜10倍高速化（約1.5〜3秒）** |
| **6. 照明・色調の一貫性** | **Reinhard LAB Color Transfer (Photometric Harmonization)**<br>生成された背景画像に対し、`thumb.jpg`（背景マスク領域）の $L^*a^*b^*$ 平均・標準偏差をブレンド適用。 | 元画像の時間帯・色温度・環境光と背景が自然に一致 |
| **7. 光学的被写界深度** | **Depth-Layered Pyramid Bokeh + Foreground Occlusion Inpainting**<br>ブラー前に背景の前景領域を `cv2.inpaint` で埋め、深度を3層（遠・中・近）に分割して異なるカーネルで合成。 | 前景の色滲みがない自然な一眼レフ風ボケ |
| **8. 環境非依存の再現性** | **CPU-Generator Deterministic Noise (`torch.Generator("cpu")`)**<br>GPUアーキテクチャに依存しないCPU乱数生成器をパイプラインに渡す。 | 異なるGPU環境間での **Semantic Drift 抑止** |
| **9. 研究評価メトリクス** | **Rate-Distortion Metrics (`bpp`, `Compression Ratio`, `Foreground PSNR`, `SSIM`)**<br>符号化効率（bits per pixel）および前景/全体の再構成指標を自動算出。 | 研究プロトタイプとしての客観的説得力を確立 |

---

## 1. ターゲットディレクトリ構造

リポジトリ内の `SemanticCodec/` 配下を以下の構成に整理・再構築する。

```text
SemanticCodec/
├── README.md                  # [NEW] アーキテクチャ図・先行研究比較・使用方法を含む総合ドキュメント
├── requirements.txt           # [MODIFY] 依存パッケージ整理 (peft等の明記)
├── codec_engine.py            # [NEW] Encode/Decode/VRAM管理/画像処理/評価指標のコアエンジン
├── app.py                     # [MODIFY] Flask Web API & SSE サーバー (codec_engine を呼び出す薄い層)
├── cli.py                     # [MODIFY] CLI (encode / decode / inspect / benchmark サブコマンド)
├── templates/
│   └── index.html             # [MODIFY] Before/After比較スライダー、Turboモード、bpp表示、色調和トグル追加
├── static/
│   ├── app.js                 # [MODIFY] スライダー制御、SSE、メトリクス表示
│   └── style.css              # [MODIFY] Before/After比較スライダー等のUIスタイル追加
└── docs/
    ├── PROJECT_RULES.md
    └── v1.0.0/                # [NEW] v1.0.0 仕様書・計画書・ウォークスルー
        ├── implementation_plan_v1.0.0.md
        ├── specifications_v1.0.0.md
        ├── task_v1.0.0.md
        └── walkthrough_v1.0.0.md
```

---

## 2. ステップバイステップ実装手順

### Step 1: `requirements.txt` の更新

LoRA統合やLCM-LoRAロードに必要な `peft`、および定量評価・科学計算用のパッケージを明記する。

```txt
Flask>=3.0.0
Werkzeug>=3.0.1
Pillow>=10.1.0
numpy>=1.26.2
rembg[gpu]>=2.0.53
torch>=2.1.2
torchvision>=0.16.2
transformers>=4.36.2
diffusers>=0.25.0
accelerate>=0.25.0
peft>=0.7.0
opencv-python>=4.8.1.78
piexif>=1.1.3
python-multipart
```

---

### Step 2: コアエンジン `codec_engine.py` の新規作成

`app.py` と `cli.py` に分散・重複していたAIモデル管理・エンコード・デコード・画像処理・評価指標計算を `codec_engine.py` に一元化する。以下の仕様をすべて実装すること。

#### 2.1 VRAM ステージ分離型モデルマネージャ (`ModelManager`)
8GB VRAM 環境で最高のパフォーマンスを出すため、**Encoder用モデル群**と**Decoder用モデル群**のGPUメモリ占有を排他制御する。

- **定数**:
  - `ADE20K_PALETTE`: 既存の 150×3 `np.uint8` 配列を維持。さらに `PIL.Image.putpalette()` 用に 256×3（768要素）にゼロパディングした1次元リスト `ADE20K_FLAT_PALETTE` を用意する。
- **`ModelManager` クラス**:
  - `self.lock = threading.Lock()` で推論をスレッドセーフ化。
  - `load_encoder_models()`:
    - デコーダモデル（`sd_pipeline`）がGPU上にある場合は `torch.cuda.empty_cache()` を呼ぶ。
    - `blip_processor`, `blip_model` (`Salesforce/blip-image-captioning-large`, FP16)
    - `depth_estimator` (`depth-anything/Depth-Anything-V2-Small-hf`)
    - `seg_feature_extractor`, `seg_model` (`nvidia/segformer-b0-finetuned-ade-512-512`, FP16)
    - エンコード完了時は `offload_encoder_to_cpu()` で `blip_model` と `seg_model` を CPU に退避し `torch.cuda.empty_cache()` を実行する（デコード時に約1.2GBのVRAMを空けるため）。
  - `load_decoder_models()`:
    - エンコーダモデルがGPUにあればCPUへ退避。
    - `controlnet_depth` (`lllyasviel/control_v11f1p_sd15_depth`, FP16)
    - `controlnet_seg` (`lllyasviel/control_v11p_sd15_seg`, FP16)
    - `sd_pipeline` (`runwayml/stable-diffusion-inpainting`, FP16, `safety_checker=None`)
    - デフォルトスケジューラ設定用として `UniPCMultistepScheduler.from_config(sd_pipeline.scheduler.config)` と、高速モード用として `LCMScheduler.from_config(sd_pipeline.scheduler.config)` の両方を切り替え可能にする。
    - `sd_pipeline.enable_model_cpu_offload()` および `xformers`（フォールバックとして `enable_attention_slicing()` と `enable_vae_slicing()`）を有効化。

#### 2.2 エンコードプロファイル (`ENCODE_PROFILES`) とビットレート最適化
エンコードモード（`Lite` / `Standard` / `HighFi`）に実際の圧縮パラメータを紐づける：

```python
ENCODE_PROFILES = {
    "Lite": {
        "preview_max_dim": 256,
        "preview_quality": 60,
        "thumb_max_dim": 128,
        "thumb_quality": 6,
        "fg_color_quality": 72,
        "fg_alpha_lossless": False,
        "fg_alpha_quality": 80,
        "depth_max_dim": 256,
        "depth_lossless": False,
        "depth_quality": 55,
        "include_seg": False,
        "seg_max_dim": 0,
    },
    "Standard": {
        "preview_max_dim": 384,
        "preview_quality": 75,
        "thumb_max_dim": 256,
        "thumb_quality": 8,
        "fg_color_quality": 84,
        "fg_alpha_lossless": True,
        "fg_alpha_quality": 100,
        "depth_max_dim": 384,
        "depth_lossless": False,
        "depth_quality": 70,
        "include_seg": True,
        "seg_max_dim": 384,
    },
    "HighFi": {
        "preview_max_dim": 512,
        "preview_quality": 82,
        "thumb_max_dim": 256,
        "thumb_quality": 12,
        "fg_color_quality": 92,
        "fg_alpha_lossless": True,
        "fg_alpha_quality": 100,
        "depth_max_dim": 512,
        "depth_lossless": True,
        "depth_quality": 100,
        "include_seg": True,
        "seg_max_dim": 512,
    },
}
```

#### 2.3 前景の Tight-BBox 切り出しと Edge Color Decontamination
`extract_and_optimize_foreground(original_img: Image.Image)` 関数を実装する：
1. `fg_rgba = remove(original_img)` でRGBA画像を取得。
2. Alphaチャンネル `alpha_np = np.array(fg_rgba.split()[3], dtype=np.uint8)` に対し、微小ノイズを除去（`alpha_np[alpha_np < 8] = 0`）。
3. 非ゼロ領域（`alpha_np > 0`）の最小外接矩形 `bbox = [x_min, y_min, x_max, y_max]` を算出する。
   - もし非ゼロ画素が全体の 0.1% 未満なら「前景なし（風景のみ）」と判定し `has_foreground = False`, `bbox = None` とする。
4. `bbox` で `original_img`（RGB）と `alpha_np` をクロップする。
5. **Color Decontamination（境界色汚染の除去とWebPリンギング防止）**:
   - クロップ後の RGB 配列 `rgb_crop` において、確実な前景内部マスク `core_mask = (alpha_crop >= 210).astype(np.uint8)` を作る。
   - 半透明境界およびすぐ外側のリング領域 `ring_mask = ((alpha_crop > 0) & (alpha_crop < 210)).astype(np.uint8)` を求める。さらに `cv2.dilate` で外側4pxまで広げた領域に対し、`cv2.inpaint(rgb_crop, inpaint_mask, 3, cv2.INPAINT_TELEA)`（またはモルフォロジー膨張による前景内部色の引き伸ばし）を適用し、完全に透明な領域（外側から4px以上離れた `alpha == 0`）は `[0, 0, 0]` でゼロクリアする。
   - これにより、(a) 元画像の背景色が半透明エッジに残る現象を防ぎ、(b) Lossy WebP圧縮時に黒背景 `(0,0,0)` が前景エッジに滲む現象を防ぎ、(c) 背景領域のエントロピーを完全に0にしてファイルサイズを激減させる。

#### 2.4 Segmentation マップの 8-bit インデックスパレットPNG化
`generate_seg_map(image: Image.Image, max_dim: int)` 関数を刷新する：
1. `Segformer` の出力 `logits` を元画像のアスペクト比を保ったまま `max_dim` 以下に bilinear 補間し、`argmax(dim=1)` でクラスID配列 `seg_ids`（`uint8`, 値域 `0..149`）を得る。
2. RGB変換（24bit化）は**行わず**、`seg_pil = Image.fromarray(seg_ids, mode="P")` として `seg_pil.putpalette(ADE20K_FLAT_PALETTE)` を適用し、**`format="PNG", optimize=True`** で保存する（ファイル名は `seg_map.png`、後方互換のためデコーダは `seg_map.png` と旧 `seg_map.webp` の両方に対応）。
3. デコード時に ControlNet Seg へ渡す際は、まず `mode="P"` のまま `Image.NEAREST` でターゲット解像度 `(proc_w, proc_h)` にリサイズし、**リサイズ後に `.convert("RGB")` を呼ぶ**。これでクラス境界に偽の中間色が一切発生しなくなる。

#### 2.5 デコード時の3大品質向上処理
1. **LCM 4-Step 超高速モード (`decode_mode == "Turbo"` または `use_lcm=True`)**:
   - `latent-consistency/lcm-lora-sdv1-5` をロード可能な場合は `LCMScheduler` に切り替え、`steps=4`（またはユーザー指定）、`guidance_scale=1.8`、`strength=0.85` で高速生成する（オフライン等で LCM-LoRA が取得できない場合は `UniPCMultistepScheduler` の 8ステップへ安全にフォールバック）。
2. **Photometric Color Harmonization (`harmonize_color_lab`)**:
   - `strength` のデフォルトを `1.0` から **`0.85`** に変更し、`thumb.jpg` の大域的な明暗・色相を初期潜在空間に残す。
   - さらに SD 生成後の `out_bg` に対し、`color_Lock`（デフォルトON、強度 `0.6`）が有効な場合、`base_bg`（アップスケール＆ブラー済みの `thumb.jpg`）の背景マスク領域（`bg_mask > 128`）における CIE $L^*a^*b^*$ 色空間の平均 $(\mu_L, \mu_a, \mu_b)$ と標準偏差 $(\sigma_L, \sigma_a, \sigma_b)$ を計算し、Reinhard の色変換式で `out_bg` の色調を元の背景に調和させる：
     $$C_{\text{harm}} = (1 - \alpha_{\text{lock}}) C_{\text{gen}} + \alpha_{\text{lock}} \left( \frac{\sigma_{\text{thumb}}}{\sigma_{\text{gen}} + \epsilon} (C_{\text{gen}} - \mu_{\text{gen}}) + \mu_{\text{thumb}} \right)$$
3. **Depth-Layered Pyramid Bokeh (`apply_layered_dof`)**:
   - `dof_strength > 0` の場合：
     1. まず `out_bg` の前景領域（`fg_alpha > 128`）を `cv2.inpaint` で周囲の背景色に馴染ませる（ブラー時に前景領域の偽色が背景側に滲み出るのを防ぐため）。
     2. `Depth-Anything-V2` の出力（255=近景, 0=遠景）を正規化し、遠景度マップ $f(x,y) = 1.0 - \text{depth}(x,y)/255.0 \in [0, 1]$ を作る。
     3. 中間ボケ画像 $B_{\text{mid}}$（半径 $r/2$）と最大ボケ画像 $B_{\text{far}}$（半径 $r$）を生成し、$f(x,y) \in [0, 0.5]$ では元画像と $B_{\text{mid}}$ を、$f(x,y) \in [0.5, 1.0]$ では $B_{\text{mid}}$ と $B_{\text{far}}$ を滑らかに線形補間する。

#### 2.6 定量評価メトリクス計算 (`compute_metrics`)
エンコード時およびデコード時に以下の研究指標を算出して返却する：
- **`bpp` (Bits Per Pixel)**: `(semic_size_bytes * 8) / (width * height)`
- **`compression_ratio_pct`**: `(1 - semic_size_bytes / original_file_size) * 100`
- **`fg_area_ratio_pct`**: 前景画素（`alpha > 128`）が画像全体に占める割合（%）
- デコード時（Lite背景または元画像との比較）:
  - **`fg_psnr_db`**: 前景領域における PSNR（dB）
  - **`bg_color_fidelity`**: `thumb.jpg` と生成背景のヒストグラム相関（0〜100%）

---

### Step 3: `manifest.json` 仕様のアップデート（v1.0.0 形式 + 完全後方互換）

v1.0.0 の `.semic.jpg` 内に格納する `manifest.json` のスキーマ：

```json
{
  "version": "1.0.0",
  "guide_type": "depth_seg",
  "width": 1920,
  "height": 1080,
  "has_foreground": true,
  "fg_bbox": [420, 180, 1380, 1060],
  "prompt": "the background scene shows ...",
  "negative_prompt": "blurry, ...",
  "image_style": "photo",
  "random_seed": 123456789,
  "model_version": "sd15-inpaint_multi-cn-depth-seg_v1.0",
  "encoded_mode": "Standard",
  "bpp": 0.42,
  "lora_model": "",
  "lora_scale": 1.0
}
```

- **後方互換ルール（Decoder）**:
  - `manifest.get("fg_bbox")` が存在しない旧ファイル（v0.1.0〜v0.4.1）は、`fg_bbox = [0, 0, width, height]` としてフルキャンバス貼り付けを行う。
  - `seg_map.png` がなければ `seg_map.webp` を探し、どちらもなければ（または `Lite` エンコードファイルであれば）Seg スケールを自動的に `0.0` に設定して推論する。
  - `depth_map.webp` がなければ `depth_map.png` を探す。

---

### Step 4: `app.py` のリファクタリングとエンドポイント拡張

`app.py` は `codec_engine.py` を呼び出すクリーンな Flask アプリケーションに書き換える。

1. **一時ファイル自動クリーンアップ**:
   - エンコード時に `temp_encode/` 内の作成から1時間以上経過した `.semic.jpg` を自動削除する `cleanup_old_temp_files()` を実装。
2. **エンドポイント一覧**:
   - `GET /`: `index.html` レンダリング
   - `POST /api/encode`:
     - `codec_engine.encode_to_semic()` を呼び出し、サイズ統計に加えて **`bpp`**, **`fg_area_ratio_pct`**, **`prompt`**, **`image_style`** を JSON で返却する（UI側で抽出されたプロンプトやスタイル分類結果も確認できるようにする！）。
   - `GET /api/encode/download/<download_id>`: 生成した `.semic.jpg` のダウンロード
   - `POST /api/inspect`: **[新規]** `.semic.jpg` をデコードする前にドロップした時点で、`manifest.json` の中身（解像度、抽出プロンプト、スタイル、シード値、エンコードモード、Liteプレビュー画像Base64）を即座に返す軽量エンドポイント。これにより、デコード後の Before/After 比較スライダーの「Before画像（Lite プレビュー）」を即座にセットできる。
   - `POST /api/decode`:
     - デコードモード `Lite` / `Turbo` (LCM 4-step) / `Standard` (20-step) / `HighFi` (50-step) に対応。
     - パラメータとして `controlnet_scale`, `seg_scale`, `strength` (default `0.85`), `guidance_scale`, `steps`, `dof_strength`, `color_harmonize` (default `true`), `custom_prompt`（オプション：ユーザーがプロンプトを上書きして背景を再生成できる機能！）を受け取る。
     - レスポンスヘッダに `X-Session-Id`, `X-Decode-Time-Sec` を付与して JPEG を返却。
   - `GET /api/progress/<session_id>`: 既存の SSE 進捗エンドポイントを維持。

---

### Step 5: フロントエンド（`index.html`, `app.js`, `style.css`）の完成

1. **バージョンバッジとヘッダ更新**:
   - `v1.0.0 — ROI Semantic & Generative Image Codec`
2. **Encode パネルの強化**:
   - 統計カードに **`Bitrate (bpp)`**（例: `0.38 bpp`）と **`Foreground Ratio`**（例: `28.4%`）を追加。
   - 抽出された **AI VQA Prompt** と **Detected Style バッジ（`photo` / `illustration` 等）** を結果カード内に表示し、AIが背景をどう理解したかを可視化する。
3. **Decode パネルの強化**:
   - モード選択を4択に拡張：
     - **Lite** (`0 steps • Instant Thumbnail + Crisp FG`)
     - **Turbo** (`4 steps • LCM-LoRA (~2s)`)
     - **Standard** (`20 steps • UniPC Balanced`)
     - **HighFi** (`50 steps • Full Multi-ControlNet`)
   - Advanced Settings 内に以下を追加：
     - `Denoise Strength` のデフォルトを `0.85` に更新（範囲 `0.50〜1.00`）。
     - **Color Harmonization (LAB Match)** チェックボックス（デフォルト Checked）。
     - **Prompt Override (Optional)** テキスト欄（`.semic.jpg` ドロップ時に `/api/inspect` で読み取った元のプロンプトが自動入力され、そのままでも、書き換えて別背景を生成することも可能にする）。
4. **Before / After 比較スライダーの実装**:
   - Decode 完了時、「Lite Mode（サムネイル背景＋前景）」と「AI Reconstructed（生成背景＋前景）」を左右にドラッグして比較できるインタラクティブスライダーを `#result-container` に実装する。

---

### Step 6: `cli.py` の刷新（4つのサブコマンド対応）

`cli.py` の重複コードをすべて削除し、`codec_engine.py` を呼び出す設計にする。

```bash
# 1. エンコード
python cli.py encode input.jpg --mode Standard -o output.semic.jpg

# 2. デコード (Turbo / Standard / HighFi / Lite)
python cli.py decode output.semic.jpg --mode Turbo --dof 4 -o reconstructed.jpg

# 3. インスペクト (.semic.jpg のマニフェスト・コンポーネント容量・bpp を表示)
python cli.py inspect output.semic.jpg

# 4. ベンチマーク (Lite / Standard / HighFi の3モードで一括エンコードし、サイズ・bpp・圧縮率の比較表を出力)
python cli.py benchmark input.jpg
```

---

### Step 7: `README.md` および `docs/v1.0.0/` ドキュメントの作成

1. **リポジトリルートまたは `SemanticCodec/README.md` の新設**:
   - コンセプト（Foreground Fidelity, Background Generativity）
   - アーキテクチャ図（Mermaid フローチャート：Encoder / Polyglot `.semic.jpg` / Decoder）
   - 開発の歴史と技術的ブレイクスルー（v0.1.0 Canny暴走 → v0.3.0 Depth移行＋VQA前景マスク → v0.4.0 Multi-ControlNet → v1.0.0 BBox空間ビット配分・8bitパレットSeg・LAB色調和・LCM高速化）
   - 先行研究（*Text+Sketch / PICS*, *PerCo*, *OneDC*, *StableCodec*）との技術比較表
   - セットアップ手順・Web UI / CLI の使い方
2. **`docs/PROJECT_RULES.md` に準拠した `docs/v1.0.0/` 4点セットの作成**:
   - `implementation_plan_v1.0.0.md`
   - `task_v1.0.0.md`
   - `specifications_v1.0.0.md`
   - `walkthrough_v1.0.0.md`

---

## 3. 検証・完了条件（Verification Checklist for Opus 5.5）

実装完了後、以下のテストを必ず実行してパスすることを確認せよ：

1. **構文・インポート検証**:
   - `python -m py_compile codec_engine.py app.py cli.py` がエラーゼロで通ること。
2. **ダミー画像による単体パイプライン検証（モデル未ロード・フォールバック含む）**:
   - テスト用画像（例: 1024×1024 RGB画像）を生成し、`Lite` / `Standard` / `HighFi` の各プロファイルでエンコードを実行。
   - 生成された `.semic.jpg` が (a) `PIL.Image.open()` で有効な JPEG として開けること、(b) `zipfile.ZipFile()` で有効な ZIP として開けること（Polyglot 整合性）。
   - `seg_map.png` が `mode == "P"`（8-bit パレット）であり、リサイズ後にパレット外の色が0個であること。
   - `fg_bbox` によるクロップとデコード時の再配置座標が 1px のズレもなく元画像と一致すること。
   - `cli.py inspect` で `bpp` とコンポーネント内訳が正しく出力されること。
3. **Web UI サーバー起動確認**:
   - Flask ルート（`/`, `/api/encode`, `/api/inspect`, `/api/decode`, `/api/progress/<id>`）がすべて正常に応答すること。
