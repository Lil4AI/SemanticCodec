# Implementation Plan - Improve Style Detection Logic

The current style detection logic fails to correctly identify anime/illustrations when the BLIP model returns a descriptive caption of the subject instead of identifying the artistic medium. This leads to the system defaulting to "photo" and generating incorrect photorealistic prompts for the background reconstruction.

## User Review Required

> [!IMPORTANT]
> The default fallback for unknown styles will be changed to a more conservative evaluation. If the model is inconclusive, we will attempt a more direct "Question-Answer" prompt to force a categorical decision.

## Proposed Changes

### [app.py](file:///c:/Users/Lil4C/.gemini/antigravity/scratch/SemanticCodec/app.py)

#### [MODIFY] [generate_vqa_caption](file:///c:/Users/Lil4C/.gemini/antigravity/scratch/SemanticCodec/app.py#L186-L215)

1.  **Enhance Keyword Lists**: Expand the lists for `photo`, `illustration`, and `painting` to include common descriptive terms used by BLIP.
2.  **Dual-Stage Inference**: 
    *   Stage 1: Keep the current descriptive prefix prompt.
    *   Stage 2 (Fallback): If Stage 1 is inconclusive, use a direct VQA question: `"Question: is this image a photograph, a digital illustration, or a painting? Answer:"`.
3.  **Heuristic Alignment**: If the raw output contains specifically "2D", "manga", "artwork", etc., prioritize the classification.
4.  **Prompt Token Management**: Implement a helper function to truncate or prioritize keywords in the positive prompt to stay under the 77-token CLIP limit.

### [Optimization] GPU & Speed

1.  **CUDA Diagnostics**: Add explicit print statements in `init_models` to show available VRAM and device name.
2.  **VRAM Tuning**: For RTX 2060 Super (8GB), replace/supplement `enable_model_cpu_offload()` with `enable_xformers_memory_efficient_attention()` if available, or `enable_attention_slicing()`.
3.  **Consistent fp16**: Ensure all components (ControlNet, Pipe, Segformer) are explicitly using `torch.float16` to reduce memory bandwidth bottleneck.

## Verification Plan

### Automated Tests
- I will verify the logic using a scratch script that simulates various BLIP outputs.

### Manual Verification
- Ask the user to re-encode the anime image and check the terminal logs for `[VQA Style Match]`. It should now show `Categorized: illustration`.
