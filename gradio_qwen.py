"""
Gradio demo – Qwen-2.5-VL *object-detection only*,
with (1) free-form query, (2) regex bbox parsing,
(3) built-in click-to-run examples, **and**
(4) automatic bbox-to-image rescaling when the
processor downsizes large inputs.

Put the sample images under `gradio_examples/`
or edit the paths in the `examples=` list below.
"""

import re
import cv2
import gradio as gr
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    GenerationConfig,
)
from qwen_vl_utils import process_vision_info

# ----------------------------------------------------------------------
# 0. Image-size limits (same as before)
# ----------------------------------------------------------------------
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

# ----------------------------------------------------------------------
# 1. Load model & processor
# ----------------------------------------------------------------------
MODEL_ID = (
    "/data4/yue/trl_tool_multimodal/output/"
    "h100_dozen_vsr_qwen_add_grounded_thinking_single_turn_think_rethink_150"
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="bfloat16",
    device_map={"": 0},                 # pin to GPU 0
    attn_implementation="flash_attention_2",
).eval()

processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
)

# ----------------------------------------------------------------------
# 2. Prompt helpers
# ----------------------------------------------------------------------
DEFAULT_PROMPT = "Detect all objects in the image."
BBOX_REGEX = re.compile(r"\b\d+,\s*\d+,\s*\d+,\s*\d+\b")

generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.001,
    top_k=1,
    top_p=0.0,
)

PROMPT_SUFFIX = (
    " First, think between <think> and </think> while output necessary "
    "coordinates needed to answer the question in JSON with key 'bbox_2d'. "
    "Then, based on the thinking contents and coordinates, rethink between "
    "<rethink> </rethink> and then answer the question after <answer>.\n"
)

# ----------------------------------------------------------------------
# 3. Inference function
# ----------------------------------------------------------------------
def detect_objects(img_src, user_query: str):
    """Run one round of grounded reasoning + draw boxes."""
    if img_src is None:
        return None, "⚠️ Please upload an image first."

    # img_src is a str path (from gr.Image(type="filepath"))
    image_path = img_src if isinstance(img_src, str) else img_src.name
    if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
        return None, "⚠️ Unsupported image format. Please upload a JPG or PNG file."
    prompt = (user_query or "").strip() or DEFAULT_PROMPT
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Question: {prompt}{PROMPT_SUFFIX}"},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Encode vision
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=img_inputs,
        videos=vid_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # ——— run model ———
    with torch.inference_mode():
        gen_ids = model.generate(**inputs, generation_config=generation_config)

    out_text = processor.batch_decode(
        gen_ids[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # ——— collect bboxes ———
    bboxes = []
    for match in BBOX_REGEX.findall(out_text):
        try:
            x1, y1, x2, y2 = map(int, match.split(","))
            bboxes.append((x1, y1, x2, y2))
        except ValueError:
            pass  # ignore malformed spans

    # ——— load original image ———
    img_bgr = cv2.imread(image_path)

    # If the processor shrank the image, rescale coords
    # ------------------------------------------------------------------
    proc_h, proc_w = None, None
    if "image_grid_thw" in inputs:
        proc_h, proc_w = inputs['image_grid_thw'][0][-2:]
        proc_h, proc_w = proc_h.item()*14, proc_w.item()*14
    elif "images" in inputs:  # some processor variants use this key
        _, _, proc_h, proc_w = inputs["images"].shape

    if proc_h and proc_w:
        orig_h, orig_w = img_bgr.shape[:2]
        w_scale, h_scale = orig_w / proc_w, orig_h / proc_h
    else:  # fall back (should rarely happen)
        w_scale = h_scale = 1.0

    # ——— draw ———
    if not bboxes:
        annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return annotated, out_text

    for (x1, y1, x2, y2) in bboxes:
        x1 = int(round(x1 * w_scale))
        y1 = int(round(y1 * h_scale))
        x2 = int(round(x2 * w_scale))
        y2 = int(round(y2 * h_scale))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

    annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return annotated, out_text


# ----------------------------------------------------------------------
# 4. Gradio interface
# ----------------------------------------------------------------------
with gr.Blocks(title="Grounded Reasoning with Texts and Images (GRIT)") as demo:
    gr.Markdown(
        "## GRIT-Qwen 2.5-VL (3 B) Demo\n"
        "The model is trained with **GRIT** on only 20 VSR/TallyQA samples. "
        "See the [project page](https://grounded-reasoning.github.io) for details."
    )

    with gr.Row():
        img_input = gr.Image(
            label="Upload image",
            type="filepath",           # returns str path
            sources=["upload", "clipboard"],
        )
        query_box = gr.Textbox(
            label="Input query (leave blank for default)",
            placeholder=DEFAULT_PROMPT,
        )
        run_btn = gr.Button("Run")

    img_output = gr.Image(label="Annotated image")
    raw_output = gr.Textbox(label="Raw model output")

    run_btn.click(
        fn=detect_objects,
        inputs=[img_input, query_box],
        outputs=[img_output, raw_output],
    )

    gr.Examples(
        examples=[
            # ["gradio_examples/eggs_small.png", "How many eggs are inside the nest?"],
            ["gradio_examples/books.png", "Are all three books together?"],
            ["gradio_examples/eg3.jpg", "Is there a knife in the image?"],
            ["gradio_examples/000000072535.jpg", "Is the truck beneath the cat?"],
        ],
        inputs=[img_input, query_box],
        label="Click an example ⬇",
    )

# ----------------------------------------------------------------------
# 5. Launch
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(share=True)  # set share=True if you want a public link
