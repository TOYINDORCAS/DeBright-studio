import uuid
import os
import torch

from datetime import datetime
from PIL import Image

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File

from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    StableDiffusionUpscalePipeline,
)
from diffusers import StableDiffusionImg2ImgPipeline

# Initialize FastAPI
app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model + VAE
model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=dtype,
)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
)

# Better scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++",
)

pipe = pipe.to(device)

# Memory optimizations
pipe.enable_attention_slicing()

try:
    pipe.enable_xformers_memory_efficient_attention()
    print("xformers enabled")
except Exception:
    print("xformers not available — continuing without it")

# Upscaler
upscaler = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    torch_dtype=dtype,
).to(device)

# Create folders
os.makedirs("static", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


def upscale_image(image: Image.Image) -> Image.Image:
    return upscaler(prompt="", image=image).images[0]


def apply_lora(pipe, lora_path: str):
    pipe.load_lora_weights(lora_path)
    return pipe


# Negative prompt
negative_prompt = """
(deformed iris, deformed pupils:1.4),
(semi-realistic, cgi, 3d, render, cartoon, anime, sketch:1.5),
(wrong ethnicity, light skin on lecturer, european features:1.6),
(plastic skin, waxy skin, unnatural skin tone:1.4),
(extra fingers, mutated hands, poorly drawn hands:1.4),
(poorly drawn face, bad anatomy, bad proportions:1.3),
(blurry faces, blurry students, hidden students:1.5),
(empty classroom, no students, students not visible:1.6),
(students backs only, students facing away:1.5),
(blurry, low quality, worst quality, jpeg artifacts:1.3),
(watermark, signature, text overlay, logo:1.2),
(exposed skin, cleavage, bare shoulders:1.8),
(tight clothing, revealing outfit, sleeveless:1.8),
(casual clothing on lecturer, jeans, t-shirt:1.5),
(sexy, sensual, provocative pose:1.9),
one student only, two students only, empty seats,
closed eyes, looking away, no eye contact,
gibberish whiteboard text, incorrect equations,
missing fingers, fused fingers, floating limbs,
dark classroom, bad lighting, overexposed
"""


# Homepage
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "lecturer_studio.html",
        {
            "request": request,
            "studio_name": "DeBright Studio",
        },
    )


# Generate route
@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    text: str = Form(...),
    upscale: str = Form(None),
    lora: str = Form(None),
    outfit: str = Form(None),
    personal_image: UploadFile = File(None),
):

    global pipe

    num_inference_steps = 30
    guidance_scale = 7.5
    width = 512
    height = 512

    # Add outfit to prompt
    if outfit and outfit.strip() != "":
        text = f"{text}, wearing {outfit}"

    # Apply LoRA
    if lora and lora.strip() != "":
        pipe = apply_lora(pipe, f"weights/{lora}")

    # If personal image uploaded
    if personal_image:

        upload_filename = (
            f"static/{uuid.uuid4().hex}_{personal_image.filename}"
        )

        with open(upload_filename, "wb") as f:
            f.write(await personal_image.read())

        base_img = Image.open(upload_filename).convert("RGB")

        # Image-to-image generation
        image = pipe(
            prompt=text,
            image=base_img,
            guidance_scale=guidance_scale,
        ).images[0]

    else:
        # Text-to-image generation
        image = pipe(
            prompt=text,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        ).images[0]

    # Upscale image
    if upscale == "true":
        image = upscale_image(image)

    # Save image
    unique_filename = f"{uuid.uuid4().hex}.png"
    save_path = f"static/{unique_filename}"

    image.save(save_path)

    # Return result
    return templates.TemplateResponse(
        "lecturer_studio.html",
        {
            "request": request,
            "image_url": f"/static/{unique_filename}",
            "prompt": text,
            "studio_name": "DeBright Studio",
        },
    )