import os
import shutil
import random
import json
from PIL import Image, ExifTags
from typing import List
from cog import BasePredictor, Input, Path
from helpers.comfyui import ComfyUI

# === SETTINGS AND PARAMETERS ===
OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"

with open("face-to-many-api.json", "r") as file:
    WORKFLOW_TEMPLATE = json.loads(file.read())

LORA_WEIGHTS_MAPPING = {
    "3D": "artificialguybr/3DRedmond-3DRenderStyle-3DRenderAF.safetensors",
    "Emoji": "fofr/emoji.safetensors",
    "Video game": "artificialguybr/PS1Redmond-PS1Game-Playstation1Graphics.safetensors",
    "Pixels": "artificialguybr/PixelArtRedmond-Lite64.safetensors",
    "Clay": "artificialguybr/ClayAnimationRedm.safetensors",
    "Toy": "artificialguybr/ToyRedmond-FnkRedmAF.safetensors",
}

LORA_TYPES = list(LORA_WEIGHTS_MAPPING.keys())

class Predictor(BasePredictor):
    def setup(self):
        # Setup specific to ComfyUI and weight management
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)
        self.comfyUI.load_workflow(json.dumps(WORKFLOW_TEMPLATE), check_inputs=False)
        self.download_loras()

    def download_loras(self):
        """Download predefined LoRA weights."""
        for weight in LORA_WEIGHTS_MAPPING.values():
            self.comfyUI.weights_downloader.download_weights(weight)

    def style_to_prompt(self, style: str, prompt: str) -> str:
        """
        Generate a positive prompt based on style, including LoRA-specific keyword triggers.

        :param style: The style to apply (e.g., "3D", "Emoji").
        :param prompt: The main prompt describing the subject.
        :return: A detailed positive prompt tailored to the style.
        """
        style_prompts = {
            "3D": f"Highly detailed 3D Render, cinematic lighting, realistic materials, {prompt}, "
                f"3DRenderAF, rendered with global illumination, physically-based rendering (PBR), sharp focus",
            "Emoji": f"High-quality emoji style, memoji aesthetic, smooth textures, vibrant colours, {prompt}, "
                    f"memoji, expressive, designed for digital stickers",
            "Video game": f"Playstation 1 graphics style, retro video game aesthetic, pixelated textures, "
                        f"low-poly modelling, {prompt}, PS1Game, nostalgic and immersive environment, "
                        f"rendered with vintage CRT filter",
            "Pixels": f"Pixel art style, crisp 8-bit visuals, {prompt}, PixArFK, retro game sprite aesthetic, "
                    f"vivid colours, sharp edges, designed for side-scrolling games",
            "Clay": f"Clay animation style, handcrafted appearance, soft and tactile textures, {prompt}, "
                    f"Clay, stop-motion feel, realistic lighting for clay surfaces",
            "Toy": f"Miniature toy style, highly detailed, {prompt}, FnkRedmAF, vibrant and playful colours, "
                f"realistic plastic or wooden textures, childlike charm, studio lighting",
        }
        return style_prompts.get(style, f"{prompt}")  # Fallback to the base prompt if the style is not found


    def style_to_negative_prompt(self, style: str, negative_prompt: str = "") -> str:
        """
        Generate a negative prompt based on style, including LoRA-specific exclusions.

        :param style: The style to apply (e.g., "3D", "Emoji").
        :param negative_prompt: Additional user-provided undesirable elements.
        :return: A detailed negative prompt tailored to the style.
        """
        if negative_prompt:
            negative_prompt = f"{negative_prompt}, "

        base_negative = "blurred, overexposed, nsfw, nude, distorted, poorly lit, "
        end_negative = "ugly, broken, low-quality, watermark, text artifacts, jpeg artifacts"
        specifics = {
            "3D": "grainy textures, low-poly, flat shading, non-PBR materials, ",
            "Emoji": "dull colours, pixelated edges, jagged lines, unsaturated, blurry textures, ",
            "Video game": "modern graphics, hyper-realism, photorealistic textures, oversaturated colours, ",
            "Pixels": "blurry pixels, large gradients, anti-aliasing, photographic elements, ",
            "Clay": "digital appearance, overly smooth surfaces, flat colours, ",
            "Toy": "lifeless appearance, unpainted surfaces, dull lighting, ",
        }

        # Combine specifics, LoRA-related exclusions, and generic negatives
        return f"{specifics.get(style, '')}{base_negative}{negative_prompt}{end_negative}"

    def update_workflow(self, workflow, **kwargs):
        """Update workflow JSON with parameters."""
        style = kwargs["style"]
        prompt = kwargs["prompt"]
        negative_prompt = kwargs["negative_prompt"]
        custom_style = kwargs["lora_url"]

        if custom_style:
            uuid = self.parse_custom_lora_url(custom_style)
            lora_name = f"{uuid}/{uuid}.safetensors"
        else:
            lora_name = LORA_WEIGHTS_MAPPING[style]
            prompt = self.style_to_prompt(style, prompt)
            negative_prompt = self.style_to_negative_prompt(style, negative_prompt)

        workflow["22"]["inputs"]["image"] = kwargs["filename"]
        workflow["2"]["inputs"]["positive"] = prompt
        workflow["2"]["inputs"]["negative"] = negative_prompt
        workflow["28"]["inputs"]["strength"] = kwargs["control_depth_strength"]
        workflow["3"]["inputs"]["lora_name_1"] = lora_name
        workflow["3"]["inputs"]["lora_wt_1"] = kwargs["lora_scale"]
        workflow["41"]["inputs"]["weight"] = kwargs["instant_id_strength"]
        workflow["4"]["inputs"]["denoise"] = kwargs["denoising_strength"]
        workflow["4"]["inputs"]["seed"] = kwargs["seed"]
        workflow["4"]["inputs"]["cfg"] = kwargs["prompt_strength"]

# === BASE FUNCTIONS ===

    def parse_custom_lora_url(self, url: str):
        """Parse custom LoRA URL to extract UUID."""
        if "pbxt.replicate" in url:
            parts_after_pbxt = url.split("/pbxt.replicate.delivery/")[1]
        else:
            parts_after_pbxt = url.split("/pbxt/")[1]
        return parts_after_pbxt.split("/trained_model.tar")[0]

    def add_to_lora_map(self, lora_url: str):
        uuid = self.parse_custom_lora_url(lora_url)
        self.comfyUI.weights_downloader.download_lora_from_replicate_url(uuid, lora_url)

    def cleanup(self):
        """Clean up temporary directories."""
        self.comfyUI.clear_queue()
        for directory in [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def log_and_collect_files(self, directory, prefix=""):
        """Log and collect output files."""
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX":
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(Path(path))
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

    def predict(
        self,
        image: Path = Input(
            description="An image of a person to be converted",
            default=None,
        ),
        style: str = Input(
            default="3D",
            choices=LORA_TYPES,
            description="Style to convert to",
        ),
        prompt: str = Input(default="a person"),
        negative_prompt: str = Input(
            default="",
            description="Things you do not want in the image",
        ),
        denoising_strength: float = Input(
            default=0.65,
            ge=0,
            le=1,
            description="How much of the original image to keep. 1 is the complete destruction of the original image, 0 is the original image",
        ),
        prompt_strength: float = Input(
            default=4.5,
            ge=0,
            le=20,
            description="Strength of the prompt. This is the CFG scale, higher numbers lead to stronger prompt, lower numbers will keep more of a likeness to the original.",
        ),
        control_depth_strength: float = Input(
            default=0.8,
            ge=0,
            le=1,
            description="Strength of depth controlnet. The bigger this is, the more controlnet affects the output.",
        ),
        instant_id_strength: float = Input(
            default=1, description="How strong the InstantID will be.", ge=0, le=1
        ),
        seed: int = Input(
            default=None, description="Fix the random seed for reproducibility"
        ),
        custom_lora_url: str = Input(
            default=None,
            description="URL to a Replicate custom LoRA. Must be in the format https://replicate.delivery/pbxt/[id]/trained_model.tar or https://pbxt.replicate.delivery/[id]/trained_model.tar",
        ),
        lora_scale: float = Input(
            default=1, description="How strong the LoRA will be", ge=0, le=1
        ),
    ) -> List[Path]:
        """Run a single prediction on the model."""
        self.cleanup()

        if image is None:
            raise ValueError("No image provided")
        
        input_processor = InputProcessor(INPUT_DIR)

        filename = input_processor.process_image(
            input_file=image,
            rules={
                "rotate_based_on_exif": True,
                "convert_format": "png"
            }
        )

        if custom_lora_url:
            self.add_to_lora_map(custom_lora_url)

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Random seed set to: {seed}")

        workflow = WORKFLOW_TEMPLATE.copy()
        self.update_workflow(
            workflow,
            filename=filename,
            style=style,
            denoising_strength=denoising_strength,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_strength=prompt_strength,
            instant_id_strength=instant_id_strength,
            lora_url=custom_lora_url,
            lora_scale=lora_scale,
            control_depth_strength=control_depth_strength,
        )

        wf = self.comfyUI.load_workflow(workflow, check_weights=False)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        files = self.log_and_collect_files(OUTPUT_DIR)
        return files
    

class InputProcessor:
    def __init__(self, input_dir: str):
        """
        Initialize the InputProcessor with a target directory for processed files.

        :param input_dir: Directory where processed files will be saved.
        """
        self.input_dir = input_dir
        os.makedirs(self.input_dir, exist_ok=True)

    def process_image(self, input_file: Path, rules: dict = None) -> str:
        """
        Process an input image file according to specified rules.

        :param input_file: The path to the input image file.
        :param rules: A dictionary specifying rules for processing. Example:
                      {
                          "rotate_based_on_exif": True,
                          "convert_format": "png"
                      }
        :return: The filename of the processed image in the input directory.
        """
        rules = rules or {}
        file_extension = os.path.splitext(input_file)[1].lower()

        # Define default filename
        default_filename = "input.png"
        output_filename = f"input{file_extension}" if file_extension in [".png", ".webp", ".gif"] else default_filename
        output_path = os.path.join(self.input_dir, output_filename)

        # Open the image
        image = Image.open(input_file)

        # Apply rotation based on EXIF data
        if rules.get("rotate_based_on_exif", False):
            image = self._rotate_image_based_on_exif(image)

        # Convert format if specified
        if rules.get("convert_format") and rules["convert_format"] != file_extension:
            output_filename = f"input.{rules['convert_format']}"
            output_path = os.path.join(self.input_dir, output_filename)
            image = image.convert("RGB")  # Ensure compatibility for formats like JPEG

        # Save the processed image
        image.save(output_path)

        return output_filename

    def _rotate_image_based_on_exif(self, image: Image.Image) -> Image.Image:
        """
        Rotate the image based on EXIF orientation data, if available.

        :param image: The image to be rotated.
        :return: The rotated image.
        """
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break

            exif = image._getexif()
            if exif:
                orientation = exif.get(orientation)
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        except (KeyError, AttributeError, TypeError):
            # No EXIF data or no orientation tag
            pass

        return image

    def clean_directory(self):
        """
        Clean up the input directory by removing all files.
        """
        shutil.rmtree(self.input_dir)
        os.makedirs(self.input_dir, exist_ok=True)
