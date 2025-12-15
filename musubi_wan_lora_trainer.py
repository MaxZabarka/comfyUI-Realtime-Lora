"""
Musubi Tuner Wan 2.2 LoRA Trainer Node for ComfyUI

Trains Wan 2.2 LoRAs using kohya-ss/musubi-tuner.
Supports single-frame training with High/Low noise mode selection.
"""

import os
import sys
import json
import hashlib
import tempfile
import shutil
import subprocess
from datetime import datetime
import numpy as np
from PIL import Image

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import requests
import zipfile
import io

import folder_paths

from .musubi_wan_config_template import (
    generate_dataset_config,
    save_config,
    MUSUBI_WAN_VRAM_PRESETS,
    WAN_TIMESTEP_RANGES,
)


# Global config for Musubi Wan trainer
_musubi_wan_config = {}
_musubi_wan_config_file = os.path.join(os.path.dirname(__file__), ".musubi_wan_config.json")

# S3 client globals
_s3_client = None
_s3_bucket_name = None
_s3_initialized = False


def _load_musubi_wan_config():
    """Load Musubi Wan config from disk."""
    global _musubi_wan_config
    if os.path.exists(_musubi_wan_config_file):
        try:
            with open(_musubi_wan_config_file, 'r', encoding='utf-8') as f:
                _musubi_wan_config = json.load(f)
        except:
            _musubi_wan_config = {}


def _save_musubi_wan_config():
    """Save Musubi Wan config to disk."""
    try:
        with open(_musubi_wan_config_file, 'w', encoding='utf-8') as f:
            json.dump(_musubi_wan_config, f, indent=2)
    except:
        pass


def _initialize_s3():
    """Initialize S3 client from .env file. Called once on first use."""
    global _s3_client, _s3_bucket_name, _s3_initialized

    if _s3_initialized:
        return

    # Validate required variables
    access_key = os.environ.get('BUCKET_ACCESS_KEY_ID')
    secret_key = os.environ.get('BUCKET_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('BUCKET_ENDPOINT_URL')
    bucket_name = os.environ.get('BUCKET_NAME')

    # Extract bucket name from endpoint URL and normalize to regional format
    # Boto3 needs regional endpoint (https://s3.region.domain), not bucket-specific (https://bucket.s3.region.domain)
    if endpoint_url:
        from urllib.parse import urlparse
        parsed = urlparse(endpoint_url)
        hostname = parsed.hostname

        if hostname and '.s3.' in hostname:
            parts = hostname.split('.s3.')
            if len(parts) == 2:
                # First part is the bucket name, second part is region.domain
                potential_bucket = parts[0]
                region_domain = parts[1]

                # Only extract bucket if not already set
                if not bucket_name:
                    bucket_name = potential_bucket
                    print(f"[Musubi Wan S3] Extracted bucket name from endpoint: {bucket_name}")

                # Normalize endpoint to regional format (without bucket name)
                endpoint_url = f"{parsed.scheme}://s3.{region_domain}"
                print(f"[Musubi Wan S3] Normalized endpoint to regional format: {endpoint_url}")

    missing = []
    if not access_key:
        missing.append('BUCKET_ACCESS_KEY_ID')
    if not secret_key:
        missing.append('BUCKET_SECRET_ACCESS_KEY')
    if not endpoint_url:
        missing.append('BUCKET_ENDPOINT_URL')
    if not bucket_name:
        missing.append('BUCKET_NAME (or include bucket name in BUCKET_ENDPOINT_URL)')

    if missing:
        raise ValueError(
            f"Missing required S3 credentials: {', '.join(missing)}"
        )

    # Initialize boto3 S3 client
    try:
        _s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            region_name='us-east-005',  # Backblaze B2 region
        )
        _s3_bucket_name = bucket_name
        _s3_initialized = True

        # Test connection with a simple HEAD bucket request
        _s3_client.head_bucket(Bucket=_s3_bucket_name)
        print(f"[Musubi Wan S3] Connected to bucket: {_s3_bucket_name}")

    except NoCredentialsError:
        raise RuntimeError("Invalid S3 credentials. Check your .env file.")
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == '404':
            raise RuntimeError(f"S3 bucket '{bucket_name}' not found")
        elif error_code == '403':
            raise RuntimeError(f"Access denied to S3 bucket '{bucket_name}'")
        else:
            raise RuntimeError(f"S3 connection failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize S3 client: {e}")


def _check_s3_cache(lora_hash):
    """
    Check if a LoRA exists in S3 cache.
    Returns True if object exists, False otherwise.
    """
    _initialize_s3()

    s3_key = f"loras/{lora_hash}.safetensors"

    try:
        _s3_client.head_object(Bucket=_s3_bucket_name, Key=s3_key)
        print(f"[Musubi Wan S3] Cache hit for hash: {lora_hash}")
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == '404':
            print(f"[Musubi Wan S3] Cache miss for hash: {lora_hash}")
            return False
        else:
            # Unexpected error - fail fast
            raise RuntimeError(f"S3 cache check failed: {e}")


def _download_from_s3(lora_hash):
    """
    Download LoRA from S3 to a temporary file.
    Returns the path to the temporary file.
    Caller is responsible for cleanup.
    """
    _initialize_s3()

    s3_key = f"loras/{lora_hash}.safetensors"

    # Create temp file (will be cleaned up by caller)
    temp_fd, temp_path = tempfile.mkstemp(suffix='.safetensors', prefix='musubi_wan_s3_')
    os.close(temp_fd)  # Close file descriptor, we'll write with boto3

    try:
        print(f"[Musubi Wan S3] Downloading {s3_key} to {temp_path}")
        _s3_client.download_file(_s3_bucket_name, s3_key, temp_path)
        print(f"[Musubi Wan S3] Download complete")
        return temp_path
    except ClientError as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise RuntimeError(f"S3 download failed for {s3_key}: {e}")


def _upload_to_s3(local_path, lora_hash):
    """
    Upload trained LoRA to S3.
    Raises RuntimeError if upload fails (CRITICAL - must not silently fail).
    """
    _initialize_s3()

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Cannot upload - file not found: {local_path}")

    s3_key = f"loras/{lora_hash}.safetensors"

    try:
        print(f"[Musubi Wan S3] Uploading {local_path} to {s3_key}")

        # Upload with metadata
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        _s3_client.upload_file(
            local_path,
            _s3_bucket_name,
            s3_key,
            ExtraArgs={
                'Metadata': {
                    'hash': lora_hash,
                    'trained_by': 'comfyui-musubi-wan',
                    'file_size_mb': f'{file_size_mb:.2f}'
                }
            }
        )

        # Verify upload succeeded
        _s3_client.head_object(Bucket=_s3_bucket_name, Key=s3_key)

        print(f"[Musubi Wan S3] Upload complete: {s3_key} ({file_size_mb:.2f} MB)")

        # Generate public URL for logging (if needed)
        s3_url = f"{os.environ.get('BUCKET_ENDPOINT_URL').replace('s3.', '')}/{_s3_bucket_name}/{s3_key}"
        print(f"[Musubi Wan S3] Object URL: {s3_url}")

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        raise RuntimeError(
            f"CRITICAL: S3 upload failed for {s3_key}\n"
            f"Error code: {error_code}\n"
            f"Error message: {e}\n"
            f"Training will abort to prevent data loss."
        )
    except Exception as e:
        raise RuntimeError(f"CRITICAL: Unexpected S3 upload error: {e}")


def _download_and_extract_zip(zip_url, extract_to, default_caption):
    """
    Download a zip file from a public URL and extract images to a directory.
    Returns list of (image_path, caption) tuples.

    Args:
        zip_url: Public URL to zip file
        extract_to: Directory to extract images to
        default_caption: Default caption to use if no .txt file found

    Returns:
        Tuple of (image_paths, captions) - parallel lists
    """
    print(f"[Musubi Wan] Downloading zip from: {zip_url}")

    try:
        # Download zip file
        response = requests.get(zip_url, timeout=300)  # 5 minute timeout
        response.raise_for_status()

        zip_size_mb = len(response.content) / (1024 * 1024)
        print(f"[Musubi Wan] Downloaded {zip_size_mb:.2f} MB")

    except requests.exceptions.Timeout:
        raise RuntimeError(f"Zip download timed out after 5 minutes: {zip_url}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download zip file: {e}")

    # Extract zip
    try:
        zip_buffer = io.BytesIO(response.content)

        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            # Get all files
            all_files = zip_ref.namelist()
            print(f"[Musubi Wan] Zip contains {len(all_files)} files")

            # Extract all (let OS handle duplicates)
            zip_ref.extractall(extract_to)

    except zipfile.BadZipFile:
        raise RuntimeError(f"Invalid zip file from URL: {zip_url}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract zip file: {e}")

    # Scan for images and captions
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    image_paths = []
    captions = []

    for root, dirs, files in os.walk(extract_to):
        for filename in sorted(files):
            if filename.lower().endswith(image_extensions):
                full_path = os.path.join(root, filename)
                image_paths.append(full_path)

                # Look for matching caption file
                base_name = os.path.splitext(filename)[0]
                caption_file = os.path.join(root, f"{base_name}.txt")
                if os.path.exists(caption_file):
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        captions.append(f.read().strip())
                else:
                    captions.append(default_caption)

    if not image_paths:
        raise ValueError(
            f"No valid images found in zip file. "
            f"Expected extensions: {', '.join(image_extensions)}"
        )

    print(f"[Musubi Wan] Found {len(image_paths)} images in zip")
    return image_paths, captions


def _compute_image_hash(images, captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, noise_mode, use_folder_path=False, zip_url=None):
    """Compute a hash of all images, captions, and training parameters."""
    hasher = hashlib.sha256()

    # If zip URL provided, hash it instead of image data
    if zip_url:
        hasher.update(f"zip_url:{zip_url}".encode('utf-8'))
        print(f"[Musubi Wan] Hashing zip URL: {zip_url}")
    elif use_folder_path:
        # For folder paths, hash the file paths and modification times
        for img_path in images:
            hasher.update(img_path.encode('utf-8'))
            if os.path.exists(img_path):
                hasher.update(str(os.path.getmtime(img_path)).encode('utf-8'))
    else:
        # For tensor inputs, hash the image data
        for img_tensor in images:
            img_np = (img_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            img_bytes = img_np.tobytes()
            hasher.update(img_bytes)

    # Include all captions and noise mode in hash
    captions_str = "|".join(captions)
    params_str = f"musubi_wan|{noise_mode}|{captions_str}|{training_steps}|{learning_rate}|{lora_rank}|{vram_mode}|{output_name}|{len(images)}"
    hasher.update(params_str.encode('utf-8'))

    hash_value = hasher.hexdigest()[:16]
    print(f"[Musubi Wan] Computed hash: {hash_value}")
    return hash_value


def _get_venv_python_path(musubi_path):
    """Get the Python path for musubi-tuner venv based on platform.
    Checks both .venv (uv default) and venv (traditional) folders."""
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            python_path = os.path.join(musubi_path, venv_folder, "Scripts", "python.exe")
        else:
            python_path = os.path.join(musubi_path, venv_folder, "bin", "python")

        if os.path.exists(python_path):
            return python_path

    # Return traditional path for error messaging
    if sys.platform == 'win32':
        return os.path.join(musubi_path, "venv", "Scripts", "python.exe")
    else:
        return os.path.join(musubi_path, "venv", "bin", "python")


def _get_accelerate_path(musubi_path):
    """Get the accelerate path for musubi-tuner venv based on platform.
    Checks both .venv (uv default) and venv (traditional) folders."""
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            accel_path = os.path.join(musubi_path, venv_folder, "Scripts", "accelerate.exe")
        else:
            accel_path = os.path.join(musubi_path, venv_folder, "bin", "accelerate")

        if os.path.exists(accel_path):
            return accel_path

    # Return traditional path for error messaging
    if sys.platform == 'win32':
        return os.path.join(musubi_path, "venv", "Scripts", "accelerate.exe")
    else:
        return os.path.join(musubi_path, "venv", "bin", "accelerate")


def _get_model_path(name, folder_type):
    """Get full path to a model file from ComfyUI folders.
    Returns the name as-is if it's already an absolute path that exists."""
    if not name:
        return ""
    # If it's already an absolute path that exists, use it
    if os.path.isabs(name) and os.path.exists(name):
        return name
    # Try to get from ComfyUI folder
    try:
        return folder_paths.get_full_path(folder_type, name)
    except:
        return name


# Load config on module import
_load_musubi_wan_config()


class MusubiWanLoraTrainer:
    """
    Trains a Wan 2.2 LoRA from one or more images using Musubi Tuner.
    Supports single-frame training with High/Low noise mode selection.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Get saved settings or use defaults
        if sys.platform == 'win32':
            musubi_fallback = 'C:\\musubi-tuner'
        else:
            musubi_fallback = '~/musubi-tuner'

        saved = _musubi_wan_config.get('trainer_settings', {})

        # Get available models from ComfyUI folders
        diffusion_models = folder_paths.get_filename_list("diffusion_models")
        vae_models = folder_paths.get_filename_list("vae")
        # Text encoders can be in clip or text_encoders folder
        try:
            text_encoders = folder_paths.get_filename_list("text_encoders")
        except:
            text_encoders = []
        try:
            clip_models = folder_paths.get_filename_list("clip")
        except:
            clip_models = []
        text_encoder_list = sorted(set(text_encoders + clip_models))

        # Get saved model selections (for default)
        saved_dit = saved.get('dit_model', '')
        saved_vae = saved.get('vae_model', '')
        saved_t5 = saved.get('t5_model', '')

        # Build dropdown configs with saved defaults if available
        dit_config = {"tooltip": "Wan 2.2 DiT model. Use High noise model for High Noise training, Low noise model for Low Noise training."}
        if saved_dit and saved_dit in diffusion_models:
            dit_config["default"] = saved_dit

        vae_config = {"tooltip": "Wan VAE model (use Wan2.1 VAE, not Wan2.2_VAE.pth)."}
        if saved_vae and saved_vae in vae_models:
            vae_config["default"] = saved_vae

        t5_config = {"tooltip": "T5 text encoder (models_t5_umt5-xxl-enc-bf16.pth or similar)."}
        if saved_t5 and saved_t5 in text_encoder_list:
            t5_config["default"] = saved_t5

        return {
            "required": {
                "inputcount": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Number of image inputs. Click 'Update inputs' button after changing."}),
                "images_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Path to folder containing training images. If provided, images from this folder are used instead of image inputs. Caption .txt files with matching names are used if present."
                }),
                "zip_url": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Public URL to zip file containing training images. Takes precedence over images_path and image inputs. Caption .txt files with matching names are used if present."
                }),
                "musubi_path": ("STRING", {
                    "default": _musubi_wan_config.get('musubi_path', musubi_fallback),
                    "tooltip": "Path to musubi-tuner installation."
                }),
                "noise_mode": (["High Noise", "Low Noise", "Combo"], {
                    "default": saved.get('noise_mode', "High Noise"),
                    "tooltip": "Training mode. High Noise: use with High model. Low Noise: use with Low model. Combo: full range, use with Low model."
                }),
                "dit_model": (diffusion_models, dit_config),
                "vae_model": (vae_models, vae_config),
                "t5_model": (text_encoder_list, t5_config),
                "caption": ("STRING", {
                    "default": saved.get('caption', "photo of subject"),
                    "multiline": True,
                    "tooltip": "Default caption for all images. Per-image caption inputs override this."
                }),
                "training_steps": ("INT", {
                    "default": saved.get('training_steps', 500),
                    "min": 10,
                    "max": 5000,
                    "step": 10,
                    "tooltip": "Number of training steps. 500 is a good starting point."
                }),
                "learning_rate": ("FLOAT", {
                    "default": saved.get('learning_rate', 0.0003),
                    "min": 0.00001,
                    "max": 0.1,
                    "step": 0.00001,
                    "tooltip": "Learning rate. 3e-4 (0.0003) is recommended for Wan training."
                }),
                "lora_rank": ("INT", {
                    "default": saved.get('lora_rank', 16),
                    "min": 4,
                    "max": 128,
                    "step": 4,
                    "tooltip": "LoRA rank/dimension. 16 is recommended for Wan."
                }),
                "vram_mode": (["Max (1256px)", "Max (1256px) fp8", "Medium (1024px)", "Medium (1024px) fp8", "Low (768px)", "Low (768px) fp8", "Min (512px)", "Min (512px) fp8"], {
                    "default": saved.get('vram_mode', "Low (768px) fp8"),
                    "tooltip": "VRAM optimization preset. Controls resolution, fp8, and gradient checkpointing."
                }),
                "blocks_to_swap": ([str(i) for i in range(40)], {
                    "default": saved.get('blocks_to_swap', "26"),
                    "tooltip": "Number of transformer blocks to offload to CPU (0-39). Higher = less VRAM but slower. 26 is a good balance."
                }),
                "keep_lora": ("BOOLEAN", {
                    "default": saved.get('keep_lora', True),
                    "tooltip": "If True, keeps the trained LoRA file."
                }),
                "output_name": ("STRING", {
                    "default": saved.get('output_name', "MyWanLora"),
                    "tooltip": "Custom name for the output LoRA. Timestamp will be appended."
                }),
                "num_repeats": ("INT", {
                    "default": saved.get('num_repeats', 10),
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of times each image is repeated per epoch. Higher = fewer epochs for same step count. Default is 10."
                }),
                "batch_size": ("INT", {
                    "default": saved.get('batch_size', 1),
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Number of images to process per training step. Higher = faster training but more VRAM. Default is 1."
                }),
                "discrete_flow_shift": ("FLOAT", {
                    "default": saved.get('discrete_flow_shift', 3.0),
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Discrete flow shift parameter for Wan 2.2 training. Controls the noise schedule. Default is 3.0."
                }),
                "timestep_sampling": (["shift", "uniform", "sigmoid", "leading", "trailing"], {
                    "default": saved.get('timestep_sampling', "shift"),
                    "tooltip": "Timestep sampling method. 'shift' is recommended for Wan 2.2. Default is shift."
                }),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Training image (not needed if images_path is set)."}),
                "caption_1": ("STRING", {"forceInput": True, "tooltip": "Caption for image_1. Overrides default caption."}),
                "image_2": ("IMAGE", {"tooltip": "Training image."}),
                "caption_2": ("STRING", {"forceInput": True, "tooltip": "Caption for image_2. Overrides default caption."}),
                "image_3": ("IMAGE", {"tooltip": "Training image."}),
                "caption_3": ("STRING", {"forceInput": True, "tooltip": "Caption for image_3. Overrides default caption."}),
                "image_4": ("IMAGE", {"tooltip": "Training image."}),
                "caption_4": ("STRING", {"forceInput": True, "tooltip": "Caption for image_4. Overrides default caption."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)
    OUTPUT_TOOLTIPS = ("Path to the trained Wan 2.2 LoRA file.",)
    FUNCTION = "train_wan_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "Trains a Wan 2.2 LoRA from images using Musubi Tuner. Supports High/Low noise mode for T2V models."

    def train_wan_lora(
        self,
        inputcount,
        images_path,
        zip_url,
        musubi_path,
        noise_mode,
        dit_model,
        vae_model,
        t5_model,
        caption,
        training_steps,
        learning_rate,
        lora_rank,
        vram_mode,
        blocks_to_swap,
        keep_lora=True,
        output_name="MyWanLora",
        num_repeats=10,
        batch_size=1,
        discrete_flow_shift=3.0,
        timestep_sampling="shift",
        image_1=None,
        **kwargs
    ):
        # Expand paths
        musubi_path = os.path.expanduser(musubi_path.strip())

        # Get full paths from ComfyUI folders
        dit_path = _get_model_path(dit_model, "diffusion_models")
        vae_path = _get_model_path(vae_model, "vae")
        # Try text_encoders first, then clip
        t5_path = _get_model_path(t5_model, "text_encoders")
        if not t5_path or not os.path.exists(t5_path):
            t5_path = _get_model_path(t5_model, "clip")

        # Get timestep range for noise mode
        timestep_config = WAN_TIMESTEP_RANGES.get(noise_mode, WAN_TIMESTEP_RANGES["High Noise"])
        min_timestep = timestep_config["min_timestep"]
        max_timestep = timestep_config["max_timestep"]

        # Determine image source (priority: zip_url > images_path > inputs)
        use_folder_path = False
        use_zip = False
        folder_images = []
        folder_captions = []
        zip_extract_dir = None

        # Priority 1: Zip URL
        if zip_url and zip_url.strip():
            zip_url = zip_url.strip()
            use_zip = True
            print(f"[Musubi Wan] Using zip file from URL: {zip_url}")

            # Create temp directory for extraction
            zip_extract_dir = tempfile.mkdtemp(prefix="comfy_musubi_wan_zip_")

            try:
                folder_images, folder_captions = _download_and_extract_zip(zip_url, zip_extract_dir, caption)
                use_folder_path = True  # Treat like folder input from here on

            except Exception as e:
                # Cleanup on error
                if zip_extract_dir:
                    shutil.rmtree(zip_extract_dir, ignore_errors=True)
                raise RuntimeError(f"Failed to process zip file: {e}")

        # Priority 2: Local folder path
        elif images_path and images_path.strip():
            images_path = os.path.expanduser(images_path.strip())
            if os.path.isdir(images_path):
                # Find all image files in the folder
                image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
                for filename in sorted(os.listdir(images_path)):
                    if filename.lower().endswith(image_extensions):
                        img_path = os.path.join(images_path, filename)
                        folder_images.append(img_path)

                        # Look for matching caption file
                        base_name = os.path.splitext(filename)[0]
                        caption_file = os.path.join(images_path, f"{base_name}.txt")
                        if os.path.exists(caption_file):
                            with open(caption_file, 'r', encoding='utf-8') as f:
                                folder_captions.append(f.read().strip())
                        else:
                            folder_captions.append(caption)  # Use default caption

                if folder_images:
                    use_folder_path = True
                    print(f"[Musubi Wan] Using {len(folder_images)} images from folder: {images_path}")
                else:
                    print(f"[Musubi Wan] No images found in folder: {images_path}, falling back to inputs")
            else:
                print(f"[Musubi Wan] Invalid folder path: {images_path}, falling back to inputs")

        # Priority 3: Tensor inputs (fallback)
        if not use_folder_path:
            # Collect all images and captions from inputs
            all_images = []
            all_captions = []

            if image_1 is not None:
                all_images.append(image_1)
                cap_1 = kwargs.get("caption_1", "")
                all_captions.append(cap_1 if cap_1 else caption)

            for i in range(2, inputcount + 1):
                img = kwargs.get(f"image_{i}")
                if img is not None:
                    all_images.append(img)
                    cap = kwargs.get(f"caption_{i}", "")
                    all_captions.append(cap if cap else caption)

            if not all_images:
                raise ValueError("No images provided. Either set zip_url to a public zip URL, set images_path to a folder containing images, or connect at least one image input.")

        num_images = len(folder_images) if use_folder_path else len(all_images)
        print(f"[Musubi Wan] Training with {num_images} image(s)")
        print(f"[Musubi Wan] Noise Mode: {noise_mode} (timesteps {min_timestep}-{max_timestep})")
        print(f"[Musubi Wan] DiT: {dit_model}")
        print(f"[Musubi Wan] VAE: {vae_model}")
        print(f"[Musubi Wan] T5: {t5_model}")

        # Get VRAM preset settings
        preset = MUSUBI_WAN_VRAM_PRESETS.get(vram_mode, MUSUBI_WAN_VRAM_PRESETS["Low (768px) fp8"])
        blocks_to_swap_int = int(blocks_to_swap)
        print(f"[Musubi Wan] Using VRAM mode: {vram_mode}, blocks_to_swap: {blocks_to_swap_int}")

        # Validate paths
        accelerate_path = _get_accelerate_path(musubi_path)
        train_script = os.path.join(musubi_path, "src", "musubi_tuner", "wan_train_network.py")

        if not os.path.exists(accelerate_path):
            raise FileNotFoundError(f"Musubi Tuner accelerate not found at: {accelerate_path}")
        if not os.path.exists(train_script):
            raise FileNotFoundError(f"wan_train_network.py not found at: {train_script}")
        if not dit_path or not os.path.exists(dit_path):
            raise FileNotFoundError(f"DiT model not found at: {dit_path}")
        if not vae_path or not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE model not found at: {vae_path}")
        if not t5_path or not os.path.exists(t5_path):
            raise FileNotFoundError(f"T5 model not found at: {t5_path}")

        # Save settings
        global _musubi_wan_config
        _musubi_wan_config['musubi_path'] = musubi_path
        _musubi_wan_config['trainer_settings'] = {
            'noise_mode': noise_mode,
            'dit_model': dit_model,
            'vae_model': vae_model,
            't5_model': t5_model,
            'caption': caption,
            'training_steps': training_steps,
            'learning_rate': learning_rate,
            'lora_rank': lora_rank,
            'vram_mode': vram_mode,
            'blocks_to_swap': blocks_to_swap,
            'keep_lora': keep_lora,
            'output_name': output_name,
            'num_repeats': num_repeats,
            'batch_size': batch_size,
            'discrete_flow_shift': discrete_flow_shift,
            'timestep_sampling': timestep_sampling,
        }
        _save_musubi_wan_config()

        # Compute hash for caching (include zip URL if used)
        if use_zip:
            image_hash = _compute_image_hash(
                folder_images, folder_captions, training_steps, learning_rate,
                lora_rank, vram_mode, output_name, noise_mode,
                use_folder_path=True, zip_url=zip_url
            )
        elif use_folder_path:
            image_hash = _compute_image_hash(
                folder_images, folder_captions, training_steps, learning_rate,
                lora_rank, vram_mode, output_name, noise_mode,
                use_folder_path=True
            )
        else:
            image_hash = _compute_image_hash(
                all_images, all_captions, training_steps, learning_rate,
                lora_rank, vram_mode, output_name, noise_mode,
                use_folder_path=False
            )

        # S3 cache check (replaces local JSON cache)
        if _check_s3_cache(image_hash):
            # Download from S3 to temp file
            cached_lora_path = _download_from_s3(image_hash)
            print(f"[Musubi Wan] Using cached LoRA from S3: {image_hash}")

            # Cleanup zip extraction dir if used
            if use_zip and zip_extract_dir:
                shutil.rmtree(zip_extract_dir, ignore_errors=True)

            return (cached_lora_path,)

        # Generate run name with timestamp (for logging only - not used in S3 key)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        noise_suffix = {"High Noise": "high", "Low Noise": "low", "Combo": "combo"}.get(noise_mode, "low")
        run_name = f"{output_name}_{noise_suffix}_{timestamp}" if output_name else f"wan_lora_{noise_suffix}_{image_hash}"

        # CHANGED: Output to temp directory (not permanent musubi output folder)
        # We'll upload to S3 after training, then delete local file
        training_temp_dir = tempfile.mkdtemp(prefix="comfy_musubi_wan_training_")
        lora_output_path = os.path.join(training_temp_dir, f"{run_name}.safetensors")

        # Create temp directory for images
        temp_dir = tempfile.mkdtemp(prefix="comfy_musubi_wan_")
        image_folder = temp_dir  # Musubi uses image_directory directly
        os.makedirs(image_folder, exist_ok=True)

        try:
            # Save images with captions
            if use_folder_path:
                # Copy images from folder and create caption files
                for idx, (src_path, cap) in enumerate(zip(folder_images, folder_captions)):
                    ext = os.path.splitext(src_path)[1]
                    dest_path = os.path.join(image_folder, f"image_{idx+1:03d}{ext}")
                    shutil.copy2(src_path, dest_path)

                    caption_path = os.path.join(image_folder, f"image_{idx+1:03d}.txt")
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(cap)
            else:
                # Save tensor images
                for idx, img_tensor in enumerate(all_images):
                    img_data = img_tensor[0]
                    img_np = (img_data.cpu().numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)

                    image_path = os.path.join(image_folder, f"image_{idx+1:03d}.png")
                    img_pil.save(image_path, "PNG")

                    caption_path = os.path.join(image_folder, f"image_{idx+1:03d}.txt")
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(all_captions[idx])

            print(f"[Musubi Wan] Saved {num_images} images to {image_folder}")

            # Generate dataset config
            config_content = generate_dataset_config(
                image_folder=image_folder,
                resolution=preset['resolution'],
                batch_size=batch_size,
                enable_bucket=True,
                num_repeats=num_repeats,
            )

            config_path = os.path.join(temp_dir, "dataset_config.toml")
            save_config(config_content, config_path)
            print(f"[Musubi Wan] Dataset config saved to {config_path}")

            # Set up subprocess environment
            startupinfo = None
            if sys.platform == 'win32':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            python_path = _get_venv_python_path(musubi_path)

            # Pre-cache latents and text encoder outputs (REQUIRED for Musubi training)
            print(f"[Musubi Wan] Pre-caching latents and text encoder outputs...")

            # Cache latents
            cache_latents_script = os.path.join(musubi_path, "src", "musubi_tuner", "wan_cache_latents.py")
            if not os.path.exists(cache_latents_script):
                raise FileNotFoundError(f"wan_cache_latents.py not found at: {cache_latents_script}")

            print(f"[Musubi Wan] Caching VAE latents...")
            cache_latents_cmd = [
                python_path,
                cache_latents_script,
                f"--dataset_config={config_path}",
                f"--vae={vae_path}",
            ]

            cache_latents_process = subprocess.Popen(
                cache_latents_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=musubi_path,
                startupinfo=startupinfo,
                env=env,
            )

            for line in cache_latents_process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[musubi-tuner] {line}")

            cache_latents_process.wait()
            if cache_latents_process.returncode != 0:
                raise RuntimeError(f"Latent caching failed with code {cache_latents_process.returncode}")

            print(f"[Musubi Wan] VAE latents cached.")

            # Cache text encoder outputs
            cache_te_script = os.path.join(musubi_path, "src", "musubi_tuner", "wan_cache_text_encoder_outputs.py")
            if not os.path.exists(cache_te_script):
                raise FileNotFoundError(f"wan_cache_text_encoder_outputs.py not found at: {cache_te_script}")

            print(f"[Musubi Wan] Caching text encoder outputs...")
            cache_te_cmd = [
                python_path,
                cache_te_script,
                f"--dataset_config={config_path}",
                f"--t5={t5_path}",
                "--batch_size=1",
            ]

            cache_te_process = subprocess.Popen(
                cache_te_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=musubi_path,
                startupinfo=startupinfo,
                env=env,
            )

            for line in cache_te_process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[musubi-tuner] {line}")

            cache_te_process.wait()
            if cache_te_process.returncode != 0:
                raise RuntimeError(f"Text encoder caching failed with code {cache_te_process.returncode}")

            print(f"[Musubi Wan] Text encoder outputs cached.")

            # Build training command
            cmd = [
                accelerate_path,
                "launch",
                "--num_cpu_threads_per_process=1",
                f"--mixed_precision={preset['mixed_precision']}",
                train_script,
                "--task=t2v-A14B",  # Wan 2.2 T2V task
                f"--dit={dit_path}",
                f"--vae={vae_path}",
                f"--t5={t5_path}",
                f"--dataset_config={config_path}",
                "--sdpa",
                f"--mixed_precision={preset['mixed_precision']}",
                f"--timestep_sampling={timestep_sampling}",
                f"--discrete_flow_shift={discrete_flow_shift}",
                f"--min_timestep={min_timestep}",
                f"--max_timestep={max_timestep}",
                f"--optimizer_type={preset['optimizer']}",
                f"--learning_rate={learning_rate}",
                f"--network_module=networks.lora_wan",
                f"--network_dim={lora_rank}",
                f"--network_alpha={lora_rank}",
                f"--max_train_steps={training_steps}",
                "--max_data_loader_n_workers=2",
                "--persistent_data_loader_workers",
                f"--output_dir={training_temp_dir}",
                f"--output_name={run_name}",
                "--seed=42",
            ]

            # Add memory optimization flags
            if preset['gradient_checkpointing']:
                cmd.append("--gradient_checkpointing")

            if preset.get('fp8_base', False):
                cmd.append("--fp8_base")

            if blocks_to_swap_int > 0:
                cmd.append(f"--blocks_to_swap={blocks_to_swap_int}")

            print(f"[Musubi Wan] Starting training: {run_name}")
            print(f"[Musubi Wan] Images: {num_images}, Steps: {training_steps}, LR: {learning_rate}, Rank: {lora_rank}")

            # Run training
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=musubi_path,
                startupinfo=startupinfo,
                env=env,
            )

            # Stream output
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[musubi-tuner] {line}")

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"Musubi Tuner training failed with code {process.returncode}")

            print(f"[Musubi Wan] Training completed!")

            # Find the trained LoRA
            if not os.path.exists(lora_output_path):
                # Check for alternative naming
                possible_files = [f for f in os.listdir(training_temp_dir) if f.endswith('.safetensors')]
                if possible_files:
                    lora_output_path = os.path.join(training_temp_dir, possible_files[-1])
                else:
                    raise FileNotFoundError(f"No LoRA file found in {training_temp_dir}")

            print(f"[Musubi Wan] Found trained LoRA: {lora_output_path}")

            # Upload to S3 (CRITICAL - fail if upload fails)
            _upload_to_s3(lora_output_path, image_hash)

            # Download from S3 to get final path (for consistency)
            # This ensures the returned path is always from S3, not local training output
            final_lora_path = _download_from_s3(image_hash)

            print(f"[Musubi Wan] LoRA cached in S3 with hash: {image_hash}")

            return (final_lora_path,)

        finally:
            # Cleanup ALL temp directories
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                shutil.rmtree(training_temp_dir, ignore_errors=True)
                if use_zip and zip_extract_dir:
                    shutil.rmtree(zip_extract_dir, ignore_errors=True)
            except Exception as e:
                print(f"[Musubi Wan] Warning: Could not clean up temp dirs: {e}")
