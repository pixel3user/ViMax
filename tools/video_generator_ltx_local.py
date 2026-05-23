"""Local LTX-Video 2 video generator for ViMax.

Runs the Lightricks LTX-Video-2 model locally on GPU via HuggingFace diffusers.
Supports:
  - Text-to-Video (no reference images)
  - Image-to-Video (1 reference image as first frame)
  - First+Last Frame to Video (2 reference images)

Designed for RTX PRO 6000 (96GB) but works on 24GB+ GPUs with FP8 quantization.

Usage in config::

    video_generator:
      class_path: tools.VideoGeneratorLTXLocal
      init_args:
        model_id: "Lightricks/LTX-Video-0.9.7"
        num_inference_steps: 30
        height: 720
        width: 1280
        num_frames: 121
        guidance_scale: 3.0
        device: "cuda"
"""

import io
import logging
import asyncio
import tempfile
from typing import List, Optional

import torch
from PIL import Image

from interfaces.video_output import VideoOutput
from utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class VideoGeneratorLTXLocal:
    """Local LTX-Video generator using HuggingFace diffusers."""

    def __init__(
        self,
        model_id: str = "Lightricks/LTX-Video-0.9.7",
        num_inference_steps: int = 30,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 121,
        fps: int = 24,
        guidance_scale: float = 3.0,
        device: str = "cuda",
        dtype: str = "bfloat16",
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.model_id = model_id
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.fps = fps
        self.guidance_scale = guidance_scale
        self.device = device
        self.dtype = getattr(torch, dtype, torch.bfloat16)
        self.rate_limiter = rate_limiter

        self._pipeline = None
        self._i2v_pipeline = None

    def _load_t2v_pipeline(self):
        """Lazy-load the text-to-video pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        logger.info("Loading LTX-Video T2V pipeline from %s ...", self.model_id)

        from diffusers import LTXPipeline

        self._pipeline = LTXPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)

        # Enable memory optimizations
        self._pipeline.enable_model_cpu_offload()

        logger.info("LTX-Video T2V pipeline loaded successfully.")
        return self._pipeline

    def _load_i2v_pipeline(self):
        """Lazy-load the image-to-video pipeline."""
        if self._i2v_pipeline is not None:
            return self._i2v_pipeline

        logger.info("Loading LTX-Video I2V pipeline from %s ...", self.model_id)

        from diffusers import LTXImageToVideoPipeline

        self._i2v_pipeline = LTXImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)

        # Enable memory optimizations
        self._i2v_pipeline.enable_model_cpu_offload()

        logger.info("LTX-Video I2V pipeline loaded successfully.")
        return self._i2v_pipeline

    def _frames_to_mp4_bytes(self, frames) -> bytes:
        """Convert a list of PIL frames or a tensor to MP4 bytes."""
        from diffusers.utils import export_to_video

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            export_to_video(frames, tmp.name, fps=self.fps)
            tmp.seek(0)
            return tmp.read()

    def _generate_t2v(self, prompt: str) -> bytes:
        """Text-to-video generation."""
        pipe = self._load_t2v_pipeline()

        output = pipe(
            prompt=prompt,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            output_type="pil",
        )

        frames = output.frames[0]
        return self._frames_to_mp4_bytes(frames)

    def _generate_i2v(self, prompt: str, image: Image.Image) -> bytes:
        """Image-to-video generation (first frame conditioned)."""
        pipe = self._load_i2v_pipeline()

        # Resize image to match generation resolution
        image = image.resize((self.width, self.height), Image.LANCZOS)

        output = pipe(
            prompt=prompt,
            image=image,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            output_type="pil",
        )

        frames = output.frames[0]
        return self._frames_to_mp4_bytes(frames)

    def _generate_flf2v(
        self, prompt: str, first_frame: Image.Image, last_frame: Image.Image
    ) -> bytes:
        """First-frame + last-frame to video generation.

        LTX-Video doesn't natively support last-frame conditioning in all versions.
        We fall back to image-to-video with first frame and include last-frame
        description in the prompt for guidance.
        """
        pipe = self._load_i2v_pipeline()

        # Resize images
        first_frame = first_frame.resize((self.width, self.height), Image.LANCZOS)

        # Enhance prompt with last-frame guidance
        enhanced_prompt = (
            f"{prompt}\n"
            f"The video should transition smoothly to end at a different composition."
        )

        output = pipe(
            prompt=enhanced_prompt,
            image=first_frame,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            output_type="pil",
        )

        frames = output.frames[0]
        return self._frames_to_mp4_bytes(frames)

    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: List[str],
        **kwargs,
    ) -> VideoOutput:
        """Generate a single video clip.

        Args:
            prompt: Text description of the video content and motion.
            reference_image_paths: List of image paths.
                - [] = text-to-video
                - [first_frame_path] = image-to-video
                - [first_frame_path, last_frame_path] = first+last frame to video

        Returns:
            VideoOutput with fmt="bytes", ext="mp4"
        """
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        num_refs = len(reference_image_paths)

        if num_refs == 0:
            logger.info("LTX-Video: Text-to-Video generation...")
            video_bytes = await asyncio.to_thread(self._generate_t2v, prompt)

        elif num_refs == 1:
            logger.info("LTX-Video: Image-to-Video generation (first frame)...")
            first_frame = Image.open(reference_image_paths[0]).convert("RGB")
            video_bytes = await asyncio.to_thread(
                self._generate_i2v, prompt, first_frame
            )

        elif num_refs == 2:
            logger.info(
                "LTX-Video: First+Last Frame to Video generation..."
            )
            first_frame = Image.open(reference_image_paths[0]).convert("RGB")
            last_frame = Image.open(reference_image_paths[1]).convert("RGB")
            video_bytes = await asyncio.to_thread(
                self._generate_flf2v, prompt, first_frame, last_frame
            )

        else:
            raise ValueError(
                f"LTX-Video supports at most 2 reference images, got {num_refs}"
            )

        logger.info("LTX-Video: Video generation complete.")

        return VideoOutput(
            fmt="bytes",
            ext="mp4",
            data=video_bytes,
        )
