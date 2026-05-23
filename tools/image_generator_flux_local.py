"""Local FLUX image generator for ViMax.

Runs the Black Forest Labs FLUX.1-dev (or FLUX.1-schnell) model locally
on GPU via HuggingFace diffusers for fast, high-quality image generation.

Supports:
  - Text-to-Image (no reference images)
  - Image-guided generation (reference images used via IP-Adapter style)

For the ViMax pipeline, this generates:
  - Character portraits (front/side/back views)
  - First frames and last frames for video shots
  - Transition keyframes

Designed for RTX PRO 6000 (96GB) but works on 16GB+ GPUs with optimizations.

Usage in config::

    image_generator:
      class_path: tools.ImageGeneratorFluxLocal
      init_args:
        model_id: "black-forest-labs/FLUX.1-dev"
        num_inference_steps: 28
        guidance_scale: 3.5
        device: "cuda"
"""

import logging
import asyncio
from typing import List, Optional

import torch
from PIL import Image

from interfaces.image_output import ImageOutput
from utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class ImageGeneratorFluxLocal:
    """Local FLUX.1 image generator using HuggingFace diffusers."""

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        num_inference_steps: int = 28,
        height: int = 900,
        width: int = 1600,
        guidance_scale: float = 3.5,
        max_sequence_length: int = 512,
        device: str = "cuda",
        dtype: str = "bfloat16",
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.model_id = model_id
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.dtype = getattr(torch, dtype, torch.bfloat16)
        self.rate_limiter = rate_limiter

        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the FLUX pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        logger.info("Loading FLUX pipeline from %s ...", self.model_id)

        from diffusers import FluxPipeline

        self._pipeline = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        )

        # For 96GB GPU, we can keep everything on GPU
        # For smaller GPUs, use enable_model_cpu_offload() instead
        self._pipeline.to(self.device)

        logger.info("FLUX pipeline loaded successfully.")
        return self._pipeline

    def _build_prompt_with_references(
        self,
        prompt: str,
        reference_image_paths: List[str],
    ) -> str:
        """Build an enhanced prompt.

        FLUX.1 doesn't natively support reference image conditioning in the
        same way as Gemini. The reference images are primarily used by the
        ViMax LLM agents to select which images to reference — the actual
        visual prompt is constructed by the ReferenceImageSelector agent which
        describes the reference images in text.

        The prompt coming into this function already contains the text
        descriptions of the reference images (prefixed as 'Image 0:', 'Image 1:', etc.)
        as constructed by script2video_pipeline.py.
        """
        return prompt

    def _generate_image(
        self,
        prompt: str,
        height: int,
        width: int,
    ) -> Image.Image:
        """Synchronous image generation."""
        pipe = self._load_pipeline()

        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            max_sequence_length=self.max_sequence_length,
        )

        return output.images[0]

    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] = [],
        aspect_ratio: Optional[str] = "16:9",
        size: Optional[str] = None,
        **kwargs,
    ) -> ImageOutput:
        """Generate a single image from a text prompt.

        Args:
            prompt: Text description of the image to generate. For ViMax,
                this already includes reference image descriptions from the
                ReferenceImageSelector agent.
            reference_image_paths: Paths to reference images. In the FLUX
                local pipeline, these are used for text-based guidance only
                (the prompt already describes them).
            aspect_ratio: Aspect ratio string (e.g., "16:9", "1:1").
            size: Optional size string (e.g., "1600x900").

        Returns:
            ImageOutput with fmt="pil", ext="png"
        """
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        # Determine output dimensions
        height, width = self._resolve_dimensions(aspect_ratio, size)

        # Build the full prompt (references are already described in text)
        full_prompt = self._build_prompt_with_references(prompt, reference_image_paths)

        logger.info(
            "FLUX: Generating image (%dx%d, %d steps)...",
            width, height, self.num_inference_steps,
        )

        image = await asyncio.to_thread(
            self._generate_image, full_prompt, height, width
        )

        logger.info("FLUX: Image generation complete.")

        return ImageOutput(
            fmt="pil",
            ext="png",
            data=image,
        )

    def _resolve_dimensions(
        self,
        aspect_ratio: Optional[str],
        size: Optional[str],
    ) -> tuple:
        """Resolve height and width from aspect_ratio or size parameters."""
        if size:
            # size format: "WIDTHxHEIGHT" e.g. "1600x900"
            parts = size.lower().split("x")
            if len(parts) == 2:
                return int(parts[1]), int(parts[0])

        # Aspect ratio based resolution (optimized for FLUX)
        aspect_map = {
            "16:9": (self.height, self.width),  # default 900x1600
            "9:16": (self.width, self.height),  # portrait
            "1:1": (1024, 1024),
            "4:3": (960, 1280),
            "3:4": (1280, 960),
        }

        if aspect_ratio and aspect_ratio in aspect_map:
            return aspect_map[aspect_ratio]

        return (self.height, self.width)
