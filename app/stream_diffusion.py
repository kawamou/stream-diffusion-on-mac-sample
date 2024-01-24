from typing import Literal

import torch

from .utils import StreamDiffusionWrapper


class StreamDiffusion:
    def __init__(
        self,
        model_id_or_path: str = "stabilityai/sd-turbo",
        prompt: str = "1girl with brown dog hair, thick glasses, smiling",
        negative_prompt: str = "low quality, bad quality, blurry, low resolution",
        acceleration: Literal["none", "xformers", "tensorrt"] = "none",
        use_denoising_batch: bool = True,
        use_tiny_vae: bool = True,
        guidance_scale: float = 1.2,
        delta: float = 0.5,
    ) -> None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.stream = StreamDiffusionWrapper(
            model_id_or_path=model_id_or_path,
            t_index_list=[16, 32],
            warmup=10,
            device=device,  # type: ignore
            acceleration=acceleration,
            mode="img2img",
            use_denoising_batch=use_denoising_batch,
            use_tiny_vae=use_tiny_vae,
            cfg_type="self" if guidance_scale > 1.0 else "none",
        )

        self.stream.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=guidance_scale,
            delta=delta,
        )
