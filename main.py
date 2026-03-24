import argparse
import logging
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from PIL import Image

DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 480
DEFAULT_SECONDS = 5
DEFAULT_FPS = 16
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 5.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-gen",
        description="Generate short videos from text prompts with Wan2.1-T2V-1.3B.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("WAN_MODEL_ID", DEFAULT_MODEL_ID),
        help="Hugging Face model ID to load.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate-video",
        help="Generate an MP4 video from a text prompt.",
    )
    generate.add_argument("--prompt", required=True, help="Text prompt for video generation.")
    generate.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt to suppress unwanted content.",
    )
    generate.add_argument(
        "--seconds",
        type=int,
        default=DEFAULT_SECONDS,
        help=f"Video duration in seconds (default: {DEFAULT_SECONDS}).",
    )
    generate.add_argument("--output", required=True, help="Output MP4 path (for example /data/out.mp4).")
    generate.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    generate.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Output width in pixels (default: {DEFAULT_WIDTH}).",
    )
    generate.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Output height in pixels (default: {DEFAULT_HEIGHT}).",
    )
    generate.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Output frame rate (default: {DEFAULT_FPS}).",
    )
    generate.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Diffusion steps (default: {DEFAULT_STEPS}).",
    )
    generate.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE,
        help=f"Classifier-free guidance scale (default: {DEFAULT_GUIDANCE}).",
    )
    generate.set_defaults(handler=handle_generate_video)

    return parser


def validate_generate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.seconds <= 0:
        parser.error("--seconds must be greater than 0.")
    if args.fps <= 0:
        parser.error("--fps must be greater than 0.")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be greater than 0.")
    if args.width % 16 != 0 or args.height % 16 != 0:
        parser.error("--width and --height must be divisible by 16.")
    if args.steps <= 0:
        parser.error("--steps must be greater than 0.")


def ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not detected. Start the container with NVIDIA runtime, for example: "
            "docker run --gpus all ..."
        )
    logging.info("CUDA device: %s", torch.cuda.get_device_name(0))


def resolve_dtype():
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
    if is_bf16_supported():
        logging.info("Using bfloat16 weights.")
        return torch.bfloat16
    logging.info("Using float16 weights.")
    return torch.float16


def create_pipeline(model_id: str, dtype):
    logging.info("Loading model: %s", model_id)
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipeline = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=dtype,
    )

    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing("max")
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()

    if hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cuda")

    return pipeline


def extract_frames(result) -> list:
    frames = getattr(result, "frames", None)
    if frames is None:
        raise RuntimeError("Inference output did not include frames.")
    if isinstance(frames, list) and frames and isinstance(frames[0], list):
        return frames[0]
    return frames


def _to_hwc_uint8(frame) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        array = frame.detach().cpu().numpy()
    else:
        array = np.asarray(frame)

    if array.ndim == 2:
        array = array[..., None]

    if array.ndim != 3:
        raise RuntimeError(f"Unsupported frame rank {array.ndim}; expected 2D or 3D frame.")

    if array.shape[-1] in (1, 2, 3, 4):
        pass
    elif array.shape[0] in (1, 2, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    else:
        raise RuntimeError(
            f"Cannot determine channel axis from frame shape {array.shape}. "
            "Expected channel size in first or last axis."
        )

    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
        min_val = float(array.min())
        max_val = float(array.max())
        if min_val >= 0.0 and max_val <= 1.0:
            array = array * 255.0
        elif min_val >= -1.0 and max_val <= 1.0:
            array = (array + 1.0) * 127.5
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    channels = array.shape[-1]
    if channels == 1:
        array = np.repeat(array, 3, axis=-1)
    elif channels == 2:
        array = np.repeat(array[..., :1], 3, axis=-1)
    elif channels == 4:
        array = array[..., :3]

    return np.ascontiguousarray(array)


def normalize_frames_for_video(frames) -> list[np.ndarray]:
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()

    if isinstance(frames, np.ndarray):
        if frames.ndim == 5:
            frames = frames[0]
        if frames.ndim == 4:
            frame_items = [frames[i] for i in range(frames.shape[0])]
        elif frames.ndim == 3:
            frame_items = [frames]
        else:
            raise RuntimeError(f"Unsupported frame array shape: {frames.shape}")
    else:
        frame_items = list(frames)

    if not frame_items:
        raise RuntimeError("No frames were returned by the model.")

    normalized = [_to_hwc_uint8(frame) for frame in frame_items]
    return normalized


def save_debug_frame(frame: np.ndarray, output_path: Path) -> Path:
    debug_path = output_path.with_name(f"{output_path.stem}.debug-frame.png")
    iio.imwrite(debug_path, frame)
    return debug_path


def to_pil_frames(frames: list[np.ndarray]) -> list[Image.Image]:
    return [Image.fromarray(frame, mode="RGB") for frame in frames]


def handle_generate_video(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    validate_generate_args(args, parser)
    ensure_cuda()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = max(args.seconds * args.fps + 1, args.fps + 1)
    logging.info(
        "Generating %s seconds at %s fps (%s frames), resolution %sx%s.",
        args.seconds,
        args.fps,
        frame_count,
        args.width,
        args.height,
    )

    pipeline = create_pipeline(args.model, resolve_dtype())

    generator = None
    if args.seed is not None:
        logging.info("Using seed: %s", args.seed)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

    logging.info("Running inference...")
    generation_kwargs = {
        "prompt": args.prompt,
        "num_frames": frame_count,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }
    if args.negative_prompt:
        generation_kwargs["negative_prompt"] = args.negative_prompt

    result = pipeline(
        **generation_kwargs,
    )
    frames = normalize_frames_for_video(extract_frames(result))
    logging.info("Prepared %s frames for export. First frame shape: %s", len(frames), frames[0].shape)
    debug_frame_path = save_debug_frame(frames[0], output_path)
    logging.info("Saved debug frame before ffmpeg encode: %s", debug_frame_path)

    logging.info("Writing MP4: %s", output_path)
    export_to_video(to_pil_frames(frames), str(output_path), fps=args.fps, quality=9.0, macro_block_size=None)
    logging.info("Done.")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        return args.handler(args, parser)
    except KeyboardInterrupt:
        logging.error("Interrupted.")
        return 130
    except Exception as exc:
        logging.exception("Generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
