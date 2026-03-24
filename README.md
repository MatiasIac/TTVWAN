# Dockerized Wan2.1 Video Generator

Minimal CLI app to generate short MP4 videos from text prompts using `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.

## Build

```bash
docker build -t video-gen .
```

## Run

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/data \
  -v $(pwd)/hf-cache:/cache/huggingface \
  video-gen \
  generate-video \
  --prompt "a rainy street at night, cinematic, neon reflections" \
  --seconds 5 \
  --output /data/out.mp4
```

## CLI options

Global options (place before `generate-video`):

- `--model`
- `--log-level`

Required:

- `--prompt`
- `--output`

Optional:

- `--seconds` (default `5`): target video length in seconds.
- `--negative-prompt`: things you want the model to avoid (for example `blurry, distorted`).
- `--seed`: fixed random seed for repeatable outputs.
- `--width` (default `832`): output width in pixels; must be divisible by `16`.
- `--height` (default `480`): output height in pixels; must be divisible by `16`.
- `--fps` (default `16`): output frame rate for the MP4.
- `--steps` (default `20`): denoising steps; higher is usually slower but can improve detail.
- `--guidance-scale` (default `5.0`): how strongly generation follows the prompt.

## One-line Windows examples per option

`--model`

```powershell
docker run --rm --gpus all -v "$(Get-Location)/data:/data" -v "$(Get-Location)/hf-cache:/cache/huggingface" video-gen --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers generate-video --prompt "rainy neon alley" --output /data/out.mp4
```

```bat
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\hf-cache:/cache/huggingface" video-gen --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers generate-video --prompt "rainy neon alley" --output /data/out.mp4
```

`--log-level`

```powershell
docker run --rm --gpus all -v "$(Get-Location)/data:/data" -v "$(Get-Location)/hf-cache:/cache/huggingface" video-gen --log-level DEBUG generate-video --prompt "rainy neon alley" --output /data/out.mp4
```

```bat
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\hf-cache:/cache/huggingface" video-gen --log-level DEBUG generate-video --prompt "rainy neon alley" --output /data/out.mp4
```

`--prompt`

```powershell
docker run --rm --gpus all -v "$(Get-Location)/data:/data" -v "$(Get-Location)/hf-cache:/cache/huggingface" video-gen generate-video --prompt "a cyberpunk tram stop in the rain" --output /data/out.mp4
```

```bat
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\hf-cache:/cache/huggingface" video-gen generate-video --prompt "a cyberpunk tram stop in the rain" --output /data/out.mp4
```

`--output`

```powershell
docker run --rm --gpus all -v "$(Get-Location)/data:/data" -v "$(Get-Location)/hf-cache:/cache/huggingface" video-gen generate-video --prompt "fog over city rooftops at dawn" --output /data/rooftops.mp4
```

```bat
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\hf-cache:/cache/huggingface" video-gen generate-video --prompt "fog over city rooftops at dawn" --output /data/rooftops.mp4
```

`--seconds`

```powershell
docker run --rm --gpus all -v "$(Get-Location)/data:/data" -v "$(Get-Location)/hf-cache:/cache/huggingface" video-gen generate-video --prompt "storm clouds rolling over mountains" --seconds 8 --output /data/out.mp4
```

```bat
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\hf-cache:/cache/huggingface" video-gen generate-video --prompt "storm clouds rolling over mountains" --seconds 8 --output /data/out.mp4
```

`--negative-prompt`

```powershell
docker run --rm --gpus all -v "$(Get-Location)/data:/data" -v "$(Get-Location)/hf-cache:/cache/huggingface" video-gen generate-video --prompt "street parade at sunset" --negative-prompt "blurry, low quality, distorted" --output /data/out.mp4
```

```bat
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\hf-cache:/cache/huggingface" video-gen generate-video --prompt "street parade at sunset" --negative-prompt "blurry, low quality, distorted" --output /data/out.mp4
```

`--seed`

```powershell
docker run --rm --gpus all -v "$(Get-Location)/data:/data" -v "$(Get-Location)/hf-cache:/cache/huggingface" video-gen generate-video --prompt "snowy village at night" --seed 42 --output /data/out.mp4
```

```bat
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\hf-cache:/cache/huggingface" video-gen generate-video --prompt "snowy village at night" --seed 42 --output /data/out.mp4
```

`--width`

```powershell
docker run --rm --gpus all -v "$(Get-Location)/data:/data" -v "$(Get-Location)/hf-cache:/cache/huggingface" video-gen generate-video --prompt "aerial shot of ocean cliffs" --width 1024 --output /data/out.mp4
```

```bat
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\hf-cache:/cache/huggingface" video-gen generate-video --prompt "aerial shot of ocean cliffs" --width 1024 --output /data/out.mp4
```

`--height`

```powershell
docker run --rm --gpus all -v "$(Get-Location)/data:/data" -v "$(Get-Location)/hf-cache:/cache/huggingface" video-gen generate-video --prompt "aerial shot of ocean cliffs" --height 576 --output /data/out.mp4
```

```bat
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\hf-cache:/cache/huggingface" video-gen generate-video --prompt "aerial shot of ocean cliffs" --height 576 --output /data/out.mp4
```

`--fps`

```powershell
docker run --rm --gpus all -v "$(Get-Location)/data:/data" -v "$(Get-Location)/hf-cache:/cache/huggingface" video-gen generate-video --prompt "wind through tall grass" --fps 24 --output /data/out.mp4
```

```bat
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\hf-cache:/cache/huggingface" video-gen generate-video --prompt "wind through tall grass" --fps 24 --output /data/out.mp4
```

`--steps`

```powershell
docker run --rm --gpus all -v "$(Get-Location)/data:/data" -v "$(Get-Location)/hf-cache:/cache/huggingface" video-gen generate-video --prompt "old train crossing a bridge" --steps 30 --output /data/out.mp4
```

```bat
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\hf-cache:/cache/huggingface" video-gen generate-video --prompt "old train crossing a bridge" --steps 30 --output /data/out.mp4
```

`--guidance-scale`

```powershell
docker run --rm --gpus all -v "$(Get-Location)/data:/data" -v "$(Get-Location)/hf-cache:/cache/huggingface" video-gen generate-video --prompt "desert highway at golden hour" --guidance-scale 6.5 --output /data/out.mp4
```

```bat
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\hf-cache:/cache/huggingface" video-gen generate-video --prompt "desert highway at golden hour" --guidance-scale 6.5 --output /data/out.mp4
```

## Lower width/height examples

Lower resolutions are faster and use less VRAM. Keep both values divisible by `16`.

`640x352` (PowerShell)

```powershell
docker run --rm --gpus all -v "$(Get-Location)/data:/data" -v "$(Get-Location)/hf-cache:/cache/huggingface" video-gen generate-video --prompt "quiet beach sunrise" --width 640 --height 352 --seconds 4 --output /data/out-640x352.mp4
```

`640x352` (Command Prompt)

```bat
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\hf-cache:/cache/huggingface" video-gen generate-video --prompt "quiet beach sunrise" --width 640 --height 352 --seconds 4 --output /data/out-640x352.mp4
```

`512x288` (PowerShell)

```powershell
docker run --rm --gpus all -v "$(Get-Location)/data:/data" -v "$(Get-Location)/hf-cache:/cache/huggingface" video-gen generate-video --prompt "rainy city intersection at night" --width 512 --height 288 --seconds 4 --output /data/out-512x288.mp4
```

`512x288` (Command Prompt)

```bat
docker run --rm --gpus all -v "%cd%\data:/data" -v "%cd%\hf-cache:/cache/huggingface" video-gen generate-video --prompt "rainy city intersection at night" --width 512 --height 288 --seconds 4 --output /data/out-512x288.mp4
```

## Other Sizes

| Resolution  | Relative cost |
| ----------- | ------------- |
| 640×352     | Higher        |
| 512×288     | Medium        |
| 448×256     | Lower         |
| 384×224     | Lower         |
| 320×176     | Minimal       |


## Notes

- GPU is required. If CUDA is unavailable, the CLI exits with a clear error.
- Model weights download automatically on first run and are cached under `/cache/huggingface`.

## To take in consideration

### What to expect

- Short clips (3–5 seconds) work best
- Visual quality is limited by model size (Wan 1.3B)
- Temporal consistency is weak for longer videos
- Best results come from simple prompts and minimal motio