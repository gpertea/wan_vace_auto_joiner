# WAN VACE Auto Joiner for ComfyUI

Join a numbered folder of video clips with WAN VACE transition frames, optional
color/luminosity correction, and audio preservation.

The recommended workflow uses the direct video finalizer:

```text
easy forLoopStart -> WanVaceAutoJoiner -> WAN VACE -> VAEDecode
                  -> WanVaceAutoJoinerSave -> easy forLoopEnd
easy forLoopEnd -> WanVaceAutoJoinerFinalizeVideo -> final MP4
```

For `N` source clips, VACE runs exactly `N - 1` times.

## What Changed

- Added `WanVaceAutoJoinerFinalizeVideo`, the recommended finalizer for long
  assemblies. It writes an MP4 directly with ffmpeg instead of returning every
  frame as a ComfyUI `IMAGE` tensor.
- Added `recover_assembly_video.py` for finishing an interrupted run from the
  temp PNG folder outside ComfyUI.
- Added transition correction controls for overall adaptation, luminosity, and
  chroma/color drift.
- Added a memory guard to the legacy `WanVaceAutoJoinerFinalize` node so it
  fails with a clear message instead of exhausting system RAM.

## Requirements

| Requirement | Notes |
| --- | --- |
| ComfyUI | Base runtime |
| ComfyUI-Easy-Use | Required for `easy forLoopStart` and `easy forLoopEnd` |
| WAN VACE workflow | Your existing VACE model and sampler setup |
| ffmpeg | Required for direct video output and audio transfer |
| scipy | Optional; smoothing falls back to numpy when missing |

## Nodes

Nodes appear under `WAN VACE / Auto Joiner`.

| Class | Display name | Where to place it | Purpose |
| --- | --- | --- | --- |
| `WanVaceAutoJoiner` | WAN VACE Auto Joiner | Inside loop | Creates the next 33-frame VACE control batch and writes source frames to temp PNGs |
| `WanVaceAutoJoinerSave` | WAN VACE Auto Joiner - Save | Inside loop, after `VAEDecode` | Saves each VACE output batch to disk and acts as the loop barrier |
| `WanVaceAutoJoinerFinalizeVideo` | WAN VACE Auto Joiner - Finalize Video | After `easy forLoopEnd` | Recommended finalizer; writes corrected MP4 directly |
| `WanVaceAutoJoinerFinalize` | WAN VACE Auto Joiner - Finalize | After `easy forLoopEnd` | Legacy small-job finalizer; returns `IMAGE`, `AUDIO`, and `frame_rate` for Video Combine |

## Example Workflows

The `examples/` folder contains:

- `Wan Vace Auto Joiner WF_new_finalize.json`: recommended workflow using
  `WanVaceAutoJoinerFinalizeVideo`.
- `Wan Vace Auto Joiner WF_wan22_fun_vace_quality.json`: quality workflow using
  Wan 2.2 Fun VACE high/low-noise models, 20 steps, and no LightX2V LoRA.
- `Wan Vace Auto Joiner WF.json`: legacy workflow using
  `WanVaceAutoJoinerFinalize` and VHS Video Combine.

Use the `_new_finalize` workflow for large batches or high-resolution clips. Use
the Wan 2.2 quality workflow when preserving face texture, identity, and
transition continuity matters more than speed.

## Input File Naming

All clips must be in one folder and use a numeric suffix:

```text
clip_00001.mp4
clip_00002.mp4
clip_00003.mp4
```

The prefix is configurable. For `wan_00001.mp4`, set `Filename Prefix` to
`wan`.

All source clips should have the same width, height, and frame rate.

## Workflow Setup

Use the input settings nodes in the bundled example:

| Setting node | Meaning |
| --- | --- |
| `Folder` | Folder containing numbered source clips |
| `Filename Prefix` | Prefix before `_00001.mp4` |
| `First Filename Suffix` | First clip number, usually `1` |
| `Last Filename Suffix` | Last clip number |
| `Width` / `Height` | Source video dimensions |
| `Length` | VACE batch length; keep at `33` |
| `Frame Rate` | Source/output FPS |

Set `easy forLoopStart -> total` to:

```text
Last Filename Suffix - First Filename Suffix
```

Examples:

| Clips | First | Last | Loop total |
| --- | ---: | ---: | ---: |
| 3 | 1 | 3 | 2 |
| 10 | 1 | 10 | 9 |
| 71 | 1 | 71 | 70 |

The loop wiring must preserve the barrier:

```text
easy forLoopStart index  -> WanVaceAutoJoiner loop_index
easy forLoopStart value1 -> WanVaceAutoJoinerSave value1
WanVaceAutoJoinerSave value1 -> easy forLoopEnd initial_value1
easy forLoopEnd value1 -> WanVaceAutoJoinerFinalizeVideo loop_end_trigger
```

Do not place either finalizer inside the loop.

## Recommended Finalizer

`WanVaceAutoJoinerFinalizeVideo` writes a video directly and is the safest path
for tens of clips.

Important options:

| Option | Default | Notes |
| --- | ---: | --- |
| `output_prefix` | `wanVaceJoined` | Final MP4 prefix in ComfyUI output |
| `cleanup` | `false` | Keep false while testing; true deletes the temp PNG folder after success |
| `correction_strength` | `0.75` | Overall adaptation amount; `0` disables correction |
| `luma_strength` | `0.75` | Brightness/luminosity matching strength |
| `chroma_strength` | `0.60` | Per-channel color/saturation matching strength |
| `blend_region` | `30` | Context frames used for diagnostics and previews |
| `anchor_window` | `12` | Frames before/after transition used as correction anchors |
| `crf` | `12` | H.264 quality; lower is larger/better |
| `pix_fmt` | `yuv420p` | Compatible default |
| `transfer_audio` | `true` | Extracts, concatenates, and muxes source audio |

The finalizer writes intermediate recovery data under the clip folder in a
`recovery-*` directory. Unchanged frames are hardlinked when possible.

## Legacy Finalizer and Memory Use

`WanVaceAutoJoinerFinalize` still exists for small jobs that need the old
`IMAGE -> VHS Video Combine` path. It must load all frames and return a
float32 ComfyUI tensor.

Approximate memory for the final `IMAGE` tensor:

```text
frames * width * height * 3 channels * 4 bytes
```

Example from a 71-clip run:

```text
14,952 frames * 768 * 1168 * 3 * 4 = about 149.9 GiB
```

That is only the output tensor, not counting decoded PNGs, smoothing copies,
models, audio, Python overhead, or ffmpeg. For large jobs, use
`WanVaceAutoJoinerFinalizeVideo` or `recover_assembly_video.py`.

The legacy node has `max_tensor_gb` to stop unsafe runs before they exhaust RAM.

## Color and Luminosity Correction

WAN VACE can shift transition frames in saturation, per-channel balance, and
overall luminosity. This can be more visible when source clips are already
light, muted, or low saturation.

The current correction estimates before/after anchor statistics and adjusts
VACE transition frames toward a smooth target. It does not intentionally alter
the normal source clip frames.

Tuning guidance:

| Symptom | Try |
| --- | --- |
| Transition too dark or too bright | Adjust `luma_strength` first |
| Transition too saturated or color shifted | Adjust `chroma_strength` |
| Correction looks too strong overall | Lower `correction_strength` |
| Drift still visible | Raise `correction_strength` slightly, then luma/chroma |
| Scene lighting changes naturally | Lower strengths to preserve the intended change |

Good starting values:

```text
correction_strength = 0.75
luma_strength = 0.75
chroma_strength = 0.60
blend_region = 30
anchor_window = 12
```

Use the recovery script preview mode to test individual transitions before a
full assembly.

## Recovery Script

If ComfyUI crashes or is stopped after the temp PNGs are written, recover the
MP4 outside ComfyUI:

```bash
python custom_nodes/wan_vace_auto_joiner/recover_assembly_video.py \
  --temp-dir /path/to/clips/temp-YYYYMMDDHHMMSS \
  --source-dir /path/to/clips \
  --file-prefix clip \
  --first-suffix 1 \
  --last-suffix 71 \
  --output /path/to/output/wan_join_recovered.mp4 \
  --crf 12 \
  --pix-fmt yuv420p \
  --correction-strength 0.75 \
  --luma-strength 0.75 \
  --chroma-strength 0.60 \
  --blend-region 30 \
  --overwrite
```

Analyze without encoding:

```bash
python custom_nodes/wan_vace_auto_joiner/recover_assembly_video.py \
  --temp-dir /path/to/clips/temp-YYYYMMDDHHMMSS \
  --analysis-only
```

Preview a single transition:

```bash
python custom_nodes/wan_vace_auto_joiner/recover_assembly_video.py \
  --temp-dir /path/to/clips/temp-YYYYMMDDHHMMSS \
  --transition 12 \
  --preview \
  --analysis-only
```

The script writes JSON and CSV diagnostics for luma, saturation, and RGB drift.
It preserves original PNGs and writes corrected frames into a separate recovery
folder.

## Audio Handling

Both the new finalizer and recovery script:

- extract each source clip to 44.1 kHz stereo PCM WAV,
- pad or trim each clip's audio to its video frame duration,
- concatenate the duration-matched WAV files with ffmpeg,
- encode video as H.264,
- mux AAC audio into the final MP4 with video copy,
- create silent audio only when no usable source audio is found.

This per-clip duration matching prevents small source audio underruns from
accumulating into large sync drift across many joined clips.

To repair audio on an existing recovered/video-only MP4 without reprocessing
PNG frames:

```bash
python custom_nodes/wan_vace_auto_joiner/recover_assembly_video.py \
  --temp-dir /path/to/clips/temp-YYYYMMDDHHMMSS \
  --source-dir /path/to/clips \
  --output /path/to/output/wan_join_audiofixed.mp4 \
  --remux-audio-video /path/to/output/wan_join_recovered-video-only.mp4 \
  --overwrite
```

The legacy finalizer outputs a ComfyUI `AUDIO` object for VHS Video Combine.

## Troubleshooting

### The legacy finalizer says the tensor would exceed `max_tensor_gb`

Use `WanVaceAutoJoinerFinalizeVideo` or the recovery script. Raising the limit
only helps if the machine truly has enough free RAM for the float32 output
tensor plus overhead.

### Finalize runs too early

Connect `easy forLoopEnd value1` to the finalizer's `loop_end_trigger`. Keep
`WanVaceAutoJoinerSave value1` connected to `easy forLoopEnd initial_value1`.

### Loop exits before all VACE batches are saved

Check the `value1` barrier wiring. Do not replace it with FLOW_CONTROL.

### Transitions are still visible

Generate per-transition previews with `recover_assembly_video.py`. Tune
`luma_strength`, `chroma_strength`, and `correction_strength` on a few
representative transitions before assembling the full video.

### No audio appears in the output

Check that source clips have audio and that ffmpeg is available:

```bash
ffmpeg -version
```

## Installation

Manual install:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Rhovanx/wan_vace_auto_joiner.git
```

Restart ComfyUI after installation or updates.

For the local fork, use the repository remote configured in this checkout.

## Changelog

### Current

- Added direct MP4 finalization for large workflows.
- Added recovery script for interrupted PNG assemblies.
- Added luma/chroma/overall correction controls.
- Added tensor memory guard for the legacy finalizer.

### v2.0.0

- Added temporal color smoothing.
- Added per-channel RGB correction.
- Added audio transfer from original clips.
- Added standard ComfyUI `AUDIO` output for Video Combine.

### v1.0.0

- Initial three-node loop system.

## Credits

- WAN VACE: Alibaba's video-to-video consistency model.
- ComfyUI-Easy-Use: loop nodes.
- ComfyUI-VideoHelperSuite: legacy Video Combine compatibility.

## License

MIT
