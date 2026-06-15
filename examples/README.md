# Example Workflows

This folder contains two WAN VACE Auto Joiner examples.

## Recommended

### `Wan Vace Auto Joiner WF_new_finalize.json`

Use this workflow for most jobs. It uses:

- `easy forLoopStart`
- `WanVaceAutoJoiner`
- WAN VACE generation and `VAEDecode`
- `WanVaceAutoJoinerSave`
- `easy forLoopEnd`
- `WanVaceAutoJoinerFinalizeVideo`

The finalizer writes the MP4 directly with ffmpeg and avoids returning the
full frame sequence as a ComfyUI `IMAGE` tensor. This is the safe path for
large batches and high-resolution clips.

## Legacy

### `Wan Vace Auto Joiner WF.json`

This is the older small-job example. It uses:

- `WanVaceAutoJoinerFinalize`
- `VHS_VideoCombine`

Keep this workflow only when you specifically need the legacy
`IMAGE`/`AUDIO`/`frame_rate` outputs. It can require very large RAM for long
assemblies.

## How to Use

1. Load `Wan Vace Auto Joiner WF_new_finalize.json` in ComfyUI.
2. Set `Folder` to the directory containing your source clips.
3. Set `Filename Prefix` to the part before `_00001.mp4`.
4. Set `First Filename Suffix` and `Last Filename Suffix`.
5. Set `easy forLoopStart -> total` to `Last Filename Suffix - First Filename Suffix`.
6. Set `Width`, `Height`, and `Frame Rate` to match the source clips.
7. Keep `Length` at `33`.
8. Queue the workflow once.

Example input files:

```text
clip_00001.mp4
clip_00002.mp4
clip_00003.mp4
```

For those files:

```text
Filename Prefix = clip
First Filename Suffix = 1
Last Filename Suffix = 3
easy forLoopStart total = 2
```

## Finalizer Parameters

The new workflow exposes these important `WanVaceAutoJoinerFinalizeVideo`
settings:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `output_prefix` | `wan_join_new_finalize` | Final MP4 filename prefix |
| `cleanup` | `false` | Keep temp PNGs for review unless you are sure |
| `correction_strength` | `0.75` | Overall transition correction strength |
| `luma_strength` | `0.75` | Brightness/luminosity matching strength |
| `chroma_strength` | `0.60` | Color/saturation matching strength |
| `blend_region` | `30` | Context size for diagnostics/previews |
| `anchor_window` | `12` | Frames used as before/after correction anchors |
| `crf` | `12` | H.264 quality |
| `pix_fmt` | `yuv420p` | Compatible MP4 pixel format |
| `transfer_audio` | `true` | Extract and mux original clip audio |

## Required Custom Nodes

- ComfyUI-Easy-Use
- WAN VACE nodes used by the included graph
- This WAN VACE Auto Joiner node pack

ffmpeg must be available for `WanVaceAutoJoinerFinalizeVideo`.
