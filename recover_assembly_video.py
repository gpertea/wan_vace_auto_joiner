#!/usr/bin/env python3
"""
Recover a WAN VACE Auto Joiner run from an existing temp PNG folder.

This script is intentionally independent from ComfyUI execution. It preserves
the original PNG sequence, writes corrected frames into a separate recovery
sequence, encodes the video with ffmpeg, and muxes source audio back in.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


TOTAL_BATCH = 33
DEFAULT_CRf = 12
DEFAULT_PIX_FMT = "yuv420p"


def log(message: str) -> None:
    print(f"[recover_assembly_video] {message}", flush=True)


def run_command(args: Sequence[str], *, input_data: Optional[bytes] = None) -> subprocess.CompletedProcess:
    result = subprocess.run(args, input=input_data, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(args)}\n{stderr}")
    return result


def ffprobe_json(path: Path, entries: str) -> Dict[str, Any]:
    result = run_command([
        "ffprobe",
        "-v", "error",
        "-show_entries", entries,
        "-of", "json",
        str(path),
    ])
    return json.loads(result.stdout.decode("utf-8"))


def load_state(temp_dir: Path) -> Dict[str, Any]:
    state_path = temp_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state file: {state_path}")
    with state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_prefix(prefix: str) -> str:
    cleaned = prefix.replace("/", "").replace("\\", "").replace("..", "")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch in "_-")
    if not cleaned:
        raise ValueError("File prefix is empty after sanitization")
    return cleaned


def frame_path(folder: Path, prefix: str, frame_index: int) -> Path:
    return folder / f"{prefix}_{frame_index + 1:05d}.png"


def sorted_frame_files(temp_dir: Path, prefix: str) -> List[Path]:
    files = sorted(temp_dir.glob(f"{prefix}_*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG frames found in {temp_dir} for prefix {prefix!r}")
    return files


def dedupe_regions(regions: Iterable[Dict[str, int]], frame_count: int) -> List[Dict[str, int]]:
    seen: set[Tuple[int, int]] = set()
    clean: List[Dict[str, int]] = []
    for region in regions:
        start = int(region.get("start", -1))
        end = int(region.get("end", -1))
        if start < 0 or end <= start:
            continue
        start = max(0, min(start, frame_count))
        end = max(0, min(end, frame_count))
        if end <= start:
            continue
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        clean.append({"start": start, "end": end})
    clean.sort(key=lambda item: (item["start"], item["end"]))
    return clean


def selected_regions(regions: List[Dict[str, int]], selected: Optional[Sequence[int]]) -> List[Tuple[int, Dict[str, int]]]:
    if not selected:
        return list(enumerate(regions, start=1))
    output: List[Tuple[int, Dict[str, int]]] = []
    for index in selected:
        if index < 1 or index > len(regions):
            raise ValueError(f"Transition {index} is outside valid range 1..{len(regions)}")
        output.append((index, regions[index - 1]))
    return output


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, frame: np.ndarray) -> None:
    Image.fromarray(frame.astype(np.uint8), "RGB").save(path)


def frame_stats(frame: np.ndarray) -> Dict[str, Any]:
    data = frame.astype(np.float32)
    rgb = data.reshape(-1, 3).mean(axis=0)
    luma_plane = data[:, :, 0] * 0.2126 + data[:, :, 1] * 0.7152 + data[:, :, 2] * 0.0722
    maxc = data.max(axis=2)
    minc = data.min(axis=2)
    saturation = np.divide(maxc - minc, np.maximum(maxc, 1.0)).mean()
    return {
        "rgb_mean": [float(rgb[0]), float(rgb[1]), float(rgb[2])],
        "luma": float(luma_plane.mean()),
        "saturation": float(saturation),
    }


def average_stats(paths: Sequence[Path]) -> Dict[str, Any]:
    if not paths:
        raise ValueError("Cannot calculate average stats with no frames")
    stats = [frame_stats(read_rgb(path)) for path in paths]
    rgb = np.array([item["rgb_mean"] for item in stats], dtype=np.float64).mean(axis=0)
    return {
        "rgb_mean": [float(rgb[0]), float(rgb[1]), float(rgb[2])],
        "luma": float(np.mean([item["luma"] for item in stats])),
        "saturation": float(np.mean([item["saturation"] for item in stats])),
    }


def interpolate(before: np.ndarray, after: np.ndarray, progress: float) -> np.ndarray:
    return before * (1.0 - progress) + after * progress


def calculate_gain(
    current_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    correction_strength: float,
    luma_strength: float,
    chroma_strength: float,
    min_gain: float,
    max_gain: float,
) -> np.ndarray:
    current_rgb = np.maximum(current_rgb, 1.0)
    target_rgb = np.maximum(target_rgb, 1.0)
    channel_gain = target_rgb / current_rgb

    current_luma = float(current_rgb @ np.array([0.2126, 0.7152, 0.0722]))
    target_luma = float(target_rgb @ np.array([0.2126, 0.7152, 0.0722]))
    luma_gain = target_luma / max(current_luma, 1.0)
    chroma_gain = channel_gain / max(luma_gain, 1e-6)

    combined = (1.0 + luma_strength * (luma_gain - 1.0)) * (
        1.0 + chroma_strength * (chroma_gain - 1.0)
    )
    gain = 1.0 + correction_strength * (combined - 1.0)
    return np.clip(gain, min_gain, max_gain).astype(np.float32)


def correction_for_region(
    frame_files: Sequence[Path],
    region: Dict[str, int],
    *,
    anchor_window: int,
    correction_strength: float,
    luma_strength: float,
    chroma_strength: float,
    min_gain: float,
    max_gain: float,
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    start = region["start"]
    end = region["end"]
    before_paths = list(frame_files[max(0, start - anchor_window):start])
    after_paths = list(frame_files[end:min(len(frame_files), end + anchor_window)])
    if not before_paths or not after_paths:
        return {}, {
            "start": start,
            "end": end,
            "skipped": True,
            "reason": "missing before or after anchor frames",
        }

    before_stats = average_stats(before_paths)
    after_stats = average_stats(after_paths)
    before_rgb = np.array(before_stats["rgb_mean"], dtype=np.float32)
    after_rgb = np.array(after_stats["rgb_mean"], dtype=np.float32)

    corrected: Dict[int, np.ndarray] = {}
    frame_diagnostics: List[Dict[str, Any]] = []
    span = max(1, end - start - 1)

    for frame_index in range(start, end):
        frame = read_rgb(frame_files[frame_index])
        current_stats = frame_stats(frame)
        current_rgb = np.array(current_stats["rgb_mean"], dtype=np.float32)
        progress = (frame_index - start) / span
        target_rgb = interpolate(before_rgb, after_rgb, progress)
        gain = calculate_gain(
            current_rgb,
            target_rgb,
            correction_strength=correction_strength,
            luma_strength=luma_strength,
            chroma_strength=chroma_strength,
            min_gain=min_gain,
            max_gain=max_gain,
        )
        corrected_frame = np.clip(frame.astype(np.float32) * gain.reshape(1, 1, 3), 0, 255).astype(np.uint8)
        corrected[frame_index] = corrected_frame

        after_frame_stats = frame_stats(corrected_frame)
        frame_diagnostics.append({
            "frame_index": frame_index,
            "frame_number": frame_index + 1,
            "progress": float(progress),
            "current_rgb": current_stats["rgb_mean"],
            "target_rgb": [float(value) for value in target_rgb],
            "gain": [float(value) for value in gain],
            "before_luma": current_stats["luma"],
            "after_luma": after_frame_stats["luma"],
            "before_saturation": current_stats["saturation"],
            "after_saturation": after_frame_stats["saturation"],
        })

    diagnostics = {
        "start": start,
        "end": end,
        "skipped": False,
        "before_anchor": before_stats,
        "after_anchor": after_stats,
        "frames": frame_diagnostics,
    }
    return corrected, diagnostics


def analyze_regions(
    frame_files: Sequence[Path],
    regions: Sequence[Tuple[int, Dict[str, int]]],
    *,
    anchor_window: int,
) -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    for ordinal, (transition_index, region) in enumerate(regions, start=1):
        if ordinal == 1 or ordinal == len(regions) or ordinal % 10 == 0:
            log(f"Analyzing transition {transition_index} ({ordinal}/{len(regions)})")
        start = region["start"]
        end = region["end"]
        before_paths = list(frame_files[max(0, start - anchor_window):start])
        vace_paths = list(frame_files[start:end])
        after_paths = list(frame_files[end:min(len(frame_files), end + anchor_window)])
        item: Dict[str, Any] = {
            "transition": transition_index,
            "start": start,
            "end": end,
            "frame_count": end - start,
        }
        if before_paths:
            item["before_anchor"] = average_stats(before_paths)
        if vace_paths:
            item["vace_region"] = average_stats(vace_paths)
        if after_paths:
            item["after_anchor"] = average_stats(after_paths)
        diagnostics.append(item)
    return diagnostics


def write_diagnostics(path: Path, diagnostics: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(diagnostics), handle, indent=2)

    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "transition",
            "start",
            "end",
            "before_luma",
            "vace_luma",
            "after_luma",
            "before_saturation",
            "vace_saturation",
            "after_saturation",
        ])
        for item in diagnostics:
            writer.writerow([
                item.get("transition", ""),
                item.get("start", ""),
                item.get("end", ""),
                item.get("before_anchor", {}).get("luma", ""),
                item.get("vace_region", {}).get("luma", ""),
                item.get("after_anchor", {}).get("luma", ""),
                item.get("before_anchor", {}).get("saturation", ""),
                item.get("vace_region", {}).get("saturation", ""),
                item.get("after_anchor", {}).get("saturation", ""),
            ])


def hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_recovery_sequence(
    frame_files: Sequence[Path],
    output_dir: Path,
    prefix: str,
    regions: Sequence[Tuple[int, Dict[str, int]]],
    *,
    anchor_window: int,
    correction_strength: float,
    luma_strength: float,
    chroma_strength: float,
    min_gain: float,
    max_gain: float,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, src in enumerate(frame_files):
        hardlink_or_copy(src, frame_path(output_dir, prefix, index))

    diagnostics: List[Dict[str, Any]] = []
    for transition_index, region in regions:
        log(f"Correcting transition {transition_index}: frames {region['start'] + 1}-{region['end']}")
        corrected, item = correction_for_region(
            frame_files,
            region,
            anchor_window=anchor_window,
            correction_strength=correction_strength,
            luma_strength=luma_strength,
            chroma_strength=chroma_strength,
            min_gain=min_gain,
            max_gain=max_gain,
        )
        item["transition"] = transition_index
        diagnostics.append(item)
        for frame_index, frame in corrected.items():
            dst = frame_path(output_dir, prefix, frame_index)
            if dst.exists():
                dst.unlink()
            save_rgb(dst, frame)
    return diagnostics


def encode_video(
    frames_dir: Path,
    prefix: str,
    output_path: Path,
    *,
    fps: float,
    crf: int,
    pix_fmt: str,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-v", "error",
        "-y" if overwrite else "-n",
        "-framerate", str(fps),
        "-start_number", "1",
        "-i", str(frames_dir / f"{prefix}_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", pix_fmt,
        "-crf", str(crf),
        "-vf", "scale=out_color_matrix=bt709",
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        str(output_path),
    ]
    log(f"Encoding video: {output_path}")
    run_command(args)


def source_video_paths(source_dir: Path, prefix: str, first_suffix: int, last_suffix: int) -> List[Path]:
    return [source_dir / f"{prefix}_{index:05d}.mp4" for index in range(first_suffix, last_suffix + 1)]


def source_clip_duration(video_path: Path, fps: float) -> float:
    metadata = ffprobe_json(
        video_path,
        "format=duration:stream=codec_type,duration,nb_frames,avg_frame_rate",
    )
    video_stream = next(
        (stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"),
        {},
    )
    if fps > 0 and video_stream.get("nb_frames"):
        try:
            return int(video_stream["nb_frames"]) / fps
        except ValueError:
            pass
    for source in (video_stream, metadata.get("format", {})):
        if source.get("duration"):
            return float(source["duration"])
    raise RuntimeError(f"Unable to determine video duration for {video_path}")


def has_audio_stream(video_path: Path) -> bool:
    metadata = ffprobe_json(video_path, "stream=codec_type")
    return any(stream.get("codec_type") == "audio" for stream in metadata.get("streams", []))


def extract_duration_matched_audio(video_paths: Sequence[Path], output_audio_path: Path, *, fps: float) -> bool:
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    temp_audio_files: List[Path] = []
    for index, video_path in enumerate(video_paths):
        if not video_path.exists():
            log(f"Missing source video for audio: {video_path}")
            continue
        segment_duration = source_clip_duration(video_path, fps)
        temp_audio = output_audio_path.with_name(f"{output_audio_path.stem}_temp_{index:05d}.wav")
        if has_audio_stream(video_path):
            args = [
                "ffmpeg",
                "-v", "error",
                "-y",
                "-i", str(video_path),
                "-vn",
                "-af",
                f"aresample=44100:async=1:first_pts=0,apad,atrim=0:{segment_duration:.6f},asetpts=PTS-STARTPTS",
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                str(temp_audio),
            ]
            result = subprocess.run(args, capture_output=True)
        else:
            args = [
                "ffmpeg",
                "-v", "error",
                "-y",
                "-f", "lavfi",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", f"{segment_duration:.6f}",
                "-acodec", "pcm_s16le",
                str(temp_audio),
            ]
            result = subprocess.run(args, capture_output=True)

        if result.returncode == 0 and temp_audio.exists() and temp_audio.stat().st_size > 0:
            temp_audio_files.append(temp_audio)
        elif temp_audio.exists():
            temp_audio.unlink()

    if not temp_audio_files:
        return False

    if len(temp_audio_files) == 1:
        shutil.copy2(temp_audio_files[0], output_audio_path)
    else:
        concat_list = output_audio_path.with_name(f"{output_audio_path.stem}_list.txt")
        with concat_list.open("w", encoding="utf-8") as handle:
            for audio_file in temp_audio_files:
                handle.write(f"file '{audio_file}'\n")
        run_command([
            "ffmpeg",
            "-v", "error",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            str(output_audio_path),
        ])
        concat_list.unlink(missing_ok=True)

    for temp_audio in temp_audio_files:
        temp_audio.unlink(missing_ok=True)
    return True


def create_silent_audio(output_audio_path: Path, duration_seconds: float) -> None:
    run_command([
        "ffmpeg",
        "-v", "error",
        "-y",
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", f"{duration_seconds:.6f}",
        "-acodec", "pcm_s16le",
        str(output_audio_path),
    ])


def mux_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    duration_seconds: float,
    overwrite: bool,
) -> None:
    args = [
        "ffmpeg",
        "-v", "error",
        "-y" if overwrite else "-n",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-af", f"apad=whole_dur={duration_seconds + 1.0:.6f}",
        "-shortest",
        str(output_path),
    ]
    log(f"Muxing audio: {output_path}")
    run_command(args)


def make_preview(
    frame_files: Sequence[Path],
    preview_dir: Path,
    prefix: str,
    transition_index: int,
    region: Dict[str, int],
    *,
    fps: float,
    crf: int,
    pix_fmt: str,
    overwrite: bool,
    anchor_window: int,
    correction_strength: float,
    luma_strength: float,
    chroma_strength: float,
    min_gain: float,
    max_gain: float,
) -> None:
    start = max(0, region["start"] - anchor_window)
    end = min(len(frame_files), region["end"] + anchor_window)
    base_dir = preview_dir / f"transition_{transition_index:03d}_before"
    corrected_dir = preview_dir / f"transition_{transition_index:03d}_after"
    base_dir.mkdir(parents=True, exist_ok=True)
    corrected_dir.mkdir(parents=True, exist_ok=True)

    local_files = list(frame_files[start:end])
    for local_index, src in enumerate(local_files):
        hardlink_or_copy(src, frame_path(base_dir, prefix, local_index))
        hardlink_or_copy(src, frame_path(corrected_dir, prefix, local_index))

    local_region = {"start": region["start"] - start, "end": region["end"] - start}
    corrected, _ = correction_for_region(
        local_files,
        local_region,
        anchor_window=min(anchor_window, local_region["start"], len(local_files) - local_region["end"]),
        correction_strength=correction_strength,
        luma_strength=luma_strength,
        chroma_strength=chroma_strength,
        min_gain=min_gain,
        max_gain=max_gain,
    )
    for local_index, frame in corrected.items():
        dst = frame_path(corrected_dir, prefix, local_index)
        if dst.exists():
            dst.unlink()
        save_rgb(dst, frame)

    encode_video(base_dir, prefix, preview_dir / f"transition_{transition_index:03d}_before.mp4",
                 fps=fps, crf=crf, pix_fmt=pix_fmt, overwrite=overwrite)
    encode_video(corrected_dir, prefix, preview_dir / f"transition_{transition_index:03d}_after.mp4",
                 fps=fps, crf=crf, pix_fmt=pix_fmt, overwrite=overwrite)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temp-dir", required=True, type=Path, help="Auto Joiner temp folder containing state.json and PNG frames")
    parser.add_argument("--source-dir", type=Path, help="Folder containing source mp4 clips")
    parser.add_argument("--file-prefix", help="Frame/source file prefix; defaults to state.json file_prefix")
    parser.add_argument("--first-suffix", type=int, help="First source clip suffix; defaults to state.json")
    parser.add_argument("--last-suffix", type=int, help="Last source clip suffix; defaults to state.json")
    parser.add_argument("--output", type=Path, help="Final output mp4 path")
    parser.add_argument("--work-dir", type=Path, help="Recovery work directory")
    parser.add_argument("--fps", type=float, help="Output fps; defaults to state.json fps")
    parser.add_argument("--crf", type=int, default=DEFAULT_CRf)
    parser.add_argument("--pix-fmt", default=DEFAULT_PIX_FMT, choices=["yuv420p", "yuv420p10le"])
    parser.add_argument("--correction-strength", type=float, default=0.75)
    parser.add_argument("--luma-strength", type=float, default=0.75)
    parser.add_argument("--chroma-strength", type=float, default=0.60)
    parser.add_argument("--blend-region", type=int, default=30, help="Context frames used for previews and diagnostics")
    parser.add_argument("--anchor-window", type=int, help="Frames before/after each transition used as color anchors; defaults to --blend-region")
    parser.add_argument("--min-gain", type=float, default=0.75)
    parser.add_argument("--max-gain", type=float, default=1.25)
    parser.add_argument("--transition", type=int, action="append", help="1-based transition to analyze/preview/correct; may be repeated")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--preview", action="store_true", help="Generate before/after preview clips for selected transitions")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--remux-audio-video", type=Path, help="Existing video file to remux with rebuilt duration-matched source audio")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    temp_dir = args.temp_dir.resolve()
    state = load_state(temp_dir)
    prefix = sanitize_prefix(args.file_prefix or state["file_prefix"])
    source_dir = (args.source_dir or Path(state["directory"])).resolve()
    first_suffix = args.first_suffix or int(state["first_suffix"])
    last_suffix = args.last_suffix or int(state["last_suffix"])
    fps = float(args.fps or state.get("fps", 24.0))
    output = (args.output or Path.cwd() / f"wan_vace_recovered_{datetime.now():%Y%m%d%H%M%S}.mp4").resolve()
    work_dir = (args.work_dir or temp_dir / f"recovery-{datetime.now():%Y%m%d%H%M%S}").resolve()
    anchor_window = args.anchor_window if args.anchor_window is not None else args.blend_region

    if args.remux_audio_video:
        remux_source = args.remux_audio_video.resolve()
        if not remux_source.exists():
            raise FileNotFoundError(f"Missing remux source video: {remux_source}")
        duration_seconds = float(
            ffprobe_json(remux_source, "format=duration").get("format", {}).get("duration", 0.0)
        )
        audio_path = work_dir / "audio" / "duration_matched_audio.wav"
        log(f"Rebuilding duration-matched audio for: {remux_source}")
        if not extract_duration_matched_audio(
            source_video_paths(source_dir, prefix, first_suffix, last_suffix),
            audio_path,
            fps=fps,
        ):
            log("No source audio extracted; creating silent audio track")
            create_silent_audio(audio_path, duration_seconds)
        mux_audio(remux_source, audio_path, output, duration_seconds=duration_seconds, overwrite=args.overwrite)
        log(f"Audio-remuxed video written: {output}")
        return 0

    frame_files = sorted_frame_files(temp_dir, prefix)
    regions = dedupe_regions(state.get("vace_regions", []), len(frame_files))
    chosen_regions = selected_regions(regions, args.transition)
    diagnostics_path = work_dir / "diagnostics" / "transition_diagnostics.json"

    log(f"Temp dir: {temp_dir}")
    log(f"Frames: {len(frame_files)}")
    log(f"Transitions: {len(regions)} unique ({len(chosen_regions)} selected)")
    log(f"FPS: {fps}")

    diagnostics = analyze_regions(frame_files, chosen_regions, anchor_window=anchor_window)
    write_diagnostics(diagnostics_path, diagnostics)
    log(f"Wrote diagnostics: {diagnostics_path}")

    if args.preview:
        preview_dir = work_dir / "previews"
        for transition_index, region in chosen_regions:
            make_preview(
                frame_files,
                preview_dir,
                prefix,
                transition_index,
                region,
                fps=fps,
                crf=args.crf,
                pix_fmt=args.pix_fmt,
                overwrite=args.overwrite,
                anchor_window=max(anchor_window, args.blend_region),
                correction_strength=args.correction_strength,
                luma_strength=args.luma_strength,
                chroma_strength=args.chroma_strength,
                min_gain=args.min_gain,
                max_gain=args.max_gain,
            )
        log(f"Wrote previews under: {preview_dir}")

    if args.analysis_only:
        return 0

    frames_dir = work_dir / "frames"
    correction_diagnostics = build_recovery_sequence(
        frame_files,
        frames_dir,
        prefix,
        chosen_regions,
        anchor_window=anchor_window,
        correction_strength=args.correction_strength,
        luma_strength=args.luma_strength,
        chroma_strength=args.chroma_strength,
        min_gain=args.min_gain,
        max_gain=args.max_gain,
    )
    write_diagnostics(work_dir / "diagnostics" / "correction_diagnostics.json", correction_diagnostics)

    duration_seconds = len(frame_files) / fps if fps > 0 else 0.0
    no_audio_video = output.with_name(f"{output.stem}-video-only{output.suffix}")
    encode_video(frames_dir, prefix, no_audio_video, fps=fps, crf=args.crf, pix_fmt=args.pix_fmt, overwrite=args.overwrite)

    if args.no_audio:
        if output != no_audio_video:
            if output.exists() and args.overwrite:
                output.unlink()
            shutil.copy2(no_audio_video, output)
        log(f"Wrote video without audio: {output}")
        return 0

    audio_path = work_dir / "audio" / "combined_audio.wav"
    if not extract_duration_matched_audio(
        source_video_paths(source_dir, prefix, first_suffix, last_suffix),
        audio_path,
        fps=fps,
    ):
        log("No source audio extracted; creating silent audio track")
        create_silent_audio(audio_path, duration_seconds)
    mux_audio(no_audio_video, audio_path, output, duration_seconds=duration_seconds, overwrite=args.overwrite)
    log(f"Recovered video written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
