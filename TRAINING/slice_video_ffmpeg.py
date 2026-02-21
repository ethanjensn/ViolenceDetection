import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _ffmpeg_exe(explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path

    return shutil.which("ffmpeg") or ""


def _run(cmd: list[str]) -> int:
    try:
        completed = subprocess.run(cmd, check=False)
        return int(completed.returncode)
    except FileNotFoundError:
        return 127


def slice_video(
    input_video: Path,
    output_dir: Path,
    segment_time_seconds: int = 5,
    output_ext: str = "mp4",
    reencode: bool = False,
    overwrite: bool = False,
    ffmpeg_path: str | None = None,
) -> int:
    ffmpeg = _ffmpeg_exe(ffmpeg_path)
    if not ffmpeg:
        print("Error: ffmpeg not found in PATH. Install FFmpeg or pass --ffmpeg-path.", file=sys.stderr)
        return 127

    if not input_video.exists() or not input_video.is_file():
        print(f"Error: input video not found: {input_video}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep filenames safe/predictable.
    stem = input_video.stem
    output_pattern = output_dir / f"{stem}_%06d.{output_ext.lstrip('.')}"

    cmd: list[str] = [ffmpeg]

    if overwrite:
        cmd.append("-y")
    else:
        cmd.append("-n")

    cmd.extend(
        [
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_video),
            "-f",
            "segment",
            "-segment_time",
            str(int(segment_time_seconds)),
            "-reset_timestamps",
            "1",
        ]
    )

    if reencode:
        # Safer segmentation across files (keyframes will be created), but slower.
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac"])
    else:
        # Fastest. Note: exact 5s cut points depend on existing keyframes.
        cmd.extend(["-c", "copy"])

    cmd.append(str(output_pattern))

    return _run(cmd)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice a video into fixed-length chunks using FFmpeg (default: 5 seconds)."
    )

    parser.add_argument(
        "video",
        help="Path to input video file",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: ./slices/<video_stem>)",
    )

    parser.add_argument(
        "--seconds",
        type=int,
        default=5,
        help="Chunk duration in seconds (default: 5)",
    )

    parser.add_argument(
        "--ext",
        default="mp4",
        help="Output extension/format (default: mp4)",
    )

    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode for more accurate chunk boundaries (slower)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    parser.add_argument(
        "--ffmpeg-path",
        default=None,
        help="Explicit path to ffmpeg.exe (if not on PATH)",
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    input_video = Path(args.video).expanduser().resolve()

    if args.out_dir:
        output_dir = Path(args.out_dir).expanduser().resolve()
    else:
        output_dir = Path.cwd() / "slices" / input_video.stem

    if args.seconds <= 0:
        print("Error: --seconds must be > 0", file=sys.stderr)
        return 2

    return slice_video(
        input_video=input_video,
        output_dir=output_dir,
        segment_time_seconds=args.seconds,
        output_ext=args.ext,
        reencode=bool(args.reencode),
        overwrite=bool(args.overwrite),
        ffmpeg_path=args.ffmpeg_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
