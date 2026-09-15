#!/usr/bin/env python3
"""
Image Zoom-Out Video Generator

Creates a video that starts extremely zoomed into the centre of an image
(too close to recognise the subject) and slowly pulls back to the full frame.

Reads images from 'images' and saves videos to 'images_final'.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

INPUT_DIR = "images"
OUTPUT_DIR = "images_final"


def zoom_frame(image: np.ndarray, zoom: float) -> np.ndarray:
    """
    Magnify the image around its centre by `zoom`.

    zoom=1 is the full image. zoom=20 shows a 1/20 crop blown up to fill the
    frame. The affine is subpixel so the pull-back does not stair-step.
    """
    if zoom <= 1.0:
        return image

    if zoom > 3.0:
        # Magnifying a JPEG this far turns its 8x8 compression blocking into
        # hard seams, so soften the source a little before scaling it up.
        sigma = min(0.9, 0.06 * zoom)
        image = cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)

    height, width = image.shape[:2]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    matrix = np.array(
        [[zoom, 0.0, cx - zoom * cx], [0.0, zoom, cy - zoom * cy]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def open_video_writer(path: str, fps: int, size: tuple[int, int]) -> cv2.VideoWriter:
    """
    Open a writer, preferring H.264.

    An extreme zoom produces near-flat gradients, and the older mp4v encoder
    bands them into visible macroblock seams.
    """
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(f"Could not create video writer for: {path}")


def create_zoom_video(
    input_image_path: str,
    output_video_path: str,
    duration: float = 30.0,
    fps: int = 30,
    start_zoom: float = 40.0,
    hold: float = 2.0,
    ease: float = 1.0,
) -> None:
    """
    Create a video that zooms out from the centre of an image to the full frame.

    Args:
        input_image_path: Path to the input image
        output_video_path: Path for the output video file
        duration: Seconds spent zooming out
        fps: Frames per second
        start_zoom: Starting magnification (higher = closer, harder to recognise)
        hold: Extra seconds on the fully revealed image
        ease: Pacing exponent. 1.0 keeps the zoom rate constant, above 1 lingers
            on the close-up, below 1 opens up sooner
    """
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Could not load image: {input_image_path}")

    height, width = image.shape[:2]
    zoom_frames = int(duration * fps)
    hold_frames = int(hold * fps)
    total_frames = zoom_frames + hold_frames

    # The crop at the start must be a few pixels on the short side or the
    # first frames are a single colour and the zoom has nowhere to go.
    short_side = min(width, height)
    max_useful_zoom = short_side / 4.0
    start_zoom = min(start_zoom, max_useful_zoom)

    print(f"Input image: {input_image_path}")
    print(f"Image size: {width}x{height}")
    print(f"Output video: {output_video_path}")
    print(f"Duration: {duration}s zoom + {hold}s hold at {fps} FPS ({total_frames} frames)")
    print(f"Start zoom: {start_zoom:.1f}x (centre crop ~{width / start_zoom:.0f}x{height / start_zoom:.0f})")

    schedule = " → ".join(
        f"{p:.0%}: {start_zoom ** (1.0 - p**ease):.1f}x" for p in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    print(f"Zoom pacing: {schedule}")
    print()

    out = open_video_writer(output_video_path, fps, (width, height))

    try:
        for frame_num in range(total_frames):
            if zoom_frames <= 1:
                progress = 1.0
            elif frame_num >= zoom_frames:
                progress = 1.0
            else:
                progress = frame_num / (zoom_frames - 1)

            # Geometric interpolation: the field of view widens by a constant
            # factor every second, so clues keep arriving at an even pace
            # instead of all landing in the final seconds.
            zoom = start_zoom ** (1.0 - progress**ease)
            frame = zoom_frame(image, zoom)
            out.write(frame)

            if frame_num % fps == 0 or frame_num == total_frames - 1:
                pct = (frame_num + 1) / total_frames * 100
                bar_width = 40
                filled = int(bar_width * (frame_num + 1) / total_frames)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"\rGenerating: [{bar}] {pct:5.1f}%", end="", flush=True)

        print()

    finally:
        out.release()

    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path) / (1024 * 1024)
        print(f"\n✓ Video created successfully: {output_video_path}")
        print(f"  File size: {file_size:.2f} MB")
    else:
        raise RuntimeError("Failed to create output video")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a video that zooms out from the centre of an image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg                     # Creates images_final/photo_zoom.mp4
  %(prog)s photo.jpg -d 20               # 20 second zoom-out
  %(prog)s photo.jpg --start-zoom 60     # Even tighter on the centre at the start
  %(prog)s photo.jpg --ease 1.4          # Linger longer on the close-up
        """,
    )

    parser.add_argument("filename", help="Image filename (looked up in 'images' directory)")

    parser.add_argument(
        "-d", "--duration", type=float, default=30.0, help="Seconds spent zooming out (default: 30)"
    )

    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")

    parser.add_argument(
        "--start-zoom",
        type=float,
        default=40.0,
        help="Starting magnification. 40x is a tiny centre crop (default: 40)",
    )

    parser.add_argument(
        "--hold", type=float, default=2.0, help="Extra seconds on the fully revealed image (default: 2)"
    )

    parser.add_argument(
        "--ease",
        type=float,
        default=1.0,
        help="Pacing exponent: 1 zooms at a constant rate, >1 lingers on the close-up (default: 1)",
    )

    args = parser.parse_args()

    if args.start_zoom < 1.0:
        print("Error: --start-zoom must be at least 1", file=sys.stderr)
        sys.exit(1)

    input_path = Path(INPUT_DIR) / args.filename
    if not input_path.exists():
        print(f"Error: Image not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_zoom.mp4"

    try:
        create_zoom_video(
            input_image_path=str(input_path),
            output_video_path=str(output_path),
            duration=args.duration,
            fps=args.fps,
            start_zoom=args.start_zoom,
            hold=args.hold,
            ease=args.ease,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
