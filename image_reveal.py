#!/usr/bin/env python3
"""
Image Reveal Video Generator

Creates a video where an image starts completely blurred/pixelated
and progressively becomes clearer over 30 seconds.

Reads images from 'images' directory and saves videos to 'images_final'.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Hardcoded directories
INPUT_DIR = "images"
OUTPUT_DIR = "images_final"


def pixelate_image(image: np.ndarray, pixel_size: int) -> np.ndarray:
    """
    Pixelate an image by downscaling and upscaling.

    Args:
        image: Input image as numpy array
        pixel_size: Size of the pixelation blocks (higher = more pixelated)

    Returns:
        Pixelated image
    """
    if pixel_size <= 1:
        return image

    h, w = image.shape[:2]

    # Calculate small dimensions
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)

    # Averaging on the way down keeps blocks from being dominated by single pixels
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    return pixelated


def blur_image(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to an image.

    Args:
        image: Input image as numpy array
        sigma: Standard deviation of the blur in pixels (higher = more blur)

    Returns:
        Blurred image
    """
    if sigma < 0.5:
        return image

    h, w = image.shape[:2]

    # A huge kernel on a full-resolution frame is very slow, and blurring a
    # downscaled copy is visually equivalent, so cap the working sigma.
    working_sigma = 8.0
    if sigma > working_sigma:
        scale = working_sigma / sigma
        small = cv2.resize(
            image,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        blurred = cv2.GaussianBlur(small, (0, 0), working_sigma, borderType=cv2.BORDER_REPLICATE)
        return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)

    return cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)


def create_reveal_video(
    input_image_path: str,
    output_video_path: str,
    duration: float = 30.0,
    fps: int = 30,
    effect: str = "both",
    start_blocks: float = 4.0,
    blur_strength: float = 0.35,
    ease: float = 1.5,
) -> None:
    """
    Create a video that reveals an image from blurred/pixelated to clear.

    Args:
        input_image_path: Path to the input image
        output_video_path: Path for the output video file
        duration: Duration of the video in seconds
        fps: Frames per second
        effect: Type of effect - "blur", "pixelate", or "both"
        start_blocks: Mosaic blocks across the image's smaller side at the start.
            Fewer blocks means a coarser, harder opening
        blur_strength: Blur sigma as a multiple of the current block size
        ease: Pacing exponent. 1.0 doubles the resolution at a constant rate,
            above 1 lingers on the coarse mosaic
    """
    # Load the image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Could not load image: {input_image_path}")

    height, width = image.shape[:2]
    total_frames = int(duration * fps)

    # Effect strength must follow resolution, otherwise a fixed pixel amount is
    # barely visible on a large image.
    short_side = min(width, height)
    start_block_size = max(2.0, short_side / max(1.0, start_blocks))

    def block_size_at(progress: float) -> float:
        """
        Block size in pixels, shrinking geometrically to 1 (a clear image).

        Interpolating the block size geometrically keeps the resolution
        doubling at an even pace, so detail keeps arriving throughout instead
        of all at once near the end.
        """
        return start_block_size ** (1.0 - progress**ease)

    print(f"Input image: {input_image_path}")
    print(f"Image size: {width}x{height}")
    print(f"Output video: {output_video_path}")
    print(f"Duration: {duration}s at {fps} FPS ({total_frames} frames)")
    print(f"Effect: {effect}")

    schedule = " → ".join(
        f"{p:.0%}: {short_side / block_size_at(p):.0f}" for p in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    print(f"Blocks across the short side: {schedule}")
    print()

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Could not create video writer for: {output_video_path}")

    try:
        for frame_num in range(total_frames):
            # Progress from 0 (start, most blurred) to 1 (end, clear)
            progress = frame_num / (total_frames - 1) if total_frames > 1 else 1.0

            block_size = block_size_at(progress)

            # Start with the original image
            frame = image.copy()

            if effect in ("pixelate", "both"):
                pixel_size = int(round(block_size))
                if pixel_size > 1:
                    frame = pixelate_image(frame, pixel_size)

            # Blur last so pixelation blocks get softened into an unreadable
            # blob. Scaling it by the block size means both effects hide detail
            # at the same scale and both land on a clean final frame.
            if effect in ("blur", "both"):
                frame = blur_image(frame, blur_strength * (block_size - 1.0))

            out.write(frame)

            # Progress indicator
            if frame_num % fps == 0 or frame_num == total_frames - 1:
                pct = (frame_num + 1) / total_frames * 100
                bar_width = 40
                filled = int(bar_width * (frame_num + 1) / total_frames)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"\rGenerating: [{bar}] {pct:5.1f}%", end="", flush=True)

        print()  # New line after progress bar

    finally:
        out.release()

    # Verify output file was created
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path) / (1024 * 1024)
        print(f"\n✓ Video created successfully: {output_video_path}")
        print(f"  File size: {file_size:.2f} MB")
    else:
        raise RuntimeError("Failed to create output video")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a reveal video from a blurred/pixelated image to clear.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg                     # Creates images_final/photo_reveal.mp4
  %(prog)s photo.png -d 15               # 15 second reveal
  %(prog)s photo.jpg --effect blur       # Blur-only reveal
  %(prog)s photo.jpg --effect pixelate   # Pixelation effect only
  %(prog)s photo.jpg --start-blocks 3    # Even coarser opening mosaic
        """,
    )

    parser.add_argument("filename", help="Image filename (looked up in 'images' directory)")

    parser.add_argument(
        "-d", "--duration", type=float, default=30.0, help="Duration of the video in seconds (default: 30)"
    )

    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")

    parser.add_argument(
        "--effect", choices=["blur", "pixelate", "both"], default="both", help="Type of effect to use (default: both)"
    )

    parser.add_argument(
        "--start-blocks",
        type=float,
        default=4.0,
        help="Mosaic blocks across the short side at the start. Fewer is harder (default: 4)",
    )

    parser.add_argument(
        "--blur-strength",
        type=float,
        default=0.35,
        help="Blur sigma as a multiple of the current block size (default: 0.35)",
    )

    parser.add_argument(
        "--ease",
        type=float,
        default=1.5,
        help="Pacing exponent: 1 reveals at a constant rate, >1 lingers on the coarse mosaic (default: 1.5)",
    )

    args = parser.parse_args()

    # Build input path from hardcoded directory
    input_path = Path(INPUT_DIR) / args.filename

    # Validate input file exists
    if not input_path.exists():
        print(f"Error: Image not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output path
    output_path = output_dir / f"{input_path.stem}_reveal.mp4"

    # Create the video
    try:
        create_reveal_video(
            input_image_path=str(input_path),
            output_video_path=str(output_path),
            duration=args.duration,
            fps=args.fps,
            effect=args.effect,
            start_blocks=args.start_blocks,
            blur_strength=args.blur_strength,
            ease=args.ease,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
