#!/usr/bin/env python3
"""
Sentence Reveal Video Generator

Creates a video where the letters of a sentence appear one at a time in random
order, so players have to guess the phrase before it is fully revealed.

Saves videos to 'sentences_final'.
"""

import argparse
import random
import re
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "sentences_final"

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG_COLOR = (28, 24, 20)
TEXT_COLOR = (245, 245, 245)
SLOT_COLOR = (95, 85, 78)
BAR_BG_COLOR = (55, 48, 44)
BAR_COLOR = (110, 190, 95)


def to_ascii(sentence: str) -> str:
    """
    Fold a sentence down to ASCII, since OpenCV's built-in Hershey fonts can
    only draw ASCII and turn anything else into a stray glyph.

    Accents are stripped (á -> a) and inverted Spanish punctuation is flipped,
    so "¿Quién?" becomes "?Quien?".
    """
    flipped = sentence.replace("¿", "?").replace("¡", "!").replace("’", "'").replace("“", '"').replace("”", '"')
    decomposed = unicodedata.normalize("NFD", flipped)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))

    result = "".join(c if c.isascii() else "?" for c in stripped)
    if result != sentence:
        print(f"Note: rendered as ASCII (OpenCV fonts are ASCII-only): {result}")

    return result


def wrap_sentence(sentence: str, max_chars: int) -> list[str]:
    """
    Greedily wrap a sentence into lines of at most max_chars characters.

    Words longer than max_chars are kept whole on their own line; the layout
    step shrinks the font until they fit.
    """
    lines: list[str] = []
    current = ""

    for word in sentence.split():
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def measure_grid(font_scale: float, thickness: int) -> tuple[int, int]:
    """
    Return (cell_width, char_height) for a monospaced grid at the given scale.

    Hershey fonts are proportional, so the cell is sized to the widest glyph
    to keep every line aligned on the same column grid.
    """
    sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?!"
    cell_width = max(cv2.getTextSize(c, FONT, font_scale, thickness)[0][0] for c in sample)
    char_height = cv2.getTextSize("A", FONT, font_scale, thickness)[0][1]
    return cell_width, char_height


def fit_layout(
    sentence: str,
    width: int,
    height: int,
    margin_ratio: float,
    line_spacing: float,
) -> tuple[list[str], float, int]:
    """
    Choose the line wrapping and font scale that fill the frame best.

    Every possible wrap width is scored by the font scale it allows, so short
    sentences get one big line and long ones break into balanced blocks.

    Returns:
        (lines, font_scale, thickness)
    """
    usable_w = width * (1 - 2 * margin_ratio)
    usable_h = height * (1 - 2 * margin_ratio)

    # Metrics are linear in font_scale, so measure once at 1.0 and solve directly
    base_cell_w, base_char_h = measure_grid(1.0, 1)

    longest_word = max((len(w) for w in sentence.split()), default=1)
    best: tuple[list[str], float] | None = None

    for max_chars in range(max(4, longest_word), max(5, len(sentence) + 1)):
        lines = wrap_sentence(sentence, max_chars)
        columns = max(len(line) for line in lines)

        scale_w = usable_w / (base_cell_w * columns)
        scale_h = usable_h / (base_char_h * line_spacing * len(lines))
        scale = min(scale_w, scale_h)

        if best is None or scale > best[1]:
            best = (lines, scale)

    lines, font_scale = best
    font_scale = min(font_scale * 0.97, 8.0)
    thickness = max(2, int(round(font_scale * 2.2)))

    # Thickness widens glyphs, so verify the fit and shrink if it now overflows
    cell_w, char_h = measure_grid(font_scale, thickness)
    columns = max(len(line) for line in lines)
    overflow = max(cell_w * columns / usable_w, char_h * line_spacing * len(lines) / usable_h)
    if overflow > 1.0:
        font_scale /= overflow
        thickness = max(2, int(round(font_scale * 2.2)))

    return lines, font_scale, thickness


def build_slots(lines: list[str]) -> list[dict]:
    """
    Build the per-character grid cells for the sentence.

    Letters and digits are hidden until revealed; spaces and punctuation stay
    visible so word lengths and structure act as clues.
    """
    slots = []
    for row, line in enumerate(lines):
        for col, char in enumerate(line):
            if char == " ":
                continue
            slots.append({"char": char, "row": row, "col": col, "hidden": char.isalnum()})
    return slots


def render_base_frame(
    width: int,
    height: int,
    lines: list[str],
    font_scale: float,
    thickness: int,
    line_spacing: float,
    slots: list[dict],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Draw the background, the underscore placeholders and the always-visible
    punctuation once, since none of it changes between frames.

    Returns:
        (base_frame, cell_origins) where cell_origins[i] is the (x, baseline_y)
        text origin for slots[i]
    """
    frame = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)

    cell_w, char_h = measure_grid(font_scale, thickness)
    line_h = char_h * line_spacing

    underline_drop = char_h * 0.4

    block_h = line_h * len(lines)
    # The underlines sit below the last baseline, so lift the block to keep the
    # visible ink centred rather than the bare grid
    top = (height - block_h) / 2 - underline_drop / 2

    # Each line is centred on its own width so ragged wraps stay balanced
    line_left = [(width - cell_w * len(line)) / 2 for line in lines]

    origins: list[tuple[int, int]] = []

    for slot in slots:
        left = line_left[slot["row"]]
        # Centre each glyph inside its grid cell
        glyph_w = cv2.getTextSize(slot["char"], FONT, font_scale, thickness)[0][0]
        x = left + slot["col"] * cell_w + (cell_w - glyph_w) / 2
        baseline = top + slot["row"] * line_h + char_h + (line_h - char_h) / 2
        origins.append((int(round(x)), int(round(baseline))))

        if not slot["hidden"]:
            cv2.putText(frame, slot["char"], origins[-1], FONT, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)
            continue

        underline_y = int(round(baseline + underline_drop))
        pad = int(round(cell_w * 0.12))
        x0 = int(round(left + slot["col"] * cell_w)) + pad
        x1 = int(round(left + (slot["col"] + 1) * cell_w)) - pad
        cv2.line(frame, (x0, underline_y), (x1, underline_y), SLOT_COLOR, max(1, thickness // 2), cv2.LINE_AA)

    return frame, origins


def create_sentence_video(
    sentence: str,
    output_video_path: str,
    duration: float = 30.0,
    hold: float = 3.0,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    fade: float = 0.4,
    seed: int | None = None,
    show_progress_bar: bool = True,
) -> None:
    """
    Render a video that reveals a sentence letter by letter in random order.

    Args:
        sentence: The phrase to reveal
        output_video_path: Path for the output video file
        duration: Seconds over which all letters appear
        hold: Extra seconds showing the finished sentence
        fps: Frames per second
        width: Video width in pixels
        height: Video height in pixels
        fade: Seconds each letter takes to fade in
        seed: Random seed for a reproducible reveal order
        show_progress_bar: Draw a time progress bar at the bottom
    """
    margin_ratio = 0.08
    line_spacing = 1.75

    lines, font_scale, thickness = fit_layout(sentence, width, height, margin_ratio, line_spacing)
    slots = build_slots(lines)
    hidden_indices = [i for i, slot in enumerate(slots) if slot["hidden"]]

    if not hidden_indices:
        raise ValueError("The sentence has no letters or digits to reveal")

    rng = random.Random(seed)
    reveal_order = hidden_indices[:]
    rng.shuffle(reveal_order)

    # Stagger the starts so the final letter finishes exactly at `duration`
    fade = min(fade, duration)
    step = (duration - fade) / max(1, len(reveal_order) - 1)
    start_times = {slot_index: i * step for i, slot_index in enumerate(reveal_order)}

    total_frames = int(round((duration + hold) * fps))

    print(f"Sentence: {sentence}")
    print(f"Layout: {len(lines)} line(s), {len(hidden_indices)} letters to reveal")
    print(f"Output video: {output_video_path}")
    print(f"Duration: {duration}s reveal + {hold}s hold at {fps} FPS ({total_frames} frames)")
    print(f"Video size: {width}x{height}")
    print()

    base_frame, origins = render_base_frame(width, height, lines, font_scale, thickness, line_spacing, slots)

    bg = np.array(BG_COLOR, dtype=np.float64)
    fg = np.array(TEXT_COLOR, dtype=np.float64)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Could not create video writer for: {output_video_path}")

    try:
        for frame_num in range(total_frames):
            t = frame_num / fps
            frame = base_frame.copy()

            for slot_index in hidden_indices:
                alpha = (t - start_times[slot_index]) / fade if fade > 0 else 1.0
                alpha = min(1.0, max(0.0, alpha))
                if alpha <= 0.0:
                    continue

                # The background is flat, so blending the colour is equivalent
                # to blending the pixels and avoids a full-frame composite
                color = bg + (fg - bg) * alpha
                cv2.putText(
                    frame,
                    slots[slot_index]["char"],
                    origins[slot_index],
                    FONT,
                    font_scale,
                    tuple(color),
                    thickness,
                    cv2.LINE_AA,
                )

            if show_progress_bar:
                bar_h = max(4, height // 160)
                bar_y = height - bar_h * 4
                margin = int(width * margin_ratio)
                cv2.rectangle(frame, (margin, bar_y), (width - margin, bar_y + bar_h), BAR_BG_COLOR, -1)
                filled = int((width - 2 * margin) * min(1.0, t / duration))
                if filled > 0:
                    cv2.rectangle(frame, (margin, bar_y), (margin + filled, bar_y + bar_h), BAR_COLOR, -1)

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

    if not Path(output_video_path).exists():
        raise RuntimeError("Failed to create output video")

    file_size = Path(output_video_path).stat().st_size / (1024 * 1024)
    print(f"\n✓ Video created successfully: {output_video_path}")
    print(f"  File size: {file_size:.2f} MB")


def slugify(sentence: str, max_length: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", sentence.lower()).strip("_")
    return slug[:max_length].strip("_") or "sentence"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a video that reveals a sentence letter by letter in random order.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "the price is right"             # 30s reveal, 1920x1080
  %(prog)s "guess the movie" -d 45          # slower 45 second reveal
  %(prog)s "hello world" --vertical         # 1080x1920 for stories/reels
  %(prog)s "hello world" --seed 7           # reproducible reveal order

Output files are saved to the 'sentences_final/' folder.
        """,
    )

    parser.add_argument("sentence", nargs="+", help="The sentence to reveal")

    parser.add_argument("-o", "--output", help="Output filename (saved to sentences_final/)")

    parser.add_argument(
        "-d", "--duration", type=float, default=30.0, help="Seconds over which all letters appear (default: 30)"
    )

    parser.add_argument(
        "--hold", type=float, default=3.0, help="Extra seconds showing the finished sentence (default: 3)"
    )

    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")

    parser.add_argument("--width", type=int, default=1920, help="Video width in pixels (default: 1920)")

    parser.add_argument("--height", type=int, default=1080, help="Video height in pixels (default: 1080)")

    parser.add_argument("--vertical", action="store_true", help="Use a 1080x1920 portrait frame instead")

    parser.add_argument("--fade", type=float, default=0.4, help="Seconds each letter takes to fade in (default: 0.4)")

    parser.add_argument("--seed", type=int, help="Random seed for a reproducible reveal order")

    parser.add_argument("--no-progress-bar", action="store_true", help="Hide the time progress bar")

    args = parser.parse_args()

    sentence = " ".join(args.sentence).strip()
    if not sentence:
        print("Error: The sentence is empty", file=sys.stderr)
        sys.exit(1)

    sentence = to_ascii(sentence)

    width, height = (1080, 1920) if args.vertical else (args.width, args.height)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = args.output or f"{slugify(sentence)}_reveal.mp4"
    output_path = OUTPUT_DIR / output_filename

    try:
        create_sentence_video(
            sentence=sentence,
            output_video_path=str(output_path),
            duration=args.duration,
            hold=args.hold,
            fps=args.fps,
            width=width,
            height=height,
            fade=args.fade,
            seed=args.seed,
            show_progress_bar=not args.no_progress_bar,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
