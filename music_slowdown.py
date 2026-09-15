#!/usr/bin/env python3
"""
Musical Trivia Audio Processor

Takes an MP3 file and creates a version that starts at 10x speed
and progressively slows down to normal speed over 30 seconds.
"""

import argparse
import os
from pathlib import Path

import numpy as np

# Directory configuration
SCRIPT_DIR = Path(__file__).parent
SONGS_DIR = SCRIPT_DIR / "songs"
SONGS_FINAL_DIR = SCRIPT_DIR / "songs_final"
from pydub import AudioSegment
from scipy.interpolate import interp1d


def time_stretch_progressive(
    audio_data: np.ndarray,
    sample_rate: int,
    start_speed: float = 10.0,
    end_speed: float = 1.0,
    duration_seconds: float = 30.0,
) -> np.ndarray:
    """
    Progressively slow down audio from start_speed to end_speed over duration_seconds.

    This works by resampling the audio with a variable rate that changes over time.
    At 10x speed, we skip through the source audio 10x faster.
    At 1x speed, we play it normally.
    """

    # Calculate how much source audio we need
    # We integrate the speed curve to find total source samples needed
    # Speed decreases linearly from start_speed to end_speed over output_duration

    output_samples = int(duration_seconds * sample_rate)

    # For each output sample, calculate which source sample it maps to
    # Speed at time t: speed(t) = start_speed + (end_speed - start_speed) * (t / duration)
    # Position in source at time t: integral of speed from 0 to t
    # = start_speed * t + (end_speed - start_speed) * t^2 / (2 * duration)

    output_times = np.linspace(0, duration_seconds, output_samples)

    # Calculate source position for each output time
    source_positions = start_speed * output_times + (end_speed - start_speed) * output_times**2 / (2 * duration_seconds)

    # Convert to sample indices
    source_sample_indices = source_positions * sample_rate

    # Check if we have enough source audio
    max_source_sample = int(source_sample_indices[-1])

    if len(audio_data.shape) == 1:
        # Mono audio
        if max_source_sample >= len(audio_data):
            print(
                f"Warning: Source audio too short. Need {max_source_sample/sample_rate:.1f}s, have {len(audio_data)/sample_rate:.1f}s"
            )
            # Pad with silence or loop - we'll just use what we have
            source_sample_indices = np.clip(source_sample_indices, 0, len(audio_data) - 1)

        # Interpolate to get output samples
        source_indices = np.arange(len(audio_data))
        interpolator = interp1d(source_indices, audio_data, kind='linear', bounds_error=False, fill_value=0)
        output_audio = interpolator(source_sample_indices)
    else:
        # Stereo audio
        if max_source_sample >= len(audio_data):
            print(
                f"Warning: Source audio too short. Need {max_source_sample/sample_rate:.1f}s, have {len(audio_data)/sample_rate:.1f}s"
            )
            source_sample_indices = np.clip(source_sample_indices, 0, len(audio_data) - 1)

        output_audio = np.zeros((output_samples, audio_data.shape[1]))
        source_indices = np.arange(len(audio_data))

        for channel in range(audio_data.shape[1]):
            interpolator = interp1d(
                source_indices, audio_data[:, channel], kind='linear', bounds_error=False, fill_value=0
            )
            output_audio[:, channel] = interpolator(source_sample_indices)

    return output_audio.astype(audio_data.dtype)


def process_audio(
    input_path: str,
    output_path: str,
    start_speed: float = 10.0,
    end_speed: float = 1.0,
    duration: float = 30.0,
    normal_duration: float = 30.0,
) -> None:
    """
    Process an MP3 file to create a progressive slowdown effect.
    """

    print(f"Loading audio from: {input_path}")

    # Load the MP3 file
    audio = AudioSegment.from_mp3(input_path)
    sample_rate = audio.frame_rate
    channels = audio.channels

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    print(f"Duration: {len(audio)/1000:.1f} seconds")

    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())

    if channels == 2:
        samples = samples.reshape((-1, 2))

    # Normalize to float for processing
    if audio.sample_width == 2:  # 16-bit
        samples = samples.astype(np.float64) / 32768.0
    elif audio.sample_width == 1:  # 8-bit
        samples = samples.astype(np.float64) / 128.0
    else:
        samples = samples.astype(np.float64)

    # Calculate how much source audio we need
    # Total source time = integral of speed from 0 to duration
    # = start_speed * duration + (end_speed - start_speed) * duration / 2
    # = (start_speed + end_speed) * duration / 2
    source_time_needed = (start_speed + end_speed) * duration / 2
    print(f"Source audio needed for {duration}s output at {start_speed}x→{end_speed}x: {source_time_needed:.1f}s")

    source_samples_needed = int(source_time_needed * sample_rate)
    available_samples = len(samples)

    if source_samples_needed > available_samples:
        print(f"Warning: Only {available_samples/sample_rate:.1f}s available, output will be shorter")

    print(f"Processing with progressive slowdown: {start_speed}x → {end_speed}x over {duration}s...")

    # Apply the progressive time stretch
    processed = time_stretch_progressive(
        samples, sample_rate, start_speed=start_speed, end_speed=end_speed, duration_seconds=duration
    )

    # Calculate where we are in the source audio after the slowdown phase
    source_time_used = (start_speed + end_speed) * duration / 2
    source_samples_used = int(source_time_used * sample_rate)

    # Add normal speed section
    normal_samples = int(normal_duration * sample_rate)
    normal_end_sample = min(source_samples_used + normal_samples, len(samples))
    normal_section = samples[source_samples_used:normal_end_sample]

    # Concatenate slowdown and normal sections
    if len(normal_section) > 0:
        processed = np.concatenate([processed, normal_section])
        print(f"Added {len(normal_section)/sample_rate:.1f}s of normal speed playback")

    # Convert back to int16
    processed = np.clip(processed * 32768.0, -32768, 32767).astype(np.int16)

    # Create output AudioSegment
    if channels == 2:
        processed = processed.flatten()

    output_audio = AudioSegment(processed.tobytes(), frame_rate=sample_rate, sample_width=2, channels=channels)

    print(f"Output duration: {len(output_audio)/1000:.1f} seconds")
    print(f"Saving to: {output_path}")

    # Export as MP3
    output_audio.export(output_path, format="mp3", bitrate="192k")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Create a musical trivia audio that starts fast and slows down",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python music_slowdown.py song.mp3
  python music_slowdown.py song.mp3 -o custom_output.mp3
  python music_slowdown.py song.mp3 --start-speed 5 --duration 20

Input files are read from the 'songs/' folder.
Output files are saved to the 'songs_final/' folder.
        """,
    )

    parser.add_argument("input", help="Input MP3 filename (looked up in songs/ folder)")
    parser.add_argument("-o", "--output", help="Output MP3 filename (saved to songs_final/, default: input_trivia.mp3)")
    parser.add_argument("--start-speed", type=float, default=10.0, help="Starting playback speed (default: 10.0)")
    parser.add_argument("--end-speed", type=float, default=1.0, help="Ending playback speed (default: 1.0)")
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Duration in seconds for the slowdown (default: 30.0)"
    )
    parser.add_argument(
        "--normal-duration",
        type=float,
        default=30.0,
        help="Duration in seconds of normal speed playback after slowdown (default: 30.0)",
    )

    args = parser.parse_args()

    # Build full input path from songs/ folder
    input_path = SONGS_DIR / args.input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Generate output filename if not provided
    if args.output is None:
        base, _ = os.path.splitext(args.input)
        output_filename = f"{base}_trivia.mp3"
    else:
        output_filename = args.output

    # Build full output path in songs_final/ folder
    output_path = SONGS_FINAL_DIR / output_filename

    # Ensure output directory exists
    SONGS_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    process_audio(
        str(input_path),
        str(output_path),
        start_speed=args.start_speed,
        end_speed=args.end_speed,
        duration=args.duration,
        normal_duration=args.normal_duration,
    )

    return 0


if __name__ == "__main__":
    exit(main())
