# Trivia

Scripts that turn images, sentences, and songs into guess-the-answer videos and audio. Each one starts unrecognisable and reveals the answer over about 30 seconds.

## Setup

Python 3.10 or newer. `music_slowdown.py` also needs [ffmpeg](https://ffmpeg.org/) on your PATH so pydub can read and write MP3s.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put source files in the input folders below. The scripts create the output folders themselves. Both sets of folders are gitignored.

| Script | Reads from | Writes to |
| --- | --- | --- |
| `image_reveal.py`, `image_zoom.py` | `images/` | `images_final/` |
| `sentence_reveal.py` | the sentence you pass in | `sentences_final/` |
| `music_slowdown.py` | `songs/` | `songs_final/` |

## Scripts

### `image_reveal.py`

Builds an MP4 that starts pixelated and blurred, then clears.

```bash
python image_reveal.py photo.jpg
python image_reveal.py photo.png -d 15
python image_reveal.py photo.jpg --effect blur       # blur, pixelate, or both
python image_reveal.py photo.jpg --start-blocks 3    # fewer blocks = coarser start
```

Output: `images_final/photo_reveal.mp4`.

| Flag | Default | What it does |
| --- | --- | --- |
| `-d`, `--duration` | `30` | Length in seconds |
| `--fps` | `30` | Frames per second |
| `--effect` | `both` | `blur`, `pixelate`, or `both` |
| `--start-blocks` | `4` | Mosaic blocks across the short side at the start. Lower is harder |
| `--blur-strength` | `0.35` | Blur relative to the current block size |
| `--ease` | `1.5` | `1` reveals at a steady rate; higher lingers on the coarse mosaic |

### `image_zoom.py`

Builds an MP4 that starts zoomed into the centre of an image and pulls back to the full frame.

```bash
python image_zoom.py photo.jpg
python image_zoom.py photo.jpg -d 20
python image_zoom.py photo.jpg --start-zoom 60
python image_zoom.py photo.jpg --ease 1.4
```

Output: `images_final/photo_zoom.mp4`.

| Flag | Default | What it does |
| --- | --- | --- |
| `-d`, `--duration` | `30` | Seconds spent zooming out |
| `--fps` | `30` | Frames per second |
| `--start-zoom` | `40` | Starting magnification. Higher is a tighter crop |
| `--hold` | `2` | Extra seconds on the full image |
| `--ease` | `1` | `1` zooms at a steady rate; higher lingers on the close-up |

### `sentence_reveal.py`

Builds an MP4 where letters appear one at a time in random order. Spaces and punctuation stay visible as clues. OpenCV fonts are ASCII-only, so accents are stripped (`á` becomes `a`) and `¿` / `¡` are flipped.

```bash
python sentence_reveal.py "the price is right"
python sentence_reveal.py "guess the movie" -d 45
python sentence_reveal.py "hello world" --vertical
python sentence_reveal.py "hello world" --seed 7 -o hello.mp4
```

Output: `sentences_final/<slug>_reveal.mp4`, or the name you pass with `-o`.

| Flag | Default | What it does |
| --- | --- | --- |
| `-o`, `--output` | slug of the sentence | Filename inside `sentences_final/` |
| `-d`, `--duration` | `30` | Seconds until every letter is visible |
| `--hold` | `3` | Extra seconds on the finished sentence |
| `--fps` | `30` | Frames per second |
| `--width`, `--height` | `1920`, `1080` | Frame size |
| `--vertical` | off | Use `1080x1920` instead |
| `--fade` | `0.4` | Seconds each letter takes to fade in |
| `--seed` | random | Fixed reveal order |
| `--no-progress-bar` | off | Hide the bar at the bottom |

### `music_slowdown.py`

Builds an MP3 that starts sped up and slows to normal speed, then plays a stretch of the song at normal speed.

```bash
python music_slowdown.py song.mp3
python music_slowdown.py song.mp3 -o custom_output.mp3
python music_slowdown.py song.mp3 --start-speed 5 --duration 20
```

Output: `songs_final/song_trivia.mp3`. The source needs to be long enough for the sped-up section (about 165 seconds at the defaults, since 30 seconds from 10x down to 1x consumes the average of those speeds). If it is shorter, the script warns and trims the output.

| Flag | Default | What it does |
| --- | --- | --- |
| `-o`, `--output` | `<name>_trivia.mp3` | Filename inside `songs_final/` |
| `--start-speed` | `10` | Playback speed at the start |
| `--end-speed` | `1` | Playback speed at the end of the slowdown |
| `--duration` | `30` | Seconds the slowdown lasts |
| `--normal-duration` | `30` | Seconds of normal playback after the slowdown |
