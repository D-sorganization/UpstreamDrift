# Player Reference Video Acquisition Commands

Set `MEDIA_ROOT` to a private, non-Git directory before running these commands.
The commands acquire public sources and write files below `MEDIA_ROOT`; they do
not train or validate a motion model.

## Install the Download Tool

```bash
python3 -m pip install --target /tmp/yt-dlp-task yt-dlp==2026.11.23
```

## Download Tiger Compilation

This saves format 299 as the original H.264 1080p50 video-only stream with
metadata and uploader description sidecars.

```bash
PYTHONPATH=/tmp/yt-dlp-task python3 -m yt_dlp --no-playlist --js-runtimes node --remote-components ejs:github --write-info-json --write-description -f 299 -o "$MEDIA_ROOT/originals/%(id)s.%(ext)s" 'https://www.youtube.com/watch?v=t_J6Vik3Tss'
```

## Acquire the USGA Section

The downloader was asked for source presentation seconds 3250–3295 and format
398+251. The initial section file has container `start_time=-0.007` and first
video packet PTS `5.881` relative to that file. This is not an independently
verified offset to the full video's PTS; do not infer the source start from it.

```bash
PYTHONPATH=/tmp/yt-dlp-task python3 -m yt_dlp --no-playlist --js-runtimes node --remote-components ejs:github --write-info-json --write-description --download-sections '*3250-3295' -f '398+251' --merge-output-format mkv -o "$MEDIA_ROOT/originals/%(id)s-3250-3295.%(ext)s" 'https://www.youtube.com/watch?v=bado2QdgD3c'
```

## Create the Exploratory Stream-Copy Variant

This local derivative seeks within the section file and does not establish its
alignment to the full-source presentation clock.

```bash
ffmpeg -hide_banner -v error -ss 5.888 -i "$MEDIA_ROOT/originals/bado2QdgD3c-3250-3295.mkv" -t 45.025 -map 0 -c copy -avoid_negative_ts make_zero "$MEDIA_ROOT/originals/bado2QdgD3c-3250-3295-bounded.mp4"
```

## Acquire a Forced-Cut Presentation Variant

This separate acquisition uses yt-dlp's forced keyframe cuts and re-encodes the
video. Keep the initial section file as a separate source artifact.

```bash
PYTHONPATH=/tmp/yt-dlp-task python3 -m yt_dlp --no-playlist --js-runtimes node --remote-components ejs:github --write-info-json --write-description --download-sections '*3250-3295' --force-keyframes-at-cuts -f '398+251' --merge-output-format mkv -o "$MEDIA_ROOT/originals/bado2QdgD3c-requested-3250-3295-forced-cut.%(ext)s" 'https://www.youtube.com/watch?v=bado2QdgD3c'
```

## Download the Hogan Compilation

The selected formats are remuxed without transcoding into MP4; keep the MKV
stream merge as well.

```bash
PYTHONPATH=/tmp/yt-dlp-task python3 -m yt_dlp --no-playlist --js-runtimes node --remote-components ejs:github --write-info-json --write-description -f '399+140-1' --merge-output-format mkv -o "$MEDIA_ROOT/originals/%(id)s.%(ext)s" 'https://www.youtube.com/watch?v=DJDYMjmvFwg'
ffmpeg -hide_banner -v error -i "$MEDIA_ROOT/originals/DJDYMjmvFwg.mkv" -map 0 -c copy -movflags +faststart "$MEDIA_ROOT/originals/DJDYMjmvFwg.mp4"
```

## Validate Acquired Files

```bash
sha256sum "$MEDIA_ROOT"/originals/*
ffprobe -v error -show_entries format=start_time,duration:stream=codec_type,codec_name,width,height,start_time,duration,time_base,r_frame_rate,avg_frame_rate -of json "$MEDIA_ROOT/originals/bado2QdgD3c-requested-3250-3295-forced-cut.mkv"
```
