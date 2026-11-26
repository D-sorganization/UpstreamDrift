# Legend Reference Video Acquisition

The [historical capture procedure](../historical-capture-procedure.md) defines
the evidence and timing limits for reuse. The public receipt records source links,
file hashes, probe metadata and reviewed presentation times. Keep acquired video,
yt-dlp sidecars and sampled frames under a private, non-Git `MEDIA_ROOT`.

## Acquire

Use Python 3 and Node.js for yt-dlp's YouTube JavaScript challenge support. These
commands save the acquired platform rendition and source metadata; they do not
download camera masters, establish historic capture clocks, or train a model.

```bash
python3 -m venv /tmp/legend-yt-dlp-env
/tmp/legend-yt-dlp-env/bin/pip install yt-dlp==2026.11.23
YTDLP=/tmp/legend-yt-dlp-env/bin/yt-dlp
```

Run each command from a shell where `MEDIA_ROOT` names a private directory.
The format IDs below reproduce the selected rendition as observed on the
acquisition date; availability can change at the source.

```bash
"$YTDLP" --no-playlist --js-runtimes node --write-info-json --write-description -f '136+140' --merge-output-format mp4 -o "$MEDIA_ROOT/seve-alt/%(id)s.%(ext)s" 'https://www.youtube.com/watch?v=iKdLRJHqNSw'
"$YTDLP" --no-playlist --js-runtimes node --write-info-json --write-description -f '18' --merge-output-format mp4 -o "$MEDIA_ROOT/seve-ballesteros/%(id)s.%(ext)s" 'https://www.youtube.com/watch?v=7kOYdCAcpsA'
"$YTDLP" --no-playlist --js-runtimes node --write-info-json --write-description -f '616+140' --merge-output-format mp4 -o "$MEDIA_ROOT/jack-nicklaus/%(id)s.%(ext)s" 'https://www.youtube.com/watch?v=L-Oc2isGu4E'
"$YTDLP" --no-playlist --js-runtimes node --write-info-json --write-description -f '398+140-17' --merge-output-format mp4 -o "$MEDIA_ROOT/bobby-jones/%(id)s.%(ext)s" 'https://www.youtube.com/watch?v=LJsJQ0nXAxw'
"$YTDLP" --no-playlist --js-runtimes node --write-info-json --write-description -f '398+140' --merge-output-format mp4 -o "$MEDIA_ROOT/sam-snead/%(id)s.%(ext)s" 'https://www.youtube.com/watch?v=sS39129T7P0'
"$YTDLP" --no-playlist --js-runtimes node --write-info-json --write-description -f '397+140' --merge-output-format mp4 -o "$MEDIA_ROOT/byron-nelson/%(id)s.%(ext)s" 'https://www.youtube.com/watch?v=vxN7YIgibZY'
```

## Verify

Compare each SHA-256 value with the receipt before analyzing a file. The probe
reports container and stream timing separately; presentation time is not a
physical clock for these historical recordings.

```bash
sha256sum "$MEDIA_ROOT"/*/*.mp4
ffprobe -v error -show_entries format=start_time,duration,size:stream=codec_type,codec_name,width,height,start_time,duration,time_base,r_frame_rate -of json "$MEDIA_ROOT/jack-nicklaus/L-Oc2isGu4E.mp4"
```

Representative frames were inspected locally and remain private. No video or
frame image is included in this repository. Rights are unknown and publication
is not permitted by this receipt. The Snead and Nelson sources are uploader
attributions, not verified archival provenance; their recording dates and event
lineages remain unresolved.
