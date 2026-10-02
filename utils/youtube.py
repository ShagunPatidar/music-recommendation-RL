"""Finds a YouTube video id for a song so it can be embedded and auto-played.

Uses yt-dlp only to *search* (no download, no API key). Results are cached in
data/youtube_ids.json, so each song is looked up just once. If anything fails
(no internet, yt-dlp missing) an empty string is returned and the app falls back
to the plain 'Listen on YouTube' link.
"""
import json
import os


def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def get_video_id(song_id, song, artist, cache_path):
    key = str(song_id)
    cache = _load(cache_path)
    if cache.get(key):
        return cache[key]
    try:
        import yt_dlp
        opts = {"quiet": True, "no_warnings": True, "extract_flat": True,
                "skip_download": True, "socket_timeout": 6, "retries": 0}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{song} {artist} official audio", download=False)
        entries = info.get("entries") or []
        vid = entries[0].get("id", "") if entries else ""
    except Exception:
        return ""
    if vid:
        cache[key] = vid
        try:
            with open(cache_path, "w") as fh:
                json.dump(cache, fh)
        except OSError:
            pass
    return vid or ""
