# VideoForge MVP

Local-first video editing automation pipeline. This MVP provides a runnable skeleton with clear module boundaries, adapters, and a timeline.json export used by both DaVinci Resolve and EDL. It also includes a Resolve Fusion Script UI entry point for in-app execution.

## Quickstart

Requirements:
- Python 3.11+
- ffmpeg + ffprobe on PATH
- (Optional) faster-whisper, open-clip-torch, faiss-cpu, DaVinci Resolve scripting

Setup:
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

Index a B-roll library (use a real folder path):
```
videoforge library scan E:\Media\Broll --library VideoForge\data\cache\library.db
```

Analyze a main video:
```
videoforge analyze path\to\main.mp4 --stage all
```

Match B-roll:
```
videoforge match VideoForge\data\projects\<project_id>.db --library VideoForge\data\cache\library.db
```

Build timeline.json:
```
videoforge build VideoForge\data\projects\<project_id>.db --main path\to\main.mp4 --output timeline.json
```

Export:
```
videoforge export timeline.json --to resolve
videoforge export timeline.json --format edl --output output.edl
```

Debug:
```
videoforge debug VideoForge\data\projects\<project_id>.db --show-segments
```

## Notes
- AutoResolve interactions are encapsulated in adapters.
- The pipeline uses timeline.json as the single export interface.
- Optional dependencies degrade gracefully.

## Resolve Plugin
Install the Resolve Fusion Script plugin using `INSTALL_PLUGIN.md`. Run it from Workspace > Scripts > Comp > VideoForge > VideoForge.

Plugin workflow (inside Resolve):
1. Select a timeline clip.
2. Click "Analyze Selected Clip".
3. Set or scan the B-roll library.
4. Click "Match B-roll".
5. Click "Apply to Timeline".

CLI vs Plugin:
- CLI runs externally and exports `timeline.json`, which you then import or export to Resolve/EDL.
- Plugin runs inside Resolve and applies results directly to the active timeline.

## Credits

VideoForge is built upon MIT-licensed open source projects:

- AutoResolve by HawzhinBlanca
  https://github.com/HawzhinBlanca/AutoResolve
- Video-Editing-Automation by ap-atul
  https://github.com/ap-atul/Video-Editing-Automation
- eddie-smart-video-editor by ZoeDekraker
  https://github.com/ZoeDekraker/eddie-smart-video-editor
- pydavinci by Pedro Labonia
  https://github.com/pedrolabonia/pydavinci

## License

MIT
