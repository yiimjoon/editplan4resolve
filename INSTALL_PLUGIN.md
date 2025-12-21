# VideoForge Resolve Plugin Install Guide

Requirements:
- DaVinci Resolve 18.6+ (Free or Studio)

## 1. Copy the folders
Copy the Resolve script UI folder, plus the core `VideoForge` package folder.

Windows scripts path:
%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Comp\VideoForge

Copy these into the scripts path:
- `scripts/Comp/VideoForge/*` (contains `VideoForge.py` and `ui/`)
- `VideoForge/` (the core package folder from this repo)

PowerShell example (run from the repo root):
```
xcopy /E /I /Y "scripts\Comp\VideoForge" "$env:APPDATA\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Comp\VideoForge"
xcopy /E /I /Y "VideoForge" "$env:APPDATA\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Comp\VideoForge\VideoForge"
```

macOS scripts path:
~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Comp/VideoForge

Linux scripts path:
~/.local/share/DaVinciResolve/Fusion/Scripts/Comp/VideoForge

Optional: instead of copying `VideoForge/`, set `VIDEOFORGE_ROOT` to the repo root path.

## 2. Install dependencies
Resolve 20.3 may use your system Python. Check Resolve Console and install into that
Python environment.

Windows (PowerShell):
```
& "C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe" -m pip install numpy PyYAML faster-whisper open-clip-torch faiss-cpu PySide6
```

macOS (Terminal):
```
/Applications/DaVinci\ Resolve/DaVinci\ Resolve.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m pip install numpy PyYAML faster-whisper open-clip-torch faiss-cpu PySide6
```

Linux (Terminal):
```
/opt/resolve/python/bin/python3 -m pip install numpy PyYAML faster-whisper open-clip-torch faiss-cpu PySide6
```

If pip is missing, run:
```
<path-to-python> -m ensurepip --upgrade
```

## 3. Install ffmpeg
Ensure ffmpeg and ffprobe are on PATH.

## 4. Restart Resolve
Restart DaVinci Resolve.

## 5. Enable scripting (if needed)
If you see "DaVinciResolveScript not available", enable scripting in:
Resolve Preferences > System > General > External scripting (Local).

If Resolve still cannot find the scripting module, set:
```
RESOLVE_SCRIPT_API=<path-to-Developer/Scripting/Modules>
```
Windows default location (Resolve 20.x):
```
C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules
```

## 6. Run the plugin
Workspace > Scripts > Comp > VideoForge > VideoForge
