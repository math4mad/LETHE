#!/usr/bin/env python3
# bin/note-post.py — 通用备忘录张贴器 (Apple Notes → iCloud → iPhone)
# 用法: note-post.py <标题> <正文文件> [folder]      (正文按行张贴, 首行了不得)
# 政策承 daily_note.py 两役教训: 反斜杠翻倍、直双引号摘除 (弯引保留为文气), AppleScript 定界才安全。
# 幂等: 同名旧帖先删后立 (日报同款做法)。
import sys, subprocess, os
FOLDER_DEFAULT = "Concept-Space"

def sanitize(line):
    return line.replace("\\", "\\\\").replace('"', "")

def main():
    title, path = sys.argv[1], sys.argv[2]
    folder = sys.argv[3] if len(sys.argv) > 3 else FOLDER_DEFAULT
    body = open(path, encoding="utf-8").read()
    lines = [sanitize(ln.rstrip()) for ln in body.splitlines()] or ["(空)"]
    joined = " & return & ".join(f'"{ln}"' for ln in lines)
    script = f'''tell application "Notes"
  tell account "iCloud"
    if not (exists folder "{folder}") then
      make new folder with properties {{name:"{folder}"}}
    end if
    tell folder "{folder}"
      repeat with n in (every note whose name is "{sanitize(title)}")
        delete n
      end repeat
      make new note with properties {{name:"{sanitize(title)}", body:{joined}}}
    end tell
  end tell
end tell'''
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    ok = r.returncode == 0
    print(("✔ 帖已落备忘录: " if ok else "✘ notes 失败: ") + (r.stdout + r.stderr).strip()[:140])
    print(f"  ({len(lines)} 行 · folder={folder})")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
