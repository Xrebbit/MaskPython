import logging
import math
import os
import shutil
import threading
import time
import webbrowser

import cv2
import numpy as np
from flask import Flask, Response, abort, render_template_string

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_FACES = 2

logging.getLogger("werkzeug").setLevel(logging.ERROR)


def make_detector():
    cv_dir = os.path.dirname(cv2.__file__)
    src = os.path.join(cv_dir, "data", "haarcascade_frontalface_default.xml")
    dst_dir = r"C:\cascades"
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "face.xml")
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    return cv2.CascadeClassifier(dst)


def find_file(*names):
    for name in names:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            return path
    raise SystemExit("Положи рядом с main.py файл: " + " или ".join(names))


def load_sprite(path, soft=False):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:
        bgr, alpha = img[:, :, :3], img[:, :, 3]
    else:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        bgr = np.ascontiguousarray(img[:, :, :3])
        h, w = bgr.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        cv2.floodFill(bgr.copy(), mask, (0, 0), (0, 0, 0), (12, 12, 12), (12, 12, 12), flags)
        alpha = np.where(mask[1:-1, 1:-1] == 255, 0, 255).astype(np.uint8)
        if soft:
            alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8), iterations=1)
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    sprite = np.dstack([bgr, alpha])
    ys, xs = np.where(sprite[:, :, 3] > 10)
    return sprite[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def overlay(frame, sprite, cx, cy, size, smooth=False):
    size = max(int(size), 4)
    h0, w0 = sprite.shape[:2]
    new_w = size
    new_h = max(int(size * h0 / w0), 1)
    if not smooth:
        interp = cv2.INTER_NEAREST
    elif new_w < w0:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR
    s = cv2.resize(sprite, (new_w, new_h), interpolation=interp)
    x1, y1 = int(cx - new_w / 2), int(cy - new_h / 2)
    x2, y2 = x1 + new_w, y1 + new_h
    fh, fw = frame.shape[:2]
    sx1, sy1 = max(0, -x1), max(0, -y1)
    sx2, sy2 = new_w - max(0, x2 - fw), new_h - max(0, y2 - fh)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = x1 + (sx2 - sx1), y1 + (sy2 - sy1)
    if x2 <= x1 or y2 <= y1:
        return
    part = s[sy1:sy2, sx1:sx2]
    a = part[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (part[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)


HEARTS_LAYOUT = [
    (155, 0.96, 0.13, 0.0),
    (140, 1.10, 0.10, 1.0),
    (126, 1.00, 0.24, 2.0),
    (108, 1.12, 0.16, 3.0),
    (90, 1.02, 0.12, 4.0),
    (72, 1.12, 0.28, 5.0),
    (54, 1.00, 0.17, 6.0),
    (40, 1.10, 0.10, 7.0),
    (26, 0.96, 0.24, 8.0),
]

MASKS = {
    "pink_hearts": {"title": "Розовые сердца"},
    "clown": {"title": "Клоун"},
}


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = make_detector()
        self.sprites = {
            "heart": load_sprite(find_file("heart.png", "hurts.png")),
            "clown_nose": load_sprite(find_file("clownose.png"), soft=True),
            "clown_hair": load_sprite(find_file("clownhair.png"), soft=True),
        }
        self.current = None
        self.tracks = []
        self.frame = None
        self.jpeg = None
        self.lock = threading.Lock()
        threading.Thread(target=self.run, daemon=True).start()

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = 0.5
        small = cv2.resize(gray, None, fx=scale, fy=scale)
        faces = self.detector.detectMultiScale(small, 1.1, 5, minSize=(50, 50))
        now = time.time()
        found = [np.array(f, dtype=np.float32) / scale for f in faces]
        found.sort(key=lambda b: b[2] * b[3], reverse=True)
        found = found[:MAX_FACES]
        used = set()
        for box in found:
            cx = box[0] + box[2] / 2
            cy = box[1] + box[3] / 2
            best = None
            best_dist = None
            for i, track in enumerate(self.tracks):
                if i in used:
                    continue
                tb = track["box"]
                dist = math.hypot(cx - (tb[0] + tb[2] / 2), cy - (tb[1] + tb[3] / 2))
                if dist < max(tb[2], box[2]) and (best_dist is None or dist < best_dist):
                    best = i
                    best_dist = dist
            if best is None:
                self.tracks.append({"box": box, "seen": now})
                used.add(len(self.tracks) - 1)
            else:
                track = self.tracks[best]
                track["box"] = track["box"] * 0.7 + box * 0.3
                track["seen"] = now
                used.add(best)
        alive = [t for t in self.tracks if now - t["seen"] <= 0.6]
        alive.sort(key=lambda t: t["seen"], reverse=True)
        self.tracks = alive[:MAX_FACES]

    def draw_hearts(self, frame, box, index, now):
        x, y, w, h = box
        sprite = self.sprites["heart"]
        cx = x + w / 2
        cy = y + h * 0.25
        rx = w * 0.72
        ry = h * 0.78
        t = now * 2.0 + index * 1.7
        for angle, radial, size, phase in HEARTS_LAYOUT:
            a = math.radians(angle)
            px = cx + rx * radial * math.cos(a)
            py = cy - ry * radial * math.sin(a) + math.sin(t + phase) * h * 0.02
            overlay(frame, sprite, px, py, w * size)

    def draw_clown(self, frame, box):
        x, y, w, h = box
        cx = x + w / 2
        hair = self.sprites["clown_hair"]
        hair_w = w * 1.5
        hair_h = hair_w * hair.shape[0] / hair.shape[1]
        hair_bottom = y + h * 0.12
        overlay(frame, hair, cx, hair_bottom - hair_h / 2, hair_w, smooth=True)
        overlay(frame, self.sprites["clown_nose"], cx, y + h * 0.62, w * 0.22, smooth=True)

    def draw_mask(self, frame):
        current = self.current
        if current is None:
            return
        now = time.time()
        for index, track in enumerate(list(self.tracks)):
            if now - track["seen"] > 0.6:
                continue
            if current == "pink_hearts":
                self.draw_hearts(frame, track["box"], index, now)
            elif current == "clown":
                self.draw_clown(frame, track["box"])

    def run(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)
            self.detect_faces(frame)
            self.draw_mask(frame)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                with self.lock:
                    self.frame = frame
                    self.jpeg = buf.tobytes()

    def get_jpeg(self):
        with self.lock:
            return self.jpeg

    def get_png(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".png", frame)
        return buf.tobytes() if ok else None


PAGE = """
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>Маски для камеры</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  background:
    radial-gradient(circle at 15% 10%, rgba(37, 99, 235, 0.25), transparent 45%),
    radial-gradient(circle at 85% 90%, rgba(29, 78, 216, 0.22), transparent 45%),
    #04060c;
  color: #dbe7ff;
  font-family: "Segoe UI", Arial, sans-serif;
  display: flex;
  align-items: center;
  justify-content: center;
}
#stage {
  position: relative;
  width: min(96vw, 160vh * 0.75);
  aspect-ratio: 4 / 3;
  max-height: 96vh;
  background: #000;
  border: 1px solid #1e3a8a;
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 0 40px rgba(37, 99, 235, 0.35), 0 0 120px rgba(29, 78, 216, 0.18);
}
#stage:fullscreen {
  width: 100vw;
  height: 100vh;
  max-height: none;
  aspect-ratio: auto;
  border: none;
  border-radius: 0;
}
#video { width: 100%; height: 100%; object-fit: contain; display: block; background: #000; }
#flash { position: absolute; inset: 0; background: #fff; opacity: 0; pointer-events: none; }
#flash.on { animation: flash 0.35s ease-out; }
@keyframes flash { from { opacity: 0.9; } to { opacity: 0; } }
.top {
  position: absolute; top: 0; left: 0; right: 0;
  display: flex; justify-content: space-between; align-items: flex-start;
  padding: 16px 18px;
  background: linear-gradient(to bottom, rgba(2, 6, 23, 0.75), transparent);
  z-index: 5;
}
.dropdown { position: relative; }
#menuBtn {
  display: flex; align-items: center; gap: 10px;
  padding: 9px 16px; border-radius: 12px;
  background: rgba(15, 30, 70, 0.55);
  border: 1px solid #1d4ed8;
  color: #93c5fd; cursor: pointer;
  font-family: inherit; font-size: 16px; font-weight: 600; letter-spacing: 0.04em;
  text-shadow: 0 0 12px rgba(59, 130, 246, 0.8);
  transition: 0.2s;
}
#menuBtn:hover { background: rgba(29, 78, 216, 0.5); box-shadow: 0 0 16px rgba(59, 130, 246, 0.7); }
#menuBtn svg { width: 16px; height: 16px; transition: transform 0.2s; }
.dropdown.open #menuBtn svg { transform: rotate(180deg); }
#menu {
  position: absolute; left: 0; top: calc(100% + 8px);
  min-width: 210px;
  background: rgba(5, 12, 35, 0.95);
  border: 1px solid #1d4ed8;
  border-radius: 14px;
  padding: 6px;
  display: none;
  box-shadow: 0 0 28px rgba(37, 99, 235, 0.55);
  backdrop-filter: blur(8px);
}
.dropdown.open #menu { display: block; }
.opt {
  width: 100%; display: block;
  padding: 10px 14px; border-radius: 10px;
  background: transparent; border: none;
  color: #bfdbfe; cursor: pointer; font-size: 15px; text-align: left;
  transition: 0.15s;
}
.opt:hover { background: rgba(37, 99, 235, 0.35); }
.opt.active { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: #fff; }
.icon-btn {
  width: 42px; height: 42px; border-radius: 12px;
  background: rgba(15, 30, 70, 0.7);
  border: 1px solid #1d4ed8;
  color: #bfdbfe; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: 0.2s;
}
.icon-btn:hover { background: #1d4ed8; box-shadow: 0 0 16px rgba(59, 130, 246, 0.7); }
.icon-btn svg { width: 22px; height: 22px; }
.bottom {
  position: absolute; left: 0; right: 0; bottom: 0;
  padding: 18px 18px 22px;
  display: flex; flex-direction: column; align-items: center; gap: 16px;
  background: linear-gradient(to top, rgba(2, 6, 23, 0.85), transparent);
}
.row { width: 100%; display: flex; align-items: center; justify-content: center; position: relative; }
#shutter {
  width: 78px; height: 78px; border-radius: 50%;
  background: transparent;
  border: 4px solid #dbeafe;
  padding: 5px; cursor: pointer;
  box-shadow: 0 0 24px rgba(59, 130, 246, 0.8);
  transition: transform 0.12s;
}
#shutter span { display: block; width: 100%; height: 100%; border-radius: 50%; background: #fff; transition: 0.12s; }
#shutter:hover span { background: #bfdbfe; }
#shutter:active { transform: scale(0.92); }
#shutter:active span { background: #60a5fa; }
#thumb {
  position: absolute; left: 4px; bottom: 4px;
  width: 58px; height: 58px; border-radius: 12px;
  border: 2px solid #2563eb; object-fit: cover;
  display: none; cursor: pointer;
  box-shadow: 0 0 14px rgba(37, 99, 235, 0.7);
}
</style>
</head>
<body>
<div id="stage">
  <img id="video" src="/video" alt="">
  <div id="flash"></div>
  <div class="top">
    <div class="dropdown" id="dropdown">
      <button id="menuBtn">
        <span>МАСКИ</span>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M6 9l6 6 6-6"/>
        </svg>
      </button>
      <div id="menu">
        <button class="opt active" data-mask="none">Без маски</button>
        {% for key, title in masks %}
        <button class="opt" data-mask="{{ key }}">{{ title }}</button>
        {% endfor %}
      </div>
    </div>
    <button class="icon-btn" id="full" title="На весь экран (F)">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>
      </svg>
    </button>
  </div>
  <div class="bottom">
    <div class="row">
      <img id="thumb" alt="">
      <button id="shutter" title="Фото (Пробел)"><span></span></button>
    </div>
  </div>
</div>
<script>
const stage = document.getElementById('stage');
const dropdown = document.getElementById('dropdown');
const menuBtn = document.getElementById('menuBtn');
const opts = document.querySelectorAll('.opt');
const flash = document.getElementById('flash');
const thumb = document.getElementById('thumb');
let lastUrl = null;

menuBtn.addEventListener('click', e => {
  e.stopPropagation();
  dropdown.classList.toggle('open');
});

document.addEventListener('click', () => dropdown.classList.remove('open'));

opts.forEach(opt => {
  opt.addEventListener('click', () => {
    opts.forEach(o => o.classList.remove('active'));
    opt.classList.add('active');
    dropdown.classList.remove('open');
    fetch('/mask/' + opt.dataset.mask, { method: 'POST' });
  });
});

function toggleFull() {
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else {
    stage.requestFullscreen();
  }
}

async function shoot() {
  flash.classList.remove('on');
  void flash.offsetWidth;
  flash.classList.add('on');
  const res = await fetch('/snapshot');
  if (!res.ok) return;
  const blob = await res.blob();
  if (lastUrl) URL.revokeObjectURL(lastUrl);
  lastUrl = URL.createObjectURL(blob);
  thumb.src = lastUrl;
  thumb.style.display = 'block';
  const a = document.createElement('a');
  a.href = lastUrl;
  a.download = 'photo_' + Date.now() + '.png';
  a.click();
}

thumb.addEventListener('click', () => window.open(lastUrl, '_blank'));
document.getElementById('full').addEventListener('click', toggleFull);
document.getElementById('shutter').addEventListener('click', shoot);
document.addEventListener('keydown', e => {
  if (e.key === 'f' || e.key === 'F' || e.key === 'а' || e.key === 'А') toggleFull();
  if (e.code === 'Space') { e.preventDefault(); shoot(); }
  if (e.key === 'Escape') dropdown.classList.remove('open');
});
</script>
</body>
</html>
"""

app = Flask(__name__)
camera = Camera()


@app.route("/")
def index():
    masks = [(k, v["title"]) for k, v in MASKS.items()]
    return render_template_string(PAGE, masks=masks)


@app.route("/video")
def video():
    def stream():
        while True:
            jpg = camera.get_jpeg()
            if jpg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.03)

    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/mask/<key>", methods=["POST"])
def set_mask(key):
    if key == "none":
        camera.current = None
    elif key in MASKS:
        camera.current = key
    else:
        abort(404)
    return ("", 204)


@app.route("/snapshot")
def snapshot():
    png = camera.get_png()
    if png is None:
        abort(503)
    return Response(png, mimetype="image/png")


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


if __name__ == "__main__":
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=5000, threaded=True)