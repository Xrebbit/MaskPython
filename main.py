import glob
import math
import os
import shutil
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def make_detector():
    cv_dir = os.path.dirname(cv2.__file__)
    src = os.path.join(cv_dir, "data", "haarcascade_frontalface_default.xml")
    dst_dir = r"C:\cascades"
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "face.xml")
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    return cv2.CascadeClassifier(dst)


def find_sprite():
    for name in ("heart.png", "hurts.png"):
        p = os.path.join(BASE_DIR, name)
        if os.path.exists(p):
            return p
    files = [f for f in glob.glob(os.path.join(BASE_DIR, "*.png"))
             if not os.path.basename(f).startswith("photo_")]
    if files:
        return files[0]
    tmp = tk.Tk()
    tmp.withdraw()
    p = filedialog.askopenfilename(
        title="Выбери картинку сердца",
        filetypes=[("Картинки", "*.png *.jpg *.jpeg *.bmp")],
    )
    tmp.destroy()
    if not p:
        raise SystemExit("Картинка не выбрана")
    return p


def load_sprite(path):
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
    sprite = np.dstack([bgr, alpha])
    ys, xs = np.where(sprite[:, :, 3] > 10)
    return sprite[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def overlay(frame, sprite, cx, cy, size):
    size = max(int(size), 4)
    h0, w0 = sprite.shape[:2]
    new_w = size
    new_h = int(size * h0 / w0)
    s = cv2.resize(sprite, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
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
    "pink_hearts": {
        "title": "Розовые сердца",
        "sprite": find_sprite(),
        "layout": HEARTS_LAYOUT,
    },
}


class App:
    def __init__(self, window):
        self.window = window
        window.title("Маски для камеры")
        self.cap = cv2.VideoCapture(0)
        self.detector = make_detector()
        self.sprites = {k: load_sprite(v["sprite"]) for k, v in MASKS.items()}
        self.current = None
        self.face = None
        self.last_seen = 0
        self.last_frame = None

        self.video = tk.Label(window)
        self.video.pack(side=tk.LEFT, padx=10, pady=10)

        panel = tk.Frame(window)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        tk.Label(panel, text="Маски", font=("Arial", 14, "bold")).pack(pady=(0, 10))

        for key, m in MASKS.items():
            tk.Button(
                panel, text=m["title"], width=20, height=2,
                command=lambda k=key: self.set_mask(k),
            ).pack(pady=4)

        tk.Button(panel, text="Без маски", width=20, height=2,
                  command=lambda: self.set_mask(None)).pack(pady=4)
        tk.Button(panel, text="Сделать снимок", width=20, height=2,
                  command=self.snapshot).pack(pady=(30, 4))

        window.protocol("WM_DELETE_WINDOW", self.close)
        self.update()

    def set_mask(self, key):
        self.current = key

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = 0.5
        small = cv2.resize(gray, None, fx=scale, fy=scale)
        faces = self.detector.detectMultiScale(small, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        new = np.array([x, y, w, h], dtype=np.float32) / scale
        if self.face is None:
            self.face = new
        else:
            self.face = self.face * 0.7 + new * 0.3
        self.last_seen = time.time()

    def draw_mask(self, frame):
        if self.current is None:
            return
        if self.face is None or time.time() - self.last_seen > 0.6:
            return
        x, y, w, h = self.face
        cx = x + w / 2
        cy = y + h * 0.25
        rx = w * 0.72
        ry = h * 0.78
        t = time.time() * 2.0
        mask = MASKS[self.current]
        sprite = self.sprites[self.current]
        for angle, radial, size, phase in mask["layout"]:
            a = math.radians(angle)
            px = cx + rx * radial * math.cos(a)
            py = cy - ry * radial * math.sin(a) + math.sin(t + phase) * h * 0.02
            overlay(frame, sprite, px, py, w * size)

    def update(self):
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            self.detect_face(frame)
            self.draw_mask(frame)
            self.last_frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video.configure(image=img)
            self.video.image = img
        self.window.after(15, self.update)

    def snapshot(self):
        if self.last_frame is not None:
            name = os.path.join(BASE_DIR, time.strftime("photo_%Y%m%d_%H%M%S.png"))
            ok, buf = cv2.imencode(".png", self.last_frame)
            if ok:
                buf.tofile(name)

    def close(self):
        self.cap.release()
        self.window.destroy()


if __name__ == "__main__":
    app_window = tk.Tk()
    App(app_window)
    app_window.mainloop()