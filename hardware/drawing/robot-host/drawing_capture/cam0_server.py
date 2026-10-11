#!/usr/bin/env python3
"""Serve the second SO-101 USB camera on robot localhost port 8767."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


latest_jpeg = None
frame_lock = threading.Lock()


def capture_loop():
    global latest_jpeg
    camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError("Could not open /dev/video0")
    print("cam0 streaming from /dev/video0", flush=True)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                time.sleep(0.1)
                continue
            encoded, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if encoded:
                with frame_lock:
                    latest_jpeg = jpeg.tobytes()
            time.sleep(0.03)
    finally:
        camera.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format_string, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            page = b'<html><body><h3>cam0</h3><img src="/cam0" width="640"></body></html>'
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if self.path != "/cam0":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with frame_lock:
                    frame = latest_jpeg
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame + b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", 8767), Handler).serve_forever()
