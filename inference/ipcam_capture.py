"""
IPCam 图片采集模块
支持 HTTP snapshot / RTSP 流 / MJPEG 流
"""

import cv2
import time
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from threading import Thread, Lock
from typing import Optional


class IPCamCapture:
    """
    从 IP 摄像头获取图片的统一接口

    支持三种模式：
    1. HTTP Snapshot: 每次请求一张图片
    2. RTSP Stream:通过 RTSP 协议获取视频流
    3. MJPEG Stream:  通过 HTTP MJPEG 流获取图片
    """

    def __init__(
            self,
            cam_url: str,
            mode: str = "rtsp",           # "http_snapshot", "rtsp", "mjpeg"
            username: str = "",
            password: str = "",
            resolution: tuple = (640, 480),
            fps: int = 10,
    ):
        self.cam_url = cam_url
        self.mode = mode
        self.username = username
        self.password = password
        self.resolution = resolution
        self.fps = fps

        self._frame: Optional[np.ndarray] = None
        self._lock = Lock()
        self._running = False
        self._thread: Optional[Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

    #─────────────────────────────────────────────
    # 启动 / 停止
    # ─────────────────────────────────────────────
    def start(self):
        """启动后台线程持续抓帧（仅 RTSP / MJPEG 模式需要）"""
        if self.mode == "http_snapshot":
            print(f"[IPCam] HTTP snapshot mode, URL: {self.cam_url}")
            return

        self._running = True
        if self.mode == "rtsp":
            url = self._build_rtsp_url()
            print(f"[IPCam] Connecting RTSP: {url}")
            self._cap = cv2.VideoCapture(url)
            if not self._cap.isOpened():
                raise ConnectionError(f"无法连接到 RTSP 流: {url}")
        elif self.mode == "mjpeg":
            url = self.cam_url
            print(f"[IPCam] Connecting MJPEG: {url}")
            self._cap = cv2.VideoCapture(url)
            if not self._cap.isOpened():
                raise ConnectionError(f"无法连接到 MJPEG 流: {url}")

        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("[IPCam] 后台抓帧线程已启动")

    def stop(self):
        """停止后台线程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        print("[IPCam] 已停止")

    # ─────────────────────────────────────────────
    # 获取当前帧
    # ─────────────────────────────────────────────
    def get_frame_pil(self) -> Optional[Image.Image]:
        """
        获取当前帧，返回 PIL.Image (RGB)
        这是传给 OmniVLA 的核心接口
        """
        if self.mode == "http_snapshot":
            return self._fetch_http_snapshot()

        with self._lock:
            if self._frame is None:
                print("[IPCam] 警告:尚未获取到帧")
                return None
            frame = self._frame.copy()

        # OpenCV 的 BGR → RGB → PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def get_frame_cv2(self) -> Optional[np.ndarray]:
        """获取当前帧，返回 OpenCV BGR numpy array"""
        if self.mode == "http_snapshot":
            pil_img = self._fetch_http_snapshot()
            if pil_img is None:
                return None
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    # ─────────────────────────────────────────────
    # 内部实现
    # ─────────────────────────────────────────────
    def _build_rtsp_url(self) -> str:
        """构建含认证信息的 RTSP URL"""
        if self.username and self.password:
            # rtsp://user:pass@192.168.1.100:554/stream
            parts = self.cam_url.replace("rtsp://", "")
            return f"rtsp://{self.username}:{self.password}@{parts}"
        return self.cam_url

    def _capture_loop(self):
        """后台持续抓帧"""
        interval = 1.0 / self.fps
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                print("[IPCam] 读取帧失败,尝试重连...")
                time.sleep(1)
                continue

            # 可选: resize
            if self.resolution:
                frame = cv2.resize(frame, self.resolution)

            with self._lock:
                self._frame = frame

            time.sleep(interval)

    def _fetch_http_snapshot(self) -> Optional[Image.Image]:
        """HTTP 方式获取单张快照"""
        try:
            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)

            response = requests.get(self.cam_url, auth=auth, timeout=5)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content)).convert("RGB")
            if self.resolution:
                img = img.resize(self.resolution)
            return img
        except Exception as e:
            print(f"[IPCam] HTTP snapshot 失败: {e}")
            return None


# ─────────────────────────────────────────────────
# 快速测试
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    # 示例1: RTSP
    # cam = IPCamCapture("rtsp://192.168.1.100:554/stream1", mode="rtsp")

    # 示例2: HTTP Snapshot (海康/大华常见)
    # cam = IPCamCapture(
    #     "http://192.168.1.100/ISAPI/Streaming/channels/1/picture",
    #     mode="http_snapshot",
    #     username="admin",
    #     password="12345"
    # )

    # 示例3: MJPEG
    cam = IPCamCapture("http://10.99.255.235:8080/video", mode="mjpeg")

    cam.start()
    time.sleep(2)

    for i in range(5):
        pil_img = cam.get_frame_pil()
        if pil_img:
            pil_img.save(f"test_frame_{i}.jpg")
            print(f"保存帧 {i},尺寸: {pil_img.size}")
        time.sleep(1)

    cam.stop()