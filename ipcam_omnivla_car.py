#!/usr/bin/env python3
# ===============================================================
# OmniVLA Inference: IPCam → OmniVLA → Car Control
# ===============================================================
#
# 用法示例:
#   python3 ipcam_omnivla_car.py \
#       --phone_url http://10.89.75.5:8080/shot.jpg \
#       --car_ip 10.89.75.54 \
#       --max_linear 0.05 \
#       --max_angular 0.1 \
#       --modality language \
#       --instruction "move to gray trash bin" \
#       --stop_linear_thresh 0.005 \
#       --stop_angular_thresh 0.02 \
#       --stop_consec_steps 8
#
# 停止方式:
#   1. 自动到达目标停止
#   2. 按 Enter 键主动停止
#   3. Ctrl+C 中断停止
# ===============================================================

import sys, os

sys.path.insert(0, '..')
import roslibpy
import argparse
import time
import math
import json
import io
import threading
import requests
from typing import Optional, Tuple, Type, Dict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utm

# ---------------------------
# Custom Imports
# ---------------------------
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat, L1RegressionDistHead
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE

from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor


# ===============================================================
# 命令行参数解析
# ===============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="OmniVLA: IPCam → Model Inference → Car Control"
    )
    parser.add_argument("--phone_url", type=str, required=True,
                        help="手机IP摄像头URL")
    parser.add_argument("--car_ip", type=str, required=True,
                        help="小车IP地址")
    parser.add_argument("--car_port", type=int, default=9090,
                        help="小车控制端口 (默认 9090)")
    parser.add_argument("--max_linear", type=float, default=0.3,
                        help="最大线速度 m/s")
    parser.add_argument("--max_angular", type=float, default=0.3,
                        help="最大角速度 rad/s")
    parser.add_argument("--modality", type=str, default="image",
                        choices=["language", "image", "pose", "language_pose", "pose_image", "image_only", "satellite",
                                 "satellite_image", "satellite_pose", "all"],
                        help="导航模态")
    parser.add_argument("--instruction", type=str, default="",
                        help="语言指令")
    parser.add_argument("--goal_image", type=str, default="./inference/goal_img.jpg",
                        help="目标图像路径")
    parser.add_argument("--goal_lat", type=float, default=37.8738930785863)
    parser.add_argument("--goal_lon", type=float, default=-122.26746181032362)
    parser.add_argument("--goal_compass", type=float, default=0.0)
    parser.add_argument("--current_lat", type=float, default=37.87371258374039)
    parser.add_argument("--current_lon", type=float, default=-122.26729417226024)
    parser.add_argument("--current_compass", type=float, default=270.0)
    parser.add_argument("--stop_distance", type=float, default=0.45,
                        help="waypoint停止距离阈值 (米)")
    parser.add_argument("--stop_linear_thresh", type=float, default=0.005)
    parser.add_argument("--stop_angular_thresh", type=float, default=0.02)
    parser.add_argument("--stop_consec_steps", type=int, default=3)
    parser.add_argument("--tick_rate", type=float, default=30.0)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--save_dir", type=str, default="./inference")
    parser.add_argument("--vla_path", type=str, default="./omnivla-original")
    parser.add_argument("--resume_step", type=int, default=285000)
    return parser.parse_args()


# ===============================================================
# Utility Functions
# ===============================================================
def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if (not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt"))
            and module_name == "pose_projector"):
        module_name = "proprio_projector"
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(module_class, module_name, cfg, device_id, module_args, to_bf16=False):
    module = module_class(**module_args)
    count_parameters(module, module_name)
    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return module


def clip_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


# ===============================================================
# 主动停止控制器（按 Enter 停止）
# ===============================================================
class StopController:
    """
    后台线程监听键盘输入。

    停止方式:
      - 按 Enter: 优雅停止，小车减速归零
      - Ctrl+C: 由主循环的 except KeyboardInterrupt 捕获
    """

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        print("\n" + "=" * 50)
        print("▶ 按 Enter 键可随时停止小车并退出")
        print("  ▶ 或按 Ctrl+C 强制中断")
        print("=" * 50 + "\n")

    def _listen(self):
        try:
            while not self._stop_event.is_set():
                try:
                    input()  # 阻塞等待 Enter
                    print("\n⚠⚠⚠ 收到 Enter，正在停止小车... ⚠⚠⚠\n")
                    self._stop_event.set()
                    break
                except EOFError:
                    time.sleep(1)
        except Exception:
            pass

    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def cleanup(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


# ===============================================================
# 手机摄像头
# ===============================================================
class PhoneCamera:
    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self.timeout = timeout
        print(f"[PhoneCamera] URL: {self.url}")

    def capture(self) -> Optional[Image.Image]:
        try:
            response = requests.get(self.url, timeout=self.timeout)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            return img
        except requests.exceptions.RequestException as e:
            print(f"[PhoneCamera] 抓图失败: {e}")
            return None


# ===============================================================
# 小车控制
# ===============================================================

class CarController:
    """通过 roslibpy (WebSocket) 控制小车，封装自用户已有的 RobotController"""

    def __init__(
            self,
            car_ip: str,
            car_port: int = 9090,
            cmd_vel_topic: str = "/cmd_vel",
            max_linear: float = 0.3,
            max_angular: float = 0.3,
    ):
        self.car_ip = car_ip
        self.car_port = car_port
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        # ── 连接 rosbridge ──
        print(f"[CarController] 正在连接rosbridge ws://{car_ip}:{car_port} ...")
        self.client = roslibpy.Ros(host=car_ip, port=car_port)
        self.client.run()
        if not self.client.is_connected:
            raise ConnectionError(
                f"无法连接 rosbridge ws://{car_ip}:{car_port}"
            )
        print(f"[CarController]✅ rosbridge 连接成功!")
        # ── 创建 publisher ──
        self.cmd_vel = roslibpy.Topic(
            self.client, cmd_vel_topic, 'geometry_msgs/Twist'
        )
        print(f"[CarController] ✅ Publisher 就绪 → {cmd_vel_topic}")
        # 发零速确保通道畅通
        self.stop()

    def _make_twist_msg(self, linear: float, angular: float):
        return roslibpy.Message({
            'linear': {'x': float(linear), 'y': 0.0, 'z': 0.0},
            'angular': {'x': 0.0, 'y': 0.0, 'z': float(angular)},
        })

    @staticmethod
    def clamp(val, lo, hi):
        return max(lo, min(val, hi))

    def smooth(self, prev, target, alpha=0.4):
        return prev + alpha * (target - prev)

    def send_velocity(self, linear: float, angular: float, use_smooth: bool = True) -> bool:
        """发送速度指令，带限幅+平滑"""
        try:
            # 限幅
            linear = self.clamp(linear, -self.max_linear, self.max_linear)
            angular = self.clamp(angular, -self.max_angular, self.max_angular)
            # 平滑
            if use_smooth:
                linear = self.smooth(self.prev_linear, linear)
                angular = self.smooth(self.prev_angular, angular)
            self.prev_linear = linear
            self.prev_angular = angular
            # 发布
            self.cmd_vel.publish(self._make_twist_msg(linear, angular))
            return True
        except Exception as e:
            print(f"[CarController] 发送失败: {e}")
            return False

    def stop(self) -> bool:
        """停车（连发5次确保收到）"""
        print("[CarController] >>> STOP <<<")
        try:
            zero_msg = self._make_twist_msg(0.0, 0.0)
            for _ in range(5):
                self.cmd_vel.publish(zero_msg)
                time.sleep(0.05)
            self.prev_linear = 0.0
            self.prev_angular = 0.0
            print("[CarController] 🛑 小车已停止")
            return True
        except Exception as e:
            print(f"[CarController] 停车失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        self.stop()
        try:
            self.cmd_vel.unadvertise()
            self.client.terminate()
        except Exception:
            pass
        print("[CarController] 🔌 rosbridge 已断开")


# ===============================================================
# 模态解析
# ===============================================================
def parse_modality(modality_str: str) -> dict:
    modality_map = {
        "language": {"satellite": False, "lan_prompt": True, "pose_goal": False, "image_goal": False},
        "image": {"satellite": False, "lan_prompt": False, "pose_goal": False, "image_goal": True},
        "pose": {"satellite": False, "lan_prompt": False, "pose_goal": True, "image_goal": False},
        "language_pose": {"satellite": False, "lan_prompt": True, "pose_goal": True, "image_goal": False},
        "pose_image": {"satellite": False, "lan_prompt": False, "pose_goal": True, "image_goal": True},
        "image_only": {"satellite": False, "lan_prompt": False, "pose_goal": False, "image_goal": True},
        "satellite": {"satellite": True, "lan_prompt": False, "pose_goal": False, "image_goal": False},
        "satellite_image": {"satellite": True, "lan_prompt": False, "pose_goal": False, "image_goal": True},
        "satellite_pose": {"satellite": True, "lan_prompt": False, "pose_goal": True, "image_goal": False},
        "all": {"satellite": True, "lan_prompt": False, "pose_goal": True, "image_goal": True},
    }
    return modality_map.get(modality_str, modality_map["image"])


# ===============================================================
# Waypoint-based 到达判断器
# ===============================================================
class ArrivalDetector:
    def __init__(self, stop_distance=0.45, metric_waypoint_spacing=0.1,
                 consecutive_frames_to_stop=3, negative_dx_threshold=-0.02,
                 stop_linear_thresh=0.005, stop_angular_thresh=0.02):
        self.stop_distance = stop_distance
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.consecutive_frames_to_stop = consecutive_frames_to_stop
        self.negative_dx_threshold = negative_dx_threshold
        self.stop_linear_thresh = stop_linear_thresh
        self.stop_angular_thresh = stop_angular_thresh

        self.close_count = 0
        self.overshoot_count = 0
        self.vel_stop_count = 0
        self.step_count = 0
        self.stopped = False
        self.stop_reason = ""
        self.distance_history = []

    def check(self, waypoints_raw, chosen_waypoint_idx=4,
              linear_vel=None, angular_vel=None):
        if self.stopped:
            return True

        self.step_count += 1
        waypoints_metric = waypoints_raw.copy()
        waypoints_metric[:, :2] *= self.metric_waypoint_spacing

        distances = np.sqrt(waypoints_metric[:, 0] ** 2 + waypoints_metric[:, 1] ** 2)
        max_dist = np.max(distances)
        mean_dist = np.mean(distances)
        chosen_dist = distances[chosen_waypoint_idx]
        chosen_dx = waypoints_metric[chosen_waypoint_idx, 0]
        chosen_dy = waypoints_metric[chosen_waypoint_idx, 1]

        self.distance_history.append({
            'step': self.step_count, 'max_dist': max_dist,
            'mean_dist': mean_dist, 'chosen_dist': chosen_dist,
            'chosen_dx': chosen_dx, 'chosen_dy': chosen_dy,
            'all_distances': distances.copy(),
        })

        print(f"\n{'=' * 60}")
        print(f"[ArrivalDetector] Step {self.step_count}")
        print(f"Waypoint distances (meters):")
        for i, (wp, d) in enumerate(zip(waypoints_metric, distances)):
            marker = "<<<" if i == chosen_waypoint_idx else ""
            print(f"    WP[{i}]: dx={wp[0]:+.4f}m, dy={wp[1]:+.4f}m, dist={d:.4f}m{marker}")
        print(f"  Summary: max={max_dist:.4f}m, mean={mean_dist:.4f}m, chosen={chosen_dist:.4f}m")
        print(f"  Stop threshold: {self.stop_distance:.3f}m")
        if linear_vel is not None:
            print(f"  Velocity: linear={linear_vel:.4f}, angular={angular_vel:.4f}")

        should_stop = False
        reason = ""

        # A: 轨迹塌缩
        if max_dist < self.stop_distance:
            self.close_count += 1
            reason = (f"trajectory collapsed: max_dist={max_dist:.4f}m < "
                      f"{self.stop_distance:.3f}m "
                      f"(count: {self.close_count}/{self.consecutive_frames_to_stop})")
            print(f"  [CLOSE] {reason}")
            if self.close_count >= self.consecutive_frames_to_stop:
                should_stop = True
        else:
            self.close_count = 0

        # B: 越过目标
        if chosen_dx < self.negative_dx_threshold:
            self.overshoot_count += 1
            reason_b = (f"overshoot: dx={chosen_dx:.4f}m < {self.negative_dx_threshold:.3f}m "
                        f"(count: {self.overshoot_count}/{self.consecutive_frames_to_stop})")
            print(f"  [OVERSHOOT] {reason_b}")
            if self.overshoot_count >= self.consecutive_frames_to_stop:
                should_stop = True
                reason = reason_b
        else:
            self.overshoot_count = 0

        # C: 选定waypoint近
        if chosen_dist < self.stop_distance:
            reason_c = f"chosen WP close: dist={chosen_dist:.4f}m < {self.stop_distance:.3f}m"
            print(f"  [CHOSEN CLOSE] {reason_c}")
            self.close_count += 1
            if self.close_count >= self.consecutive_frames_to_stop:
                should_stop = True
                reason = reason_c

        # D: 速度阈值
        if linear_vel is not None:
            if (abs(linear_vel) < self.stop_linear_thresh and
                    abs(angular_vel) < self.stop_angular_thresh):
                self.vel_stop_count += 1
                reason_d = (f"velocity near zero: lin={linear_vel:.4f}, ang={angular_vel:.4f} "
                            f"(count: {self.vel_stop_count}/{self.consecutive_frames_to_stop})")
                print(f"  [VEL STOP] {reason_d}")
                if self.vel_stop_count >= self.consecutive_frames_to_stop:
                    should_stop = True
                    reason = reason_d
            else:
                self.vel_stop_count = 0

        if should_stop:
            self.stopped = True
            self.stop_reason = reason
            print(f"  *** STOP TRIGGERED: {reason} ***")
            print(f"{'=' * 60}\n")
            return True

        print(f"  → Continue navigating")
        print(f"{'=' * 60}\n")
        return False

    def reset(self):
        self.close_count = 0
        self.overshoot_count = 0
        self.vel_stop_count = 0
        self.step_count = 0
        self.stopped = False
        self.stop_reason = ""
        self.distance_history = []

    def get_summary(self):
        if not self.distance_history:
            return "No navigation data."
        lines = [f"\n{'=' * 60}", "[Navigation Summary]",
                 f"  Total steps: {self.step_count}",
                 f"  Stopped: {self.stopped}"]
        if self.stopped:
            lines.append(f"  Stop reason: {self.stop_reason}")
        lines.append("  Distance history (chosen waypoint):")
        for h in self.distance_history:
            lines.append(f"    Step {h['step']:3d}: dist={h['chosen_dist']:.4f}m, "
                         f"dx={h['chosen_dx']:+.4f}m, max={h['max_dist']:.4f}m")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


# ===============================================================
# Inference Class
# ===============================================================
class Inference:
    def __init__(self, args, vla_model, action_head_model, pose_projector_model,
                 device_id, num_patches, action_tokenizer, processor, modality_flags):
        self.args = args
        self.tick_rate = args.tick_rate
        self.max_steps = args.max_steps
        self.lan_inst_prompt = args.instruction
        self.datastore_path_image = args.save_dir
        self.maxv = args.max_linear
        self.maxw = args.max_angular

        self.goal_utm = utm.from_latlon(args.goal_lat, args.goal_lon)
        self.goal_compass = -float(args.goal_compass) / 180.0 * math.pi
        self.goal_image_PIL = Image.open(args.goal_image).convert("RGB")

        self.current_lat = args.current_lat
        self.current_lon = args.current_lon
        self.current_compass = args.current_compass

        self.vla = vla_model
        self.action_head = action_head_model
        self.pose_projector = pose_projector_model
        self.device_id = device_id
        self.num_patches = num_patches
        self.action_tokenizer = action_tokenizer
        self.processor = processor

        self.lan_prompt = modality_flags["lan_prompt"]
        self.pose_goal = modality_flags["pose_goal"]
        self.satellite = modality_flags["satellite"]
        self.image_goal = modality_flags["image_goal"]

        self.camera = PhoneCamera(args.phone_url)
        self.car = CarController(
            car_ip=args.car_ip,
            car_port=args.car_port,
            cmd_vel_topic="/cmd_vel",
            max_linear=args.max_linear,
            max_angular=args.max_angular,
        )

        self.count_id = 0
        self.linear = 0.0
        self.angular = 0.0

        self.arrival_detector = ArrivalDetector(
            stop_distance=args.stop_distance,
            metric_waypoint_spacing=0.1,
            consecutive_frames_to_stop=args.stop_consec_steps,
            negative_dx_threshold=-0.02,
            stop_linear_thresh=args.stop_linear_thresh,
            stop_angular_thresh=args.stop_angular_thresh,
        )

        # 主动停止控制器
        self.stop_controller = StopController()

        os.makedirs(self.datastore_path_image, exist_ok=True)

    @staticmethod
    def calculate_relative_position(x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    @staticmethod
    def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    # ----------------------------
    # Main Loop
    # ----------------------------
    def run(self) -> bool:
        loop_time = 1.0 / self.tick_rate

        print(f"\n{'=' * 60}")
        print(f"[Navigation] Starting...")
        print(f"  Phone camera:  {self.args.phone_url}")
        print(f"  Car IP:        {self.args.car_ip}:{self.args.car_port}")
        print(f"  Modality:      {self.args.modality}")
        print(f"  Instruction:   '{self.lan_inst_prompt}'")
        print(f"  Stop dist:     {self.arrival_detector.stop_distance}m")
        print(f"  Max velocity:  linear={self.maxv}, angular={self.maxw}")
        print(f"  Tick rate:     {self.tick_rate} Hz")
        print(f"  Max steps:     {self.max_steps}")
        print(f"  Consec steps:  {self.args.stop_consec_steps}")
        print(f"{'=' * 60}\n")

        # 启动 Enter 键监听
        self.stop_controller.start()

        try:
            for step in range(self.max_steps):
                step_start = time.time()

                # ===== 检查用户是否按了 Enter =====
                if self.stop_controller.should_stop():
                    self.car.stop()
                    self.linear = 0.0
                    self.angular = 0.0
                    print(f"\n{'=' * 60}")
                    print(f"[Navigation] ■ 用户主动停止 (step {step})")
                    print(f"{'=' * 60}")
                    print(self.arrival_detector.get_summary())
                    return False

                # 执行一步推理
                self.tick()

                # 发送速度到小车
                self.car.send_velocity(self.linear, self.angular)

                # 检查自动到达
                if self.arrival_detector.stopped:
                    self.car.stop()
                    print(f"\n[Navigation] ✓ Arrived! Stopped at step {step}")
                    print(f"[Navigation] Reason: {self.arrival_detector.stop_reason}")
                    print(self.arrival_detector.get_summary())
                    return True

                # 控制频率
                elapsed = time.time() - step_start
                sleep_time = loop_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                total_elapsed = time.time() - step_start
                print(f"[Step {step}] Total: {total_elapsed:.3f}s, "
                      f"Vel: ({self.linear:.4f}, {self.angular:.4f})")
        except KeyboardInterrupt:
            print(f"\n{'=' * 60}")
            print("[Navigation] ■ Ctrl+C 中断!")
            print(f"{'=' * 60}")
            self.car.stop()
            print(self.arrival_detector.get_summary())
            return False

        finally:
            # 确保无论如何都停车+ 清理
            self.stop_controller.cleanup()
            self.car.stop()
            self.car.disconnect()

        # 超时
        print(f"\n[Navigation] ✗ Max steps ({self.max_steps}) reached")
        print(self.arrival_detector.get_summary())
        return False

    def tick(self):
        self.linear, self.angular = self.run_omnivla()

    # ----------------------------
    # OmniVLA Inference
    # ----------------------------
    def run_omnivla(self):
        thres_dist = 30.0
        metric_waypoint_spacing = 0.1

        cur_utm = utm.from_latlon(self.current_lat, self.current_lon)
        cur_compass = -float(self.current_compass) / 180.0 * math.pi

        delta_x, delta_y = self.calculate_relative_position(
            cur_utm[0], cur_utm[1], self.goal_utm[0], self.goal_utm[1])
        relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
        radius = np.sqrt(relative_x**2 + relative_y**2)
        if radius > thres_dist:
            relative_x *= thres_dist / radius
            relative_y *= thres_dist / radius

        # goal_pose_loc_norm = np.array([
        #     relative_y / metric_waypoint_spacing,
        #     -relative_x / metric_waypoint_spacing,
        #     np.cos(self.goal_compass - cur_compass),
        #     np.sin(self.goal_compass - cur_compass),
        #     ])
        goal_pose_loc_norm = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # 从手机摄像头抓图
        current_image_PIL = self.camera.capture()
        if current_image_PIL is None:
            print("[run_omnivla] Camera capture failed → (0, 0)")
            return 0.0, 0.0

        current_image_path = os.path.join(self.datastore_path_image, "current_img.jpg")
        current_image_PIL.save(current_image_path)

        lan_inst = self.lan_inst_prompt if self.lan_prompt else "xxxx"

        batch = self.data_transformer_omnivla(
            current_image_PIL, lan_inst, self.goal_image_PIL, goal_pose_loc_norm,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )

        actions, modality_id = self.run_forward_pass(
            vla=self.vla.eval(),
            action_head=self.action_head.eval(),
            noisy_action_projector=None,
            pose_projector=self.pose_projector.eval(),
            batch=batch,
            action_tokenizer=self.action_tokenizer,
            device_id=self.device_id,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_patches=self.num_patches,
        )
        self.count_id += 1

        waypoints = actions.float().cpu().numpy()

        # 计算速度
        # waypoint_select = 4
        # chosen_waypoint = waypoints[0][waypoint_select].copy()
        # chosen_waypoint[:2] *= metric_waypoint_spacing
        # dx, dy, hx, hy = chosen_waypoint

        waypoint_select = 0          # 原来是 4，改为 0
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint

        EPS = 1e-8
        #DT = 1.0 / self.tick_rate
        DT = 1.0 /10.0

        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_vel_value = 0.0
            angular_vel_value = 1.0 * clip_angle(np.arctan2(hy, hx)) / DT
        elif np.abs(dx) < EPS:
            linear_vel_value = 0.0
            angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)
        else:
            linear_vel_value = dx / DT
            angular_vel_value = np.arctan(dy / dx) / DT

        linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
        angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)

        maxv, maxw = self.maxv, self.maxw
        if np.abs(linear_vel_value) <= maxv:
            if np.abs(angular_vel_value) <= maxw:
                lv, av = linear_vel_value, angular_vel_value
            else:
                rd = linear_vel_value / angular_vel_value
                lv = maxw * np.sign(linear_vel_value) * np.abs(rd)
                av = maxw * np.sign(angular_vel_value)
        else:
            if np.abs(angular_vel_value) <= 0.001:
                lv = maxv * np.sign(linear_vel_value)
                av = 0.0
            else:
                rd = linear_vel_value / angular_vel_value
                if np.abs(rd) >= maxv / maxw:
                    lv = maxv * np.sign(linear_vel_value)
                    av = maxv * np.sign(angular_vel_value) / np.abs(rd)
                else:
                    lv = maxw * np.sign(linear_vel_value) * np.abs(rd)
                    av = maxw * np.sign(angular_vel_value)

        # 到达判断
        if self.arrival_detector.check(
                waypoints[0], chosen_waypoint_idx=waypoint_select, linear_vel=lv, angular_vel=av
        ):
            self.save_robot_behavior(
                current_image_PIL, self.goal_image_PIL, goal_pose_loc_norm,
                waypoints[0], 0.0, 0.0, metric_waypoint_spacing,
                modality_id.cpu().numpy(),
            )
            return 0.0, 0.0

        self.save_robot_behavior(
            current_image_PIL, self.goal_image_PIL, goal_pose_loc_norm,
            waypoints[0], lv, av, metric_waypoint_spacing,
            modality_id.cpu().numpy(),
        )

        print(f"[run_omnivla] linear={lv:.4f}, angular={av:.4f}")
        return lv, av
    # def run_omnivla(self):
    #     thres_dist = 30.0
    #     metric_waypoint_spacing = 0.1
    #
    #     cur_utm = utm.from_latlon(self.current_lat, self.current_lon)
    #     cur_compass = -float(self.current_compass) / 180.0 * math.pi
    #
    #     delta_x, delta_y = self.calculate_relative_position(
    #         cur_utm[0], cur_utm[1], self.goal_utm[0], self.goal_utm[1])
    #     relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
    #     radius = np.sqrt(relative_x ** 2 + relative_y ** 2)
    #     if radius > thres_dist:
    #         relative_x *= thres_dist / radius
    #         relative_y *= thres_dist / radius
    #
    #     goal_pose_loc_norm = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    #
    #     # 从手机摄像头抓图
    #     current_image_PIL = self.camera.capture()
    #     if current_image_PIL is None:
    #         print("[run_omnivla] Camera capture failed → (0, 0)")
    #         return 0.0, 0.0
    #
    #     current_image_path = os.path.join(self.datastore_path_image, "current_img.jpg")
    #     current_image_PIL.save(current_image_path)
    #
    #     lan_inst = self.lan_inst_prompt if self.lan_prompt else "xxxx"
    #
    #     # ==============================================================
    #     # 方案C：自回归解码，逐步生成 8 个 waypoint
    #     # 每步只信任第wp_idx 个预测（其上下文是干净的）
    #     # ==============================================================
    #     current_actions = np.zeros((8, 4), dtype=np.float32)
    #     final_waypoints = np.zeros((8, 4), dtype=np.float32)
    #
    #     vla_eval = self.vla.eval()
    #     action_head_eval = self.action_head.eval()
    #     pose_projector_eval = self.pose_projector.eval()
    #
    #     for wp_idx in range(8):
    #         # 构建 batch：前 wp_idx 个位置已填入预测值，后面为 zero
    #         batch = self.data_transformer_omnivla(
    #             current_image_PIL, lan_inst, self.goal_image_PIL, goal_pose_loc_norm,
    #             prompt_builder=PurePromptBuilder,
    #             action_tokenizer=self.action_tokenizer,
    #             processor=self.processor,
    #             actions=current_actions.copy(),  # ← 传入当前已有的预测
    #         )
    #
    #         actions_pred, modality_id = self.run_forward_pass(
    #             vla=vla_eval,
    #             action_head=action_head_eval,
    #             noisy_action_projector=None,
    #             pose_projector=pose_projector_eval,
    #             batch=batch,
    #             action_tokenizer=self.action_tokenizer,
    #             device_id=self.device_id, use_l1_regression=True,
    #             use_diffusion=False,
    #             use_film=False,
    #             num_patches=self.num_patches,
    #         )
    #
    #         wp_all = actions_pred.float().cpu().numpy()[0]  # (8, 4)
    #
    #         # 只取第 wp_idx 个预测（它的因果上下文完全干净）
    #         final_waypoints[wp_idx] = wp_all[wp_idx]
    #
    #         # 回填到 current_actions，作为下一步的上下文
    #         current_actions[wp_idx] = wp_all[wp_idx]
    #
    #         print(f"  [AR step {wp_idx}] dx={wp_all[wp_idx][0]:.4f}, "
    #               f"dy={wp_all[wp_idx][1]:.4f}, "
    #               f"hx={wp_all[wp_idx][2]:.4f}, "
    #               f"hy={wp_all[wp_idx][3]:.4f}")
    #
    #     self.count_id += 1
    #
    #     # ==============================================================
    #     # 现在全部 8 个 waypoint 都基于干净上下文
    #     # 可以安全使用 WP[4]（中期规划点）
    #     # ==============================================================
    #     waypoints = final_waypoints[np.newaxis, :]  # (1, 8, 4)，与原格式一致
    #
    #     waypoint_select = 4  # 现在可以放心用 WP[4]
    #     chosen_waypoint = waypoints[0][waypoint_select].copy()
    #     chosen_waypoint[:2] *= metric_waypoint_spacing
    #     dx, dy, hx, hy = chosen_waypoint
    #
    #     EPS = 1e-8
    #     DT = 1.0 / 3.0  # WP[4] 对应约0.5m处，DT = 1/3
    #
    #     base_dt = 1.0 / 3
    #     DT = (waypoint_select + 1) * base_dt  # WP[4] → 1.667s
    #     print(f"  [Vel Calc] dx={dx:.4f}, dy={dy:.4f}, DT={DT:.4f}")
    #
    #     if np.abs(dx) < EPS and np.abs(dy) < EPS:
    #         linear_vel_value = 0.0
    #         angular_vel_value = 1.0 * clip_angle(np.arctan2(hy, hx)) / DT
    #     elif np.abs(dx) < EPS:
    #         linear_vel_value = 0.0
    #         angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)
    #     else:
    #         linear_vel_value = dx / DT
    #         angular_vel_value = np.arctan(dy / dx) / DT
    #
    #     linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
    #     angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)
    #     print(f"  [Pre-limit] linear={linear_vel_value:.4f}, angular={angular_vel_value:.4f}")
    #
    #     maxv, maxw = self.maxv, self.maxw
    #     if np.abs(linear_vel_value) <= maxv:
    #         if np.abs(angular_vel_value) <= maxw:
    #             lv, av = linear_vel_value, angular_vel_value
    #         else:
    #             rd = linear_vel_value / angular_vel_value
    #             lv = maxw * np.sign(linear_vel_value) * np.abs(rd)
    #             av = maxw * np.sign(angular_vel_value)
    #     else:
    #         if np.abs(angular_vel_value) <= 0.001:
    #             lv = maxv * np.sign(linear_vel_value)
    #             av = 0.0
    #         else:
    #             rd = linear_vel_value / angular_vel_value
    #             if np.abs(rd) >= maxv / maxw:
    #                 lv = maxv * np.sign(linear_vel_value)
    #                 av = maxv * np.sign(angular_vel_value) / np.abs(rd)
    #             else:
    #                 lv = maxw * np.sign(linear_vel_value) * np.abs(rd)
    #                 av = maxw * np.sign(angular_vel_value)
    #
    #     # 到达判断
    #     if self.arrival_detector.check(
    #             waypoints[0], chosen_waypoint_idx=waypoint_select, linear_vel=lv, angular_vel=av
    #     ):
    #         self.save_robot_behavior(
    #             current_image_PIL, self.goal_image_PIL, goal_pose_loc_norm,
    #             waypoints[0], 0.0, 0.0, metric_waypoint_spacing,
    #             modality_id.cpu().numpy(),
    #         )
    #         return 0.0, 0.0
    #
    #     self.save_robot_behavior(
    #         current_image_PIL, self.goal_image_PIL, goal_pose_loc_norm,
    #         waypoints[0], lv, av, metric_waypoint_spacing,
    #         modality_id.cpu().numpy(),
    #     )
    #
    #     print(f"[run_omnivla] linear={lv:.4f}, angular={av:.4f}")
    #     return lv, av

    # ----------------------------
    # Save Behavior
    # ----------------------------
    def save_robot_behavior(self, cur_img, goal_img, goal_pose, waypoints,
                            linear_vel, angular_vel, metric_waypoint_spacing, mask_number):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2, 2)
        ax_ob = fig.add_subplot(gs[0, 0])
        ax_goal = fig.add_subplot(gs[1, 0])
        ax_graph = fig.add_subplot(gs[:, 1])

        ax_ob.imshow(np.array(cur_img).astype(np.uint8))
        ax_goal.imshow(np.array(goal_img).astype(np.uint8))

        x_seq = waypoints[:, 0]
        y_seq_inv = -waypoints[:, 1]
        ax_graph.plot(np.insert(y_seq_inv, 0, 0.0), np.insert(x_seq, 0, 0.0),
                      linewidth=4.0, markersize=12, marker='o', color='blue')

        mask_type = int(mask_number[0])
        mask_texts = ["satellite only", "pose and satellite", "satellite and image", "all",
                      "pose only", "pose and image", "image only", "language only", "language and pose"]
        if mask_type < len(mask_texts):
            ax_graph.annotate(mask_texts[mask_type], xy=(1.0, 0.0),
                              xytext=(-20, 20), fontsize=18, textcoords='offset points')

        ax_ob.set_title("Egocentric current image", fontsize=18)
        ax_goal.set_title("Egocentric goal image", fontsize=18)
        ax_graph.tick_params(axis='x', labelsize=15)
        ax_graph.tick_params(axis='y', labelsize=15)

        if int(mask_number[0]) in [1, 3, 4, 5, 8]:
            ax_graph.plot(-goal_pose[1], goal_pose[0], marker='*', color='red', markersize=15)

        ax_graph.set_xlim(-3.0, 3.0)
        ax_graph.set_ylim(-0.1, 10.0)

        waypoints_m = waypoints.copy()
        waypoints_m[:, :2] *= metric_waypoint_spacing
        dists = np.sqrt(waypoints_m[:, 0] ** 2 + waypoints_m[:, 1] ** 2)
        status = "STOPPED" if self.arrival_detector.stopped else "NAVIGATING"
        info = (f"Status: {status}\n"
                f"Chosen WP dist: {dists[4]:.3f}m\n"
                f"Max WP dist: {np.max(dists):.3f}m\n"
                f"Stop threshold: {self.arrival_detector.stop_distance:.3f}m\n"
                f"Vel: ({linear_vel:.4f}, {angular_vel:.4f})")
        color = 'wheat' if status == "NAVIGATING" else 'lightgreen'
        ax_graph.text(-2.8, 8.5, info, fontsize=14, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        ax_graph.set_title("Normalized generated 2D trajectories from OmniVLA", fontsize=18)

        save_path = os.path.join(self.datastore_path_image, f"{self.count_id}_ex.jpg")
        plt.savefig(save_path)
        plt.close(fig)

    # ----------------------------
    # Collator
    # ----------------------------
    def collator_custom(self, instances, model_max_length, pad_token_id,
                        padding_side="right", pixel_values_dtype=torch.float32):
        IGNORE_INDEX = -100
        input_ids = pad_sequence([inst["input_ids"] for inst in instances],
                                 batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([inst["labels"] for inst in instances],
                              batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :model_max_length]
        labels = labels[:, :model_max_length]
        attention_mask = input_ids.ne(pad_token_id)

        pixel_values = [inst["pixel_values_current"] for inst in instances]
        dataset_names = ([inst["dataset_name"] for inst in instances] if "dataset_name" in instances[0] else None)

        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_goal" in instances[0]:
                pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
                pixel_values = torch.cat(
                    (torch.stack(pixel_values), torch.stack(pixel_values_goal)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported pixel_values type: {type(pixel_values)}")

        actions = torch.stack([torch.from_numpy(np.copy(inst["actions"])) for inst in instances])
        goal_pose = torch.stack([torch.from_numpy(np.copy(inst["goal_pose"])) for inst in instances])

        output = dict(pixel_values=pixel_values, input_ids=input_ids,
                      attention_mask=attention_mask, labels=labels,
                      actions=actions, goal_pose=goal_pose)
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

    # ----------------------------
    # Transform Data
    # ----------------------------
    def transform_datatype(self, inst_obj, actions, goal_pose_cos_sin, current_image_PIL, goal_image_PIL,
                           prompt_builder,
                           action_tokenizer, base_tokenizer, image_transform,
                           predict_stop_token=True):
        IGNORE_INDEX = -100
        current_action = actions[0]
        future_actions = actions[1:]
        # future_actions_string = ''.join(action_tokenizer(future_actions))
        # current_action_string = action_tokenizer(current_action)
        # action_chunk_string = current_action_string + future_actions_string
        # action_chunk_len = len(action_chunk_string)

        current_action_string = ''.join(action_tokenizer(current_action))
        future_actions_string = ''.join(''.join(action_tokenizer(a)) for a in future_actions)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        if inst_obj == "xxxx":
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {inst_obj}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]

        prompt_builder_inst = prompt_builder("openvla")
        for turn in conversation:
            prompt_builder_inst.add_turn(turn["from"], turn["value"])

        input_ids = torch.tensor(
            base_tokenizer(prompt_builder_inst.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX

        pixel_values_current = image_transform(current_image_PIL)
        pixel_values_goal = image_transform(goal_image_PIL)

        return dict(
            pixel_values_current=pixel_values_current,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids, labels=labels,
            dataset_name="lelan",
            actions=torch.as_tensor(actions),
            goal_pose=goal_pose_cos_sin,
            img_PIL=current_image_PIL, inst=inst_obj,
        )

    # ----------------------------
    # Data Transformer
    # ----------------------------
    def data_transformer_omnivla(self, current_image_PIL, lan_inst, goal_image_PIL,
                                 goal_pose_loc_norm, prompt_builder, action_tokenizer, processor, actions=None):
        if actions is None:
            actions = np.random.rand(8, 4)
        batch_data = self.transform_datatype(
            lan_inst, actions, goal_pose_loc_norm,
            current_image_PIL, goal_image_PIL,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
        )
        batch = self.collator_custom(
            instances=[batch_data],
            model_max_length=processor.tokenizer.model_max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        return batch

    # ----------------------------
    # Forward Pass (唯一定义，不重复)
    # ----------------------------
    def run_forward_pass(self, vla, action_head, noisy_action_projector, pose_projector,
                         batch, action_tokenizer, device_id, use_l1_regression, use_diffusion,
                         use_film, num_patches, compute_diffusion_l1=False,
                         num_diffusion_steps_train=None, mode="vali", idrun=0):
        noisy_actions, diffusion_timestep_embeddings = None, None

        s, l, p, i = self.satellite, self.lan_prompt, self.pose_goal, self.image_goal

        if s and not l and not p and not i:
            mid = 0
        elif s and not l and p and not i:
            mid = 1
        elif s and not l and not p and i:
            mid = 2
        elif s and not l and p and i:
            mid = 3
        elif not s and not l and p and not i:
            mid = 4
        elif not s and not l and p and i:
            mid = 5
        elif not s and not l and not p and i:
            mid = 6
        elif not s and l and not p and not i:
            mid = 7
        elif not s and l and p and not i:
            mid = 8
        else:
            mid = 7

        modality_id = torch.as_tensor([mid], dtype=torch.float32)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                modality_id=modality_id.to(torch.bfloat16).to(device_id),
                labels=batch["labels"].to(device_id),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id),
                proprio_projector=pose_projector,
                noisy_actions=noisy_actions if use_diffusion else None,
                noisy_action_projector=noisy_action_projector if use_diffusion else None,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
                use_film=use_film,
            )

        ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )

        with torch.no_grad():
            predicted_actions = action_head.predict_action(
                actions_hidden_states,
                modality_id.to(torch.bfloat16).to(device_id)
            )

        return predicted_actions, modality_id


# ===============================================================
# Model Configuration & Loading
# ===============================================================
class InferenceConfig:
    def __init__(self, vla_path, resume_step):
        self.resume = True
        self.vla_path = vla_path
        self.resume_step = resume_step
        self.use_l1_regression = True
        self.use_diffusion = False
        self.use_film = False
        self.num_images_in_input = 2
        self.use_lora = True
        self.lora_rank = 32
        self.lora_dropout = 0.0


def define_model(cfg):
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Loading OpenVLA Model `{cfg.vla_path}`")

    device_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    print(f"Detected constants:\n"
          f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
          f"\tACTION_DIM: {ACTION_DIM}\n"
          f"\tPOSE_DIM: {POSE_DIM}\n"
          f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device_id)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.to(dtype=torch.bfloat16, device=device_id)

    pose_projector = init_module(
        ProprioProjector, "pose_projector", cfg, device_id,
        {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM})

    action_head = None
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead_idcat, "action_head", cfg, device_id,
            {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM}, to_bf16=True)

    num_patches = (vla.vision_backbone.get_num_patches()
                   * vla.vision_backbone.get_num_images_in_input() + 1)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    return vla, action_head, pose_projector, device_id, num_patches, action_tokenizer, processor


# ===============================================================
# Main
# ===============================================================
if __name__ == "__main__":
    args = parse_args()

    modality_flags = parse_modality(args.modality)
    print(f"[Modality] {args.modality} → {modality_flags}")

    cfg = InferenceConfig(vla_path=args.vla_path, resume_step=args.resume_step)
    vla, action_head, pose_projector, device_id, num_patches, action_tokenizer, processor = (
        define_model(cfg)
    )

    inference = Inference(
        args=args,
        vla_model=vla,
        action_head_model=action_head,
        pose_projector_model=pose_projector,
        device_id=device_id,
        num_patches=num_patches,
        action_tokenizer=action_tokenizer,
        processor=processor,
        modality_flags=modality_flags,
    )

    arrived = inference.run()

    if arrived:
        print("\n✓ Navigation completed successfully!")
    else:
        print("\n✗ Navigation stopped (user/timeout/error).")
