#!/usr/bin/env python3
"""
完整流程：手机IPCam图片→ OmniVLA推理 → roslibpy → ROS1小车控制
"""

import sys, os

sys.path.insert(0, '')

import time
import math
import argparse
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import threading

# ── 用roslibpy，不用 rospy ──
import roslibpy

# ── OmniVLA 依赖 ──
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor, PrismaticProcessor
)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import (
    get_current_action_mask, get_next_actions_mask
)
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM

from transformers import (
    AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor
)

# ===============================================================
# 0. 全局急停
# ===============================================================
emergency_stop = False


def emergency_stop_listener():
    global emergency_stop
    input("\n⚠️  按回车键紧急停车...\n")
    emergency_stop = True
    print("\n🛑 紧急停车已触发！")


# ===============================================================
# 1. 从手机抓图
# ===============================================================
def grab_image_from_phone(url: str) -> Image.Image:
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


# ===============================================================
# 2. 加载模型
# ===============================================================
def remove_ddp(state_dict):
    return {
        k[7:] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def load_ckpt(name, path, step, device="cpu"):
    if (not os.path.exists(os.path.join(path, f"{name}--{step}_checkpoint.pt"))
            and name == "pose_projector"):
        name = "proprio_projector"
    ckpt_path = os.path.join(path, f"{name}--{step}_checkpoint.pt")
    print(f"  Loading: {ckpt_path}")
    return remove_ddp(torch.load(ckpt_path, map_location=device))


def load_all_models(model_path, resume_step):
    print("=" * 50)
    print("加载模型中...")
    print("=" * 50)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(
        OpenVLAConfig, OpenVLAForActionPrediction_MMNv1
    )

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    vla.vision_backbone.set_num_images_in_input(2)
    vla.to(dtype=torch.bfloat16, device=device)

    pose_proj = ProprioProjector(
        llm_dim=vla.llm_dim, proprio_dim=POSE_DIM
    )
    pose_proj.load_state_dict(
        load_ckpt("pose_projector", model_path, resume_step)
    )
    pose_proj = pose_proj.to(device)

    action_head = L1RegressionActionHead_idcat(
        input_dim=vla.llm_dim,
        hidden_dim=vla.llm_dim,
        action_dim=ACTION_DIM,
    )
    action_head.load_state_dict(
        load_ckpt("action_head", model_path, resume_step)
    )
    action_head = action_head.to(torch.bfloat16).to(device)

    num_patches = vla.vision_backbone.get_num_patches() * 2 + 1
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    print("✅ 模型加载完成\n")
    return (
        vla, action_head, pose_proj, device,
        num_patches, action_tokenizer, processor,
    )


# ===============================================================
# 3. 数据预处理
# ===============================================================
def prepare_batch(current_img, goal_img, instruction,
                  goal_pose, action_tokenizer, processor,
                  ):
    IGNORE_INDEX = -100
    tokenizer = processor.tokenizer
    img_transform = processor.image_processor.apply_transform

    actions = np.random.rand(8, 4)

    action_str = ''.join(action_tokenizer(actions[0])) + ''.join(
    ''.join(action_tokenizer(a)) for a in actions[1:])
    action_len = len(action_str)

    if instruction == "xxxx":
        prompt_text = "No language instruction"
    else:
        prompt_text = f"What action should the robot take to {instruction}?"

    conversation = [
        {"from": "human", "value": prompt_text},
        {"from": "gpt", "value": action_str},
    ]
    pb = PurePromptBuilder("openvla")
    for turn in conversation:
        pb.add_turn(turn["from"], turn["value"])

    input_ids = torch.tensor(
        tokenizer(pb.get_prompt(), add_special_tokens=True).input_ids
    )
    labels = input_ids.clone()
    labels[:-(action_len + 1)] = IGNORE_INDEX

    pv_cur = img_transform(current_img)
    pv_goal = img_transform(goal_img)

    input_ids = input_ids.unsqueeze(0)
    labels = labels.unsqueeze(0)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    pixel_values = torch.cat(
    (pv_cur.unsqueeze(0), pv_goal.unsqueeze(0)), dim=1
    )

    return {
    "input_ids": input_ids,
    "labels": labels,
    "attention_mask": attention_mask,
    "pixel_values": pixel_values,
    "actions": torch.from_numpy(actions).unsqueeze(0),
    "goal_pose": torch.from_numpy(
        np.array(goal_pose, dtype=np.float32)
    ).unsqueeze(0),
}


# ===============================================================
# 4. 推理+ 后处理
# ===============================================================
def clip_angle(a):
    while a > math.pi:   a -= 2 * math.pi
    while a < -math.pi:  a += 2 * math.pi
    return a


def run_inference(
        batch, vla, action_head, pose_proj,
        device, num_patches, modality_id_val,
):
    modality_id = torch.tensor([modality_id_val], dtype=torch.float32)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla.eval()(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
            modality_id=modality_id.to(torch.bfloat16).to(device),
            labels=batch["labels"].to(device),
            output_hidden_states=True,
            proprio=batch["goal_pose"].to(torch.bfloat16).to(device),
            proprio_projector=pose_proj.eval(),
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=False,
        )

    gt_ids = batch["labels"][:, 1:].to(device)
    cur_mask = get_current_action_mask(gt_ids)
    nxt_mask = get_next_actions_mask(gt_ids)
    last_hs = output.hidden_states[-1]
    text_hs = last_hs[:, num_patches:-1]
    act_hs = (
        text_hs[cur_mask | nxt_mask]
        .reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )

    with torch.no_grad():
        pred = action_head.eval().predict_action(
            act_hs, modality_id.to(torch.bfloat16).to(device)
        )

    return pred.float().cpu().numpy()[0]


def waypoints_to_velocity(waypoints, select=4, spacing=0.1):
    w = waypoints[select].copy()
    w[:2] *= spacing
    dx, dy, hx, hy = w
    EPS, DT = 1e-8, 1 / 3

    if abs(dx) < EPS and abs(dy) < EPS:
        v = 0
        omega = clip_angle(np.arctan2(hy, hx)) / DT
    elif abs(dx) < EPS:
        v = 0
        omega = np.sign(dy) * np.pi / (2 * DT)
    else:
        v = dx / DT
        omega = np.arctan(dy / dx) / DT

    v = np.clip(v, 0, 0.5)
    omega = np.clip(omega, -1.0, 1.0)

    maxv, maxw = 0.3, 0.3
    if abs(v) > maxv or abs(omega) > maxw:
        if abs(omega) < 0.001:
            v, omega = maxv * np.sign(v), 0.0
        else:
            rd = v / omega
            if abs(v) <= maxv and abs(omega) > maxw:
                v = maxw * np.sign(v) * abs(rd)
                omega = maxw * np.sign(omega)
            elif abs(rd) >= maxv / maxw:
                v = maxv * np.sign(v)
                omega = maxv * np.sign(omega) / abs(rd)
            else:
                v = maxw * np.sign(v) * abs(rd)
                omega = maxw * np.sign(omega)

    return float(v), float(omega)


# ===============================================================
# 5. roslibpy 控制器（替代 rospy）
# ===============================================================
class RobotController:
    """通过 roslibpy (WebSocket) 控制小车"""

    def __init__(
            self,
            car_ip: str,
            ros_port: int = 9090,
            cmd_vel_topic: str = "/cmd_vel",
            max_linear: float = 0.3,
            max_angular: float = 0.3,
    ):
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.prev_linear = 0.0
        self.prev_angular = 0.0

        # ── 连接 rosbridge ──
        print(f"正在连接小车 rosbridge ws://{car_ip}:{ros_port} ...")
        self.client = roslibpy.Ros(host=car_ip, port=ros_port)
        self.client.run()

        if not self.client.is_connected:
            raise ConnectionError(
                f"无法连接 rosbridge ws://{car_ip}:{ros_port}"
            )
        print(f"✅ rosbridge 连接成功!")

        # ── 创建 publisher ──
        self.cmd_vel = roslibpy.Topic(
            self.client, cmd_vel_topic, 'geometry_msgs/Twist'
        )
        print(f"✅ Publisher 就绪 → {cmd_vel_topic}")

        # 发一个零速确保通道畅通
        self.stop()

    def _make_twist_msg(self, linear: float, angular: float):
        """构造 Twist 消息"""
        return roslibpy.Message({
            'linear': {'x': linear, 'y': 0.0, 'z': 0.0},
            'angular': {'x': 0.0, 'y': 0.0, 'z': angular},
        })

    def clamp(self, val, lo, hi):
        return max(lo, min(val, hi))

    def smooth(self, prev, target, alpha=0.4):
        """指数平滑，避免突变"""
        return prev + alpha * (target - prev)

    def send_velocity(self, linear, angular, use_smooth=True):
        """发送速度指令"""
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
        return linear, angular

    def stop(self):
        """停车（连发5次确保收到）"""
        zero_msg = self._make_twist_msg(0.0, 0.0)
        for _ in range(5):
            self.cmd_vel.publish(zero_msg)
            time.sleep(0.05)
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        print("🛑 小车已停止")

    def disconnect(self):
        """断开连接"""
        self.stop()
        try:
            self.cmd_vel.unadvertise()
            self.client.terminate()
        except Exception:
            pass
        print("🔌 rosbridge 已断开")


# ===============================================================
# 6. 主函数
# ===============================================================
def main():
    global emergency_stop

    parser = argparse.ArgumentParser(
        description="手机IPCam → OmniVLA → roslibpy → ROS1小车"
    )
    # ── 图像/ 模型参数 ──
    parser.add_argument("--phone_url", type=str,
                        default="http://10.99.255.235:8080/shot.jpg")
    parser.add_argument("--goal_image", type=str,
                        default="./inference/goal_img.jpg")
    parser.add_argument("--instruction", type=str,
                        default="move toward blue trash bin")
    parser.add_argument("--model_path", type=str,
                        default="./omnivla-original")
    parser.add_argument("--resume_step", type=int, default=285000)
    parser.add_argument("--modality", type=str, default="image_goal",
                        choices=["image_goal", "language", "pose_goal"])
    parser.add_argument("--save_dir", type=str, default="./test_output")

    # ── 小车 / 控制参数 ──
    parser.add_argument("--car_ip", type=str, default="192.168.110.143",
                        help="小车IP地址")
    parser.add_argument("--ros_port", type=int, default=9090,
                        help="rosbridge websocket端口")
    parser.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel")
    parser.add_argument("--max_linear", type=float, default=0.10,
                        help="最大线速度 m/s（初始建议0.05-0.1）")
    parser.add_argument("--max_angular", type=float, default=0.3,
                        help="最大角速度 rad/s")
    parser.add_argument("--control_hz", type=float, default=15.0,
                        help="控制频率 Hz")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="最大运行步数")
    parser.add_argument("--dry_run", action="store_true",
                        help="干跑模式：只打印不发送指令")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 模态 ──
    modality_map = {"image_goal": 6, "language": 7, "pose_goal": 4}
    modality_id = modality_map[args.modality]
    use_lan = (args.modality == "language")
    instruction = args.instruction if use_lan else "xxxx"

    # ── 急停线程 ──
    threading.Thread(target=emergency_stop_listener, daemon=True).start()

    # ── 加载OmniVLA ──
    (vla, action_head, pose_proj, device,
     num_patches, action_tokenizer, processor) = \
        load_all_models(args.model_path, args.resume_step)

    # ── 目标图片 ──
    goal_img = Image.open(args.goal_image).convert("RGB")
    print(f"目标图片: {args.goal_image}尺寸: {goal_img.size}")
    goal_pose = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # ── 测试手机抓图 ──
    print(f"\n测试抓图: {args.phone_url}")
    try:
        test_img = grab_image_from_phone(args.phone_url)
        test_img.save(os.path.join(args.save_dir, "test_grab.jpg"))
        print(f"✅ 抓图成功!尺寸: {test_img.size}")
    except Exception as e:
        print(f"❌ 抓图失败: {e}")
        return

    # ── 初始化小车控制器 ──
    robot = None
    if not args.dry_run:
        try:
            robot = RobotController(
                car_ip=args.car_ip,
                ros_port=args.ros_port,
                cmd_vel_topic=args.cmd_vel_topic,
                max_linear=args.max_linear,
                max_angular=args.max_angular,
            )
        except Exception as e:
            print(f"❌ 连接小车失败: {e}")
            print("   请确认:")
            print(f"   1. 小车IP {args.car_ip} 可达")
            print(f"   2. rosbridge 已启动 (端口 {args.ros_port})")
            print("   3. 或先用 --dry_run 测试")
            return
    else:
        print("⚙️  干跑模式：只打印，不控制小车")

    # ── 控制循环 ──
    interval = 1.0 / args.control_hz
    step = 0

    print("\n" + "=" * 60)
    print(f"  开始控制")
    print(f"  模态: {args.modality}")
    print(f"  频率: {args.control_hz} Hz")
    print(f"  速度限制: linear={args.max_linear} m/s  "
          f"angular={args.max_angular} rad/s")
    print(f"  小车: {args.car_ip}:{args.ros_port}")
    print(f"  模式: {'🟡 干跑' if args.dry_run else '🟢 实际控制'}")
    print("=" * 60)

    try:
        while step < args.max_steps:
            # 急停检查
            if emergency_stop:
                if robot:
                    robot.stop()
                print("🛑 急停退出")
                break

            # rosbridge 连接检查
            if robot and not robot.client.is_connected:
                print("❌ rosbridge 断开，停车退出")
                break

            t0 = time.time()

            # 1) 抓图
            try:
                current_img = grab_image_from_phone(args.phone_url)
            except Exception as e:
                print(f"[Step {step}] 抓图失败: {e}，停车等待...")
                if robot:
                    robot.stop()
                time.sleep(1)
                continue

            # 2) 准备数据
            batch = prepare_batch(
                current_img, goal_img, instruction,
                goal_pose, action_tokenizer, processor,
            )

            # 3) 推理
            t_infer = time.time()
            waypoints = run_inference(
                batch, vla, action_head, pose_proj,
                device, num_patches, modality_id,
            )
            infer_time = time.time() - t_infer

            # 4) 转速度
            linear, angular = waypoints_to_velocity(waypoints)

            # 5) 发送或打印
            if robot and not args.dry_run:
                actual_lin, actual_ang = robot.send_velocity(
                    linear, angular
                )
                status = "🟢 SENT"
            else:
                actual_lin, actual_ang = linear, angular
                status = "🟡 DRY "

            # 6) 日志
            total_time = time.time() - t0
            print(
                f"[Step {step:3d}] {status} "
                f"v={actual_lin:+.4f}w={actual_ang:+.4f}  "
                f"(raw: {linear:+.4f}/{angular:+.4f}) "
                f"| infer {infer_time:.2f}s  total {total_time:.2f}s"
            )

            # 7) 保存帧
            if step % 10 == 0:
                current_img.save(
                    os.path.join(args.save_dir, f"step_{step:04d}.jpg")
                )

            step += 1

            # 控制频率
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except KeyboardInterrupt:
        print("\n⌨️  Ctrl+C 中断")

    finally:
        print("\n正在安全停车...")
        if robot:
            robot.disconnect()
        print(f"✅ 共运行 {step} 步，已安全退出")


if __name__ == "__main__":
    main()
