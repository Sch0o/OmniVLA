#!/usr/bin/env python3
"""
最简测试：手机IPCam图片 → OmniVLA推理 → 打印 linear/angular
"""

import sys, os
sys.path.insert(0, '..')

import time
import math
import argparse
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import utm

# ── OmniVLA 依赖 ──
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM

from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor
from torch.nn.utils.rnn import pad_sequence


# ===============================================================
# 1. 从手机抓图（就这一个函数）
# ===============================================================
def grab_image_from_phone(url: str) -> Image.Image:
    """从手机 IP Webcam 抓一张图"""
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return img


# ===============================================================
# 2. 加载模型（和原代码一样）
# ===============================================================
def remove_ddp(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def load_ckpt(name, path, step, device="cpu"):
    if not os.path.exists(os.path.join(path, f"{name}--{step}_checkpoint.pt")) and name == "pose_projector":
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
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    vla.vision_backbone.set_num_images_in_input(2)
    vla.to(dtype=torch.bfloat16, device=device)

    # pose projector
    pose_proj = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=POSE_DIM)
    pose_proj.load_state_dict(load_ckpt("pose_projector", model_path, resume_step))
    pose_proj = pose_proj.to(device)

    # action head
    action_head = L1RegressionActionHead_idcat(
        input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM
    )
    action_head.load_state_dict(load_ckpt("action_head", model_path, resume_step))
    action_head = action_head.to(torch.bfloat16).to(device)

    num_patches = vla.vision_backbone.get_num_patches() * 2+ 1
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    print("✅ 模型加载完成\n")
    return vla, action_head, pose_proj, device, num_patches, action_tokenizer, processor


# ===============================================================
# 3. 数据预处理（精简版）
# ===============================================================
def prepare_batch(current_img, goal_img, instruction, goal_pose, action_tokenizer, processor):
    IGNORE_INDEX = -100
    tokenizer = processor.tokenizer
    img_transform = processor.image_processor.apply_transform

    # dummy actions
    actions = np.random.rand(8, 4)
    action_str = ''.join(action_tokenizer(actions[0])) + ''.join(''.join(action_tokenizer(a)) for a in actions[1:])
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

    input_ids = torch.tensor(tokenizer(pb.get_prompt(), add_special_tokens=True).input_ids)
    labels = input_ids.clone()
    labels[:-(action_len + 1)] = IGNORE_INDEX

    pv_cur = img_transform(current_img)
    pv_goal = img_transform(goal_img)

    # collate (batch_size=1)
    input_ids = input_ids.unsqueeze(0)
    labels = labels.unsqueeze(0)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    pixel_values = torch.cat((pv_cur.unsqueeze(0), pv_goal.unsqueeze(0)), dim=1)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "actions": torch.from_numpy(actions).unsqueeze(0),
        "goal_pose": torch.from_numpy(np.array(goal_pose, dtype=np.float32)).unsqueeze(0),
    }


# ===============================================================
# 4. 推理 + 后处理
# ===============================================================
def clip_angle(a):
    while a > math.pi:  a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def run_inference(batch, vla, action_head, pose_proj, device, num_patches, modality_id_val):
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
            noisy_actions=None, noisy_action_projector=None,
            diffusion_timestep_embeddings=None, use_film=False,
        )

    gt_ids = batch["labels"][:, 1:].to(device)
    cur_mask = get_current_action_mask(gt_ids)
    nxt_mask = get_next_actions_mask(gt_ids)
    last_hs = output.hidden_states[-1]
    text_hs = last_hs[:, num_patches:-1]
    act_hs = text_hs[cur_mask | nxt_mask].reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)

    with torch.no_grad():
        pred = action_head.eval().predict_action(act_hs, modality_id.to(torch.bfloat16).to(device))

    return pred.float().cpu().numpy()[0]

def waypoints_to_velocity(waypoints, select=4, spacing=0.1):
    w = waypoints[select].copy()
    w[:2] *= spacing
    dx, dy, hx, hy = w
    EPS, DT = 1e-8, 1/3

    if abs(dx) < EPS and abs(dy) < EPS:
        v = 0; omega = clip_angle(np.arctan2(hy, hx)) / DT
    elif abs(dx) < EPS:
        v = 0; omega = np.sign(dy) * np.pi / (2 * DT)
    else:
        v = dx / DT; omega = np.arctan(dy / dx) / DT

    v = np.clip(v, 0, 0.5)
    omega = np.clip(omega, -1.0, 1.0)

    #限幅
    maxv, maxw = 0.3, 0.3
    if abs(v) > maxv or abs(omega) > maxw:
        if abs(omega) < 0.001:
            v, omega = maxv * np.sign(v), 0.0
        else:
            rd = v / omega
            if abs(v) <= maxv and abs(omega) > maxw:
                v, omega = maxw * np.sign(v) * abs(rd), maxw * np.sign(omega)
            elif abs(rd) >= maxv / maxw:
                v, omega = maxv * np.sign(v), maxv * np.sign(omega) / abs(rd)
            else:
                v, omega = maxw * np.sign(v) * abs(rd), maxw * np.sign(omega)

    return float(v), float(omega)


# ===============================================================
# 5. 主函数
# ===============================================================
def main():
    parser = argparse.ArgumentParser(description="手机发图 → OmniVLA → 输出结果")
    parser.add_argument("--phone_url", type=str,
                        help="手机图片URL, 例如 http://192.168.1.120:8080/shot.jpg",default="http://10.99.255.235:8080/shot.jpg")
    parser.add_argument("--goal_image", type=str, default="./inference/goal_img.jpg")
    parser.add_argument("--instruction", type=str, default="move toward blue trash bin")
    parser.add_argument("--model_path", type=str, default="./omnivla-original")
    parser.add_argument("--resume_step", type=int, default=285000)
    parser.add_argument("--modality", type=str, default="image_goal",
                        choices=["image_goal", "language", "pose_goal"])
    parser.add_argument("--loop", action="store_true", help="持续循环推理")
    parser.add_argument("--interval", type=float, default=1.0, help="循环间隔(秒)")
    parser.add_argument("--save_dir", type=str, default="./test_output")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 模态映射 ──
    modality_map = {"image_goal": 6, "language": 7, "pose_goal": 4}
    modality_id = modality_map[args.modality]
    use_lan = (args.modality == "language")

    # ── 加载模型 ──
    vla, action_head, pose_proj, device, num_patches, action_tokenizer, processor = \
        load_all_models(args.model_path, args.resume_step)

    # ── 加载目标图片 ──
    goal_img = Image.open(args.goal_image).convert("RGB")
    print(f"目标图片: {args.goal_image}尺寸: {goal_img.size}")

    # ── 默认 goal_pose（image_goal 模式下不太重要但需要传入） ──
    goal_pose = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # ── 先测试抓一张图 ──
    print(f"\n测试从手机抓图: {args.phone_url}")
    try:
        test_img = grab_image_from_phone(args.phone_url)
        test_img.save(os.path.join(args.save_dir, "test_grab.jpg"))
        print(f"✅ 抓图成功!尺寸: {test_img.size}")
    except Exception as e:
        print(f"❌ 抓图失败: {e}")
        print("请检查:")
        print("  1. 手机和服务器在同一局域网")
        print("  2. 手机 IP Webcam App 已启动")
        print("  3. URL 是否正确")
        return

    # ── 开始推理 ──
    instruction = args.instruction if use_lan else "xxxx"
    step = 0

    print("\n" + "=" * 50)
    print("开始推理")
    print("=" * 50)

    while True:
        t0 = time.time()

        # 1)抓图
        try:
            current_img = grab_image_from_phone(args.phone_url)
        except Exception as e:
            print(f"[Step {step}] 抓图失败: {e}")
            time.sleep(1)
            continue

        # 2) 准备数据
        batch = prepare_batch(current_img, goal_img, instruction, goal_pose, action_tokenizer, processor)

        # 3) 推理
        t_infer = time.time()
        waypoints = run_inference(batch, vla, action_head, pose_proj, device, num_patches, modality_id)
        infer_time = time.time() - t_infer

        # 4) 转换为速度
        linear, angular = waypoints_to_velocity(waypoints)

        # 5) 输出结果
        total_time = time.time() - t0
        print(
            f"[Step {step:3d}] "
            f"linear {linear:.4f}  angular {angular:.4f}  "
            f"| infer {infer_time:.2f}s  total {total_time:.2f}s"
        )

        # 6) 保存当前帧 + 结果（可选）
        current_img.save(os.path.join(args.save_dir, f"step_{step:04d}.jpg"))

        step += 1

        if not args.loop:
            print("\n单次推理完成。加--loop 可持续运行。")
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()