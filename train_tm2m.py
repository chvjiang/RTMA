# train_tm2m.py
import os
import random
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

from hymotion.utils.loaders import load_object
from hymotion.pipeline.motion_diffusion import MotionFlowMatching, length_to_mask

from data.tm2m_dataset import TextToMotionBaseDataset

# ==================== 1. 时移参考动作构造 ====================
def build_ref_motion(x, x_mask, shift):
    """
    x: (L, D), x_mask: (L,) bool
    ref    = x[0 : shift+1]      # 长度 shift+1
    target = x[shift+1 : L]      # 长度 L - shift - 1
    """
    L = x.shape[0]
    ref_end = shift
    assert ref_end < L, f"need shift+1 < L, got shift={shift}, L={L}"

    ref = x[:ref_end]                 # (shift+1, D)
    ref_mask = x_mask[:ref_end]
    # target = x[ref_end:]              # (L-shift-1, D)
    # target_mask = x_mask[ref_end:]
    return ref, ref_mask#, target, target_mask


class TM2MWrapper(Dataset):
    """包装已有 T2M dataset，在 __getitem__ 中构造时移参考。"""
    def __init__(self, base_dataset, ref_shift_max=60, ref_shift_prob=0.8):
        self.base = base_dataset
        self.ref_shift_max = ref_shift_max
        self.ref_shift_prob = ref_shift_prob

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        x = sample["motion"]          # (L, 201)
        x_mask = sample["motion_mask"]  # (L,) bool
        # print(torch.isnan(x).any())
        if random.random() < self.ref_shift_prob and self.ref_shift_max > 0:
            # shift = random.randint(1, self.ref_shift_max)
            shift = self.ref_shift_max
        else:
            shift = 0

        ref, ref_mask = build_ref_motion(x, x_mask, shift)
        # ref, ref_mask, target, target_mask = build_ref_motion(x, x_mask, shift)
        # sample["motion"] = target
        # sample["motion_mask"] = target_mask
        sample["ref_motion"] = ref
        sample["ref_mask_temporal"] = ref_mask
        
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     print(f"NaN/Inf in motion for sample {idx}")
        # if torch.isnan(ref_motion).any():
        #     print(f"NaN in ref_motion for sample {idx}")
    
        return sample


# ==================== 2. 训练主流程 ====================
def train():
    # ---- 加载配置 ----
    with open("train_tm2m.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 构建 pipeline（内部含 motion_transformer）----
    pipeline = load_object(
        cfg["train_pipeline"],
        cfg["train_pipeline_args"],
        network_module=cfg["network_module"],
        network_module_args=cfg["network_module_args"],
    ).to(device)

    net = pipeline.motion_transformer
    ckpt = torch.load("./ckpts/tencent/HY-Motion-1.0/latest.ckpt", map_location="cpu")
    net.load_state_dict(ckpt, strict=False) 
    net.train()

    base_dataset = TextToMotionBaseDataset(
        motion_dir="data/motions",
        annotation_file="data/annotations.json",
        train_frames=360,
        min_frames=20,
        normalize=True,
        mean_std_dir="./stats",
        random_crop=True,
    )
    dataset = TM2MWrapper(
        base_dataset,
        ref_shift_max=cfg["tm2m"]["ref_shift_max"],
        ref_shift_prob=cfg["tm2m"]["ref_shift_prob"],
    )
    dataloader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                            shuffle=True, num_workers=4, pin_memory=True)

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scaler = GradScaler(enabled=cfg["training"]["mixed_precision"])

    global_step = 0
    os.makedirs(cfg["training"]["output_dir"], exist_ok=True)

    # ---- 训练循环 ----
    for epoch in range(cfg["training"]["epochs"]):
        for batch in dataloader:
            x0 = batch["motion"].to(device)              # (B, L, 201)
            x_mask = batch["motion_mask"].to(device)     # (B, L)
            ref_motion = batch["ref_motion"].to(device)  # (B, L, 201)
            ref_mask = batch["ref_mask_temporal"].to(device)  # (B, L)

            # 文本编码（需根据你的 dataset 调整：若已缓存特征则直接用）
            text_list = batch["text"]
            with torch.no_grad():
                hidden = pipeline.encode_text({"text": text_list})
            vtxt = hidden["text_vec_raw"].to(device)
            ctxt = hidden["text_ctxt_raw"].to(device)
            ctxt_len = hidden["text_ctxt_raw_length"].to(device)
            ctxt_mask = length_to_mask(ctxt_len, ctxt.shape[1]).to(device)

            # Flow Matching 加噪
            B, L, D = x0.shape
            t = torch.rand(B, device=device)
            noise = torch.randn_like(x0)
            t_exp = t.view(B, 1, 1)
            xt = (1 - t_exp) * x0 + t_exp * noise
            target_v = noise - x0

            with autocast(enabled=cfg["training"]["mixed_precision"]):
                pred_v = net(
                    x=xt,
                    ctxt_input=ctxt,
                    vtxt_input=vtxt,
                    timesteps=t,
                    x_mask_temporal=x_mask,
                    ctxt_mask_temporal=ctxt_mask,
                    ref_motion=ref_motion,
                    ref_mask_temporal=ref_mask,
                )
                # print(pred_v.shape, target_v.shape)
                # print("pose pred max abs", pred_v.abs().max())
                # print("pose target max abs", target_v.abs().max())
                # print("pred has nan?", torch.isnan(pred_v).any())
                # print("pred has inf?", torch.isinf(pred_v).any())

                loss = F.mse_loss(pred_v, target_v)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % cfg["training"]["log_every"] == 0:
                print(
                    f"epoch {epoch} | step {global_step} | "
                    f"loss {loss.item():.6f} | grad_norm {grad_norm.item():.4f}"
                )

            if global_step % cfg["training"]["save_every"] == 0:
                ckpt_path = os.path.join(
                    cfg["training"]["output_dir"], f"step_{global_step:07d}.pt"
                )
                torch.save({
                    "step": global_step,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, ckpt_path)
                print(f"saved {ckpt_path}")


if __name__ == "__main__":
    train()