import numpy as np
import torch
import smplx
# from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3)"""
    theta = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis_angle / theta
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    cos, sin = torch.cos(theta).squeeze(-1), torch.sin(theta).squeeze(-1)
    C = 1 - cos
    R = torch.stack([
        cos + x*x*C, x*y*C - z*sin, x*z*C + y*sin,
        y*x*C + z*sin, cos + y*y*C, y*z*C - x*sin,
        z*x*C - y*sin, z*y*C + x*sin, cos + z*z*C,
    ], dim=-1).reshape(*axis_angle.shape[:-1], 3, 3)
    return R

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6)，取旋转矩阵前两列"""
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)

def convert_npz_to_201(
    npz_path: str,
    smplh_model_path: str,
    output_path: str = None,
):

    data = np.load(npz_path, allow_pickle=True)
    
    # ---- 读取字段 ----
    # Rh: (L, 3) 根关节轴角; trans: (L, 3) 平移
    # poses: (L, 156) 前66维=22关节轴角(含根), 后续为手部
    # betas: (16,) 或 (1,16); gender: (1,) 或标量
    poses_full = torch.tensor(data["poses"], dtype=torch.float32)  # (L, 156)
    trans = torch.tensor(data["trans"], dtype=torch.float32)       # (L, 3)
    
    # betas = torch.tensor(data["betas"], dtype=torch.float32).reshape(1, -1)  # (1, 16)
    betas = torch.tensor(data["betas"], dtype=torch.float32).reshape(1, -1)[:, :10]

    gender = str(data["gender"].item()) if hasattr(data["gender"], "item") else str(data["gender"][0])
    L = poses_full.shape[0]
    
    # 取前 66 维作为 22 个身体关节的轴角（SMPL-H 前 21 个身体关节 + 根关节）
    poses_aa = poses_full[:, :66]  # (L, 66) = 22 joints × 3
    
    # ---- 加载 SMPL-H 模型做 FK ----
    body_model = smplx.create(
        model_path=smplh_model_path,
        model_type="smplh",
        gender=gender,
        use_pca=False,
        batch_size=L,
    )
    
    output = body_model(
        # betas=betas.expand(L, -1),
        global_orient=poses_aa[:, :3],   # 根关节轴角
        body_pose=poses_aa[:, 3:66],     # 21 个身体关节轴角
        transl=trans,
    )
    # joints: (L, 52, 3)，取前 22 个身体关节
    joints_global = output.joints[:, :22, :]  # (L, 22, 3)
    
    # ---- 计算 201 维各部分 ----
    # 1) 根平移 (3)
    root_trans = trans  # (L, 3)
    
    # 2) 全局朝向 6D (6)
    root_rot_mat = axis_angle_to_matrix(poses_aa[:, :3])    # (L, 3, 3)
    global_orient_6d = matrix_to_rotation_6d(root_rot_mat)  # (L, 6)
    
    # 3) 21 个关节局部旋转 6D (126)
    local_aa = poses_aa[:, 3:66].reshape(L, 21, 3)          # (L, 21, 3)
    local_rot_mat = axis_angle_to_matrix(local_aa)          # (L, 21, 3, 3)
    local_rot_6d = matrix_to_rotation_6d(local_rot_mat)     # (L, 21, 6)
    local_rot_flat = local_rot_6d.reshape(L, -1)            # (L, 126)
    
    # 4) 22 个关节局部位置 (66)：转到相对于根节点的局部坐标系
    root_pos = joints_global[:, 0:1, :]                     # (L, 1, 3)
    relative_pos = joints_global - root_pos                 # (L, 22, 3)
    root_rot_inv = root_rot_mat.transpose(-1, -2)           # (L, 3, 3)
    # (L, 22, 3) × (L, 3, 3) -> (L, 22, 3)
    local_pos = torch.einsum("ljd,lde->lje", relative_pos, root_rot_inv)
    local_pos_flat = local_pos.reshape(L, -1)               # (L, 66)
    
    # ---- 拼接 ----
    motion = torch.cat([
        root_trans,          # 3
        global_orient_6d,    # 6
        local_rot_flat,      # 126
        local_pos_flat,      # 66
    ], dim=-1).detach().numpy().astype(np.float32)  # (L, 201)
    
    length = L
    
    if output_path is not None:
        np.savez_compressed(
            output_path,
            motion=motion,
            length=np.int32(length),
        )
        print(f"saved {output_path}, motion shape={motion.shape}")
    
    return motion, length


if __name__ == "__main__":
    convert_npz_to_201(
        npz_path="/inspire/hdd3/project/agent-driven-world-model/public/models/hunyuan/output/local_infer/00000000_000.npz",
        smplh_model_path="/inspire/hdd3/project/agent-driven-world-model/public/GENMO/inputs/",
        output_path="./data/motions/000001.npz",
    )