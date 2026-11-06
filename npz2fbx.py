import numpy as np
from hymotion.utils.smplh2woodfbx import SMPLH2WoodFBX

npz_file="/inspire/hdd3/project/agent-driven-world-model/public/deepggen/qblim2/data_zoo/interaction_data/AMASS/raw/KIT/3/912_3_01_poses.npz"

# 2. 构造转换器，指定模板 FBX 路径
# 模板文件在 HY-Motion 仓库的 assets/wooden_models/ 下
converter = SMPLH2WoodFBX(
    template_fbx_path="assets/wooden_models/boy_Rigging_smplx_tex.fbx",
    scale=100,  # 米转厘米，默认值通常适用
)

# 3. 直接传入 npz 路径或数据字典进行转换
# 你的 npz 里已经有 poses 和 trans，convert_npz_to_fbx 可以直接处理
converter.convert_npz_to_fbx(
    npz_file=npz_file,
    outname="output.fbx",
    fps=120,
)

print("FBX 导出完成: output.fbx")