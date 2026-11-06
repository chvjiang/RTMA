**Quick Start**

```bash
git clone https://github.com/chvjiang/RTMA.git
cd RTMA
git lfs pull
pip install -r requirements.txt
```

Download weights per [`ckpts/README.md`](ckpts/README.md) (≥26GB VRAM), then run:

```bash
python3 local_infer.py --model_path ckpts/tencent/HY-Motion-1.0        # standard
```

Use English prompts under 60 words, focused on limb/torso motion. Not supported: animals, multi-person, scenes/camera, in-place loops.

Based on [HY-Motion 1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0); original parts under Tencent Hunyuan Community License, modifications under [your license].