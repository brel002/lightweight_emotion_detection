# backbone_sizer.py
import json, math
from pathlib import Path
import torch
import timm

def describe_backbone(name: str):
    # Create as a feature extractor (no logits head), no download
    model = timm.create_model(name, pretrained=False, num_classes=0, global_pool="avg")
    model.eval()
    # Count bytes directly from state_dict (params + buffers)
    sd = model.state_dict()
    numel = sum(t.numel() for t in sd.values())
    bytes_fp32 = sum(t.numel() * 4 for t in sd.values())  # float32 -> 4 bytes
    # Probe embedding dim with dummy forward using model.default_cfg
    in_h, in_w = (model.default_cfg or {}).get("input_size", (3, 224, 224))[-2:]
    with torch.no_grad():
        y = model(torch.zeros(1, 3, in_h, in_w))
    emb_dim = int(y.shape[-1]) if y.ndim == 2 else int(y.flatten(1).shape[-1])

    return {
        "backbone": name,
        "params_M": numel / 1e6,
        "size_mb_fp32": bytes_fp32 / (1024**2),
        "size_mb_fp16_est": bytes_fp32 / 2 / (1024**2),     # rough half
        "size_mb_int8_weight_only_est": bytes_fp32 / 4 / (1024**2),  # rough quarter
        "emb_dim": emb_dim,
        "input_hw": (in_h, in_w),
    }

def add_head_size(row: dict, head_size_mb_fp32: float | None):
    if head_size_mb_fp32 is None:
        return row
    row = row.copy()
    row["total_size_mb_fp32"] = row["size_mb_fp32"] + head_size_mb_fp32
    row["total_size_mb_fp16_est"] = row["size_mb_fp16_est"] + head_size_mb_fp32 / 2
    row["total_size_mb_int8_weight_only_est"] = row["size_mb_int8_weight_only_est"] + head_size_mb_fp32 / 4
    return row



if __name__ == "__main__":
    backbones = [
        "mobilenetv4_conv_small_050.e3000_r224_in1k",
        "mobilenetv3_small_100",
        "mobilenetv3_large_100",
        "efficientnet_b0",
        "mobilenetv2_100",
        "mobilevit_s.cvnets_in1k",
        "deit_tiny_patch16_224",
        "efficientnet_b1",
        "efficientnetv2_s"       
    ]

    # If you log your inference head size in MB (FP32), map it here per run/backbone
    # (leave None to skip)
    head_mb = {
        # "mobilenetv4_conv_small_050.e3000_r224_in1k": 2.96,  # example from your logs
        "mobilenetv4_conv_small_050.e3000_r224_in1k": 0.0,
        "mobilenetv3_small_100": 0.0,
        "mobilenetv3_large_100": 0.0,
        "efficientnet_b0": 0.0,
        "mobilenetv2_100": 0.0,
        "mobilevit_s.cvnets_in1k": 0.0,
        "deit_tiny_patch16_224": 0.0,
        "efficientnet_b1": 0.0,
        "efficientnetv2_s": 0.0
    }

    rows = []
    for name in backbones:
        row = describe_backbone(name)
        row = add_head_size(row, head_mb.get(name))
        rows.append(row)

    out = Path("backbone_sizes.json")
    out.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
