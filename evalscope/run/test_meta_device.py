import os

import torch
import transformers
import sentence_transformers


def main() -> None:
    print("torch", torch.__version__)
    print("transformers", transformers.__version__)
    print("sentence-transformers", sentence_transformers.__version__)
    if hasattr(torch, "get_default_device"):
        print("default_device", torch.get_default_device())
    else:
        print("default_device", "N/A")

    for key in [
        "TORCH_DEFAULT_DEVICE",
        "TRANSFORMERS_LOW_CPU_MEM_USAGE",
        "TRANSFORMERS_DEVICE_MAP",
        "ACCELERATE_USE_CPU",
        "ACCELERATE_USE_MPS_DEVICE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ]:
        print(key, os.environ.get(key))

    model_path = (
        "/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/"
        "evalscope/dataprocess/model/models--sentence-transformers--all-MiniLM-L6-v2/"
        "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    )
    model = sentence_transformers.SentenceTransformer(
        model_path,
        device="cpu",
        local_files_only=True,
        model_kwargs={"low_cpu_mem_usage": False, "device_map": None},
    )
    param = next(model.parameters())
    print("param device:", param.device, "is_meta:", param.is_meta)


if __name__ == "__main__":
    main()
