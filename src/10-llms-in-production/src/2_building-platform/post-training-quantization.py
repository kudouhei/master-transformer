import copy
import torch
import torch.ao.quantization as q
from torchvision import models

# 1. 加载预训练模型
model_fp32 = models.resnet18(pretrained=True)
model_fp32.eval()

# 2. 深拷贝模型（量化会就地修改）
model_to_quantize = copy.deepcopy(model_fp32)

# 3. 配置量化后端（qnnpack for ARM, fbgemm for x86）
model_to_quantize.qconfig = q.get_default_qconfig("fbgemm")

# 4. 准备量化模型
prepared_model = q.prepare(model_to_quantize)

# 5. 校准（使用代表性数据）
calibration_data = [torch.rand(1, 3, 224, 224) for _ in range(100)]  # 模拟输入
with torch.inference_mode():
    for x in calibration_data:
        prepared_model(x)

# 6. 转换为量化模型 quantized_resnet18.pth 是 ResNet-18 的 INT8 量化版本
model_quantized = q.convert(prepared_model)

# 7. 保存量化模型
torch.save(model_quantized.state_dict(), "./quantized_resnet18.pth")