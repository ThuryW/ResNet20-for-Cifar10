import torch
import sys
import numpy as np
import pandas as pd

torch.set_printoptions(threshold=sys.maxsize) # 显示所有的元素，不会有省略号出现
np.set_printoptions(threshold=np.inf)
checkpoint_path = '/home/wangtianyu/my_resnet20/autorun/autorun_ckpt/20250724_152650/best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Check if the state_dict is nested under a 'net' key or is the dict itself
if 'net' in checkpoint:
    state_dict = checkpoint['net']
else:
    state_dict = checkpoint # Assume the checkpoint is just the state_dict

for name, param in state_dict.items():
    param_np = param.numpy()
    
    if len(param_np.shape) > 2:
        # 将高维数组展平成二维数组
        param_np = param_np.reshape(param_np.shape[0], -1)
    elif param_np.shape == ():
        # 将标量转换为二维数组
        param_np = param_np.reshape(1, 1)
    
    # 保存为 CSV 文件
    pd.DataFrame(param_np).to_csv(f'/home/wangtianyu/my_resnet20/weights/prune/test1/{name}.csv', header=False, index=False)
