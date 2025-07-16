import torch
import sys
import numpy as np
import pandas as pd

torch.set_printoptions(threshold=sys.maxsize) # 显示所有的元素，不会有省略号出现
np.set_printoptions(threshold=np.inf)
checkpoint_path = '/home/wangtianyu/my_resnet20/pruned_checkpoints/resnet20_pruned_finetuned_prune80_ft10.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['net']

for name, param in state_dict.items():
    param_np = param.numpy()
    
    if len(param_np.shape) > 2:
        # 将高维数组展平成二维数组
        param_np = param_np.reshape(param_np.shape[0], -1)
    elif param_np.shape == ():
        # 将标量转换为二维数组
        param_np = param_np.reshape(1, 1)
    
    # 保存为 CSV 文件
    pd.DataFrame(param_np).to_csv(f'/home/wangtianyu/my_resnet20/weights/prune/weights_20/{name}.csv', header=False, index=False)
