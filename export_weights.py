import torch
import sys
import numpy as np
import pandas as pd

torch.set_printoptions(threshold=sys.maxsize) # 显示所有的元素，不会有省略号出现
np.set_printoptions(threshold=np.inf)
checkpoint_path = '/home/wangtianyu/pytorch_resnet_cifar10/save_resnet20/checkpoint.th'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['state_dict']

for name, param in state_dict.items():
    param_np = param.numpy()
    
    if len(param_np.shape) > 2:
        # 将高维数组展平成二维数组
        param_np = param_np.reshape(param_np.shape[0], -1)
    elif param_np.shape == ():
        # 将标量转换为二维数组
        param_np = param_np.reshape(1, 1)
    
    # 保存为 CSV 文件
    pd.DataFrame(param_np).to_csv(f'./weights/{name}.csv', header=False, index=False)

# # 创建一个空的DataFrame来存储权重数据
# weights_data = pd.DataFrame(columns=['Layer Name', 'Weight Shape', 'Weights'])

# # 遍历state_dict，将权重值添加到DataFrame中
# for key, value in state_dict.items():
#     if isinstance(value, torch.Tensor):
#         weights_data = pd.concat([
#             weights_data,
#             pd.DataFrame({
#                 'Layer Name': [key],
#                 'Weight Shape': [tuple(value.shape)],
#                 'Weights': [value.flatten().numpy()]  # 将权重展平为一维数组并转换为NumPy数组
#             })
#         ], ignore_index=True)

# # 保存DataFrame为CSV文件
# csv_filename = 'weights_export.csv'
# weights_data.to_csv(csv_filename, index=False)

# print(f"Exported weights to {csv_filename}")