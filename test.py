import torch

state = torch.load('E8500_state.pth',  map_location=torch.device('cpu'))
train_result_history = state['eval_result_history']
print(train_result_history)

# X = []
# Y = []
# Z = []
# for epoch, metrics in train_result_history.items():
#     X.append(epoch)
#     Y.append(metrics['psnr'])
#     Z.append(metrics['ssim'])

# import matplotlib.pyplot as plt
# # plt.plot(X, Y)
# plt.plot(X, Z)
# plt.xlabel('Epoch')
# plt.ylabel('PSNR')
# plt.show()
