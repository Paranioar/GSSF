# # ----------------------- visualize the distribution of 3D ---------------------------------
# import numpy as np
#
# smooth_x, smooth_y, smooth_xy = 1.5, 2, -1
# x = np.arange(-1000, 1001) * 0.001
# y1 = np.sqrt(1 - x ** 2)
# y2 = - np.sqrt(1 - x ** 2)
#
# x_label = x[None, :].repeat(2, axis=0).reshape(-1)
# y_label = np.concatenate((y1, y2), axis=0)
# z_label = smooth_x * x_label ** 2 + smooth_y * y_label ** 2
# m_label = smooth_x * x_label ** 2 + smooth_y * y_label ** 2 + smooth_xy * x_label * y_label
# n_label = np.ones_like(x_label)
#
# from matplotlib import pyplot as plt
#
# ax = plt.subplot(111, projection='3d')
# ax.scatter(x_label, y_label, z_label, c='y', s=5)
# # ax.scatter(x_label, y_label, m_label, c='r', s=5)
# ax.scatter(x_label, y_label, n_label, c='g', s=5)
#
# ax.set_zlabel('Z')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# plt.show()

# ----------------------- visualize the matrix weights ---------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import palettable

# --------- save model weight ---------- #
# model_weight = model.module.sim_enc.sim_tran_w.data.cpu().numpy()
# np.save('./f30k_scan_vector_t2i.npy', model_weight)
#
# Or
# 
# emb_dim, sim_dim = 1024, 256
# model_weight = np.zeros((emb_dim, emb_dim))
# for i, param in enumerate(model.module.sim_enc.sim_tran_M):
#     model_weight[i*sim_dim:(i+1)*sim_dim, i*sim_dim:(i+1)*sim_dim] = \
#     param.weight.data.cpu().numpy()
# np.save('./f30k_scan_matrix_t2i.npy', model_weight)

# --------- visualize weights ----------- #
import torch

path = './visualization/weights'
# scan_vector_t2i = np.load('{}/f30k_scan_vector_t2i.npy'.format(path))
# scan_vector_t2i = scan_vector_t2i[None, :]
# scan_vector_i2t = np.load('{}/f30k_scan_vector_i2t.npy'.format(path))
# scan_vector_i2t = scan_vector_i2t[None, :]
# scan_matrix_t2i = np.load('{}/f30k_scan_matrix_t2i.npy'.format(path))
# scan_matrix_i2t = np.load('{}/f30k_scan_matrix_i2t.npy'.format(path))

# vsrn_vector = np.load('{}/f30k_vsrn_vector.npy'.format(path))
# vsrn_vector = vsrn_vector[None, :]
# vsrn_vector_abs = np.load('{}/f30k_vsrn_vector_abs.npy'.format(path))
# vsrn_vector_abs = vsrn_vector_abs[None, :]
# vsrn_matrix = np.load('{}/f30k_vsrn_matrix.npy'.format(path))
# vsrn_matrix_abs = np.load('{}/f30k_vsrn_matrix_abs.npy'.format(path))

vse_matrix_abs = np.load('{}/f30k_vse_matrix_abs.npy'.format(path))

bfan_matrix_equal = np.load('{}/f30k_bfan_matrix_equal.npy'.format(path))
bfan_matrix_prob = np.load('{}/f30k_bfan_matrix_prob.npy'.format(path))

weights_m = [vse_matrix_abs, bfan_matrix_equal, bfan_matrix_prob]
weights = weights_m

color = 'RdBu_r' # 'rainbow' 'YlGnBu' 'RdBu_r'

max_v, min_v = -1e8, 1e8
for weight in weights_m:
    max_v, min_v = max(max_v, weight.max()), min(min_v, weight.min())
print(max_v, min_v)

for weight in weights:
    weight = weight[::16, ::16]
    fig, ax = plt.subplots(figsize=(32, 32))
    ax = sns.heatmap(weight, vmax=0.5, vmin=-0.5, mask=(weight > -1e-2) & (weight < 1e-2), center=0.0,
                     annot=False, xticklabels=False, yticklabels=False, square=True, cmap=color)

    diag = torch.diag(torch.Tensor(weight).diag()).numpy()
    ax_p = ax.twinx()
    ax_p = sns.heatmap(diag, vmax=2.38, vmin=0.7, mask=(diag > -1e-2) & (diag < 1e-2), center=1.0,
                       annot=False, xticklabels=False, yticklabels=False, square=True, cmap='hot_r')
    plt.show()
