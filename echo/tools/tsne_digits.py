import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import os
import numpy as np
from einops import rearrange

def get_data(video_num = 50):
    data_list = []
    npy_path = 'work_dirs/feature1/'
    max_num = len(os.listdir(npy_path))
    video_num = max_num if max_num < video_num else video_num
    for filename in os.listdir(npy_path)[:video_num]:
        if filename.endswith('.npy'):
            file_path = os.path.join(npy_path, filename)
            data = np.load(file_path)
            data_list.append(data)

    # list to ndarray
    data = np.array(data_list)
    data = rearrange(data, 'n t c h w -> (n t) (c h w)')
    label = [i // 10 for i in range(data.shape[0])]

    return data, label


# 主函数，执行t-SNE降维
def main():
    data, label = get_data(200)    # 调用函数，获取数据集信息
    print('Starting compute t-SNE Embedding...')
    ts = TSNE(n_components=2, init='pca', random_state=0)
    # t-SNE降维
    result = ts.fit_transform(data)

    # 调用函数，绘制图像
    # fig = plot_embedding(result, label, 't-SNE Embedding of feature')
    plt.scatter(result[:, 0], result[:, 1], 5, label)
    # 显示图像
    plt.savefig(fname='tsne.png')
    # plt.show()


# 主函数
if __name__ == '__main__':
    main()
