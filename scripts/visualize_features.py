import torch
from sklearn.manifold import TSNE
import time
from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    data = torch.load('knn_data_1.pth')
    time_start = time.time()
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    print(data['x'].shape, len(data['y']))
    tsne_results = tsne.fit_transform(data['x'].squeeze())
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    # df_subset['tsne-2d-one'] = tsne_results[:, 0]
    # df_subset['tsne-2d-two'] = tsne_results[:, 1]
    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1],
               tsne_results[:, 2], c='skyblue', s=60)
    plt.show()
    # sns.scatterplot(
    #     x=tsne_results[:, 0], y=tsne_results[:, 1],
    #     hue=data['y'],
    #     palette=sns.color_palette("hls", len(set(data['y']))),
    #     # data=df_subset,
    #     legend="full",
    #     # alpha=0.3
    # )
    # plt.show()
