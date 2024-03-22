import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def onehot2radius(state_batch, radius_dim):

    if state_batch.ndim == 3: 
        radius_onehot_batch = state_batch[:, -1, -radius_dim:]
    else:
        radius_onehot_batch = state_batch[:, -radius_dim:]

    radius_candidate = np.array([20,40,80])

    radius_batch = np.dot(radius_onehot_batch, radius_candidate)

    return radius_batch

def get_evalbatch(data, num_samples):

    L = 20

    batch_size, trajectory_length, feature_dim = data.shape
    
    # 결과를 저장할 배열 초기화
    minibatches = []

    # 각 배치 인덱스에 대해 실행
    for batch_index in range(batch_size):
        # 현재 배치 인덱스에 대해 여러 start time 인덱스를 무작위로 선택
        start_indices = np.random.randint(0, trajectory_length, size=100)

        # 선택된 각 start time에 대해 데이터 포인트를 미니배치에 추가
        for start_index in start_indices:
            # L+1을 고려한 범위 체크는 여기서 구체적인 L 값에 따라 달라질 수 있음
            minibatch = data[batch_index, start_index, :]
            minibatches.append(minibatch)

    # (eval) randomly choose batch and start idx
    batch_index = np.random.randint(0, batch_size)
    start_index = np.random.randint(0, trajectory_length - (L+1))

    # (eval) select data point for eval
    minibatch_eval = data[batch_index, start_index:start_index + (L), :]

    minibatches = np.array(minibatches).reshape(-1, feature_dim)

    return minibatches, minibatch_eval

def plot_graph(data, eval_data, latent_dim, writer, step):

    plt.figure(figsize=(12, 8))  

    if latent_dim == 2: # directly plot
        for i in range(len(data)):
            plt.scatter(data[i, 0], data[i, 1], color='red', alpha=0.2)  
        for i in range(len(eval_data)):
            plt.text(eval_data[i, 0], eval_data[i, 1], str(i), color='blue', alpha=1.0, fontsize=12)  
            plt.scatter(eval_data[i, 0], eval_data[i, 1], color='blue', alpha=0.2)  

    else: # PCA for higher dimension
        pca = PCA(n_components=2)
        combined_data = np.concatenate([data, eval_data], axis=0)
        pca_combined_data = pca.fit_transform(combined_data)
        pca_data = pca_combined_data[:len(data)]
        pca_data_eval = pca_combined_data[len(data):]

        for i in range(len(pca_data)):
            plt.scatter(pca_data[i, 0], pca_data[i, 1], color='red', alpha=0.2)
        for i in range(len(pca_data_eval)):
            plt.text(pca_data_eval[i, 0], pca_data_eval[i, 1], str(i), color='blue', alpha=0.5, fontsize=12)  
            plt.scatter(pca_data_eval[i, 0], pca_data_eval[i, 1], color='blue', alpha=0.2)  

    # labeling
    plt.title('Latent space visualization')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')

    # add figure
    writer.add_figure('numpy_plot', plt.gcf(), step)

    # plot
    # plt.show()