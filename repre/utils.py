import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA

def generate_data(batch_size, trajectory_length, file_path):

    # Time steps
    t = np.arange(trajectory_length)

    # Generate data for each batch
    data = np.zeros((batch_size, trajectory_length, 4))
    for i in range(batch_size):
        # Cosine and Sine functions
        w = 0.09
        a = np.random.choice(np.arange(-0.5, 0.5, 0.1))
        b = np.random.choice(np.arange(-0.5, 0.5, 0.1))
        s = np.random.choice(np.arange(-0.5, 0.5, 0.1))
        a = 1
        b = 0

        cos_value = b + a*np.cos(2 * np.pi * w * t + s)
        sin_value = b + a*np.sin(2 * np.pi * w * t + s)

        # Linearly increasing values with random slopes
        angle = np.random.choice(np.arange(0, 2*np.pi, 1/3*np.pi))
        linear_value1 = np.cos(angle) * t
        linear_value2 = np.sin(angle) * t

        # Adding noise
        # noise = np.random.normal(0, noise_level, (trajectory_length, 6))

        # Stacking and storing the values
        data[i, :, :] = np.stack([cos_value, sin_value, linear_value1, linear_value2], axis=1)
        # data[i, :, :] = np.stack([cos_value, sin_value, linear_value1, linear_value2, np.full(trajectory_length, w), np.full(trajectory_length, angle*180/np.pi)], axis=1)

    # # Save the data as a .npy file (Optional)
    # np.save(file_path, data)
    
    # # Visualize
    # visualize_traj(data, trajectory_length)
        
    return data


def visualize_traj(data, trajectory_length):

    loaded_data = data

    # Select first episode
    selected_episode = 0

    # Extract values for each dimension
    cos_values = loaded_data[selected_episode, :, 0]
    sin_values = loaded_data[selected_episode, :, 1]
    linear_values1 = loaded_data[selected_episode, :, 2]
    linear_values2 = loaded_data[selected_episode, :, 3]
    t = np.arange(trajectory_length)

    # Plot each dimension
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(t, cos_values)
    plt.title('Cosine Function')

    plt.subplot(2, 2, 2)
    plt.plot(t, sin_values)
    plt.title('Sine Function')

    plt.subplot(2, 2, 3)
    plt.plot(t, linear_values1)
    plt.title('Linear Values 1')

    plt.subplot(2, 2, 4)
    plt.plot(t, linear_values2)
    plt.title('Linear Values 2')

    plt.tight_layout()
    plt.show()
    plt.pause(0.1) 
    plt.close()


def get_minibatch(data, num_samples, args):

    L = args.period

    batch_size, trajectory_length, feature_dim = data.shape
    
    # randomly choose batch and start idx
    batch_indices = np.random.randint(0, batch_size, size=num_samples)
    start_time_indices = np.random.randint(0, trajectory_length - (L+1), size=num_samples)
    
    # select data point
    minibatch_before = np.zeros((num_samples, feature_dim))
    minibatch_before_prime = np.zeros((num_samples, feature_dim))
    minibatch_after = np.zeros((num_samples, feature_dim))
    minibatch_after_prime = np.zeros((num_samples, feature_dim))
    
    for i in range(num_samples):
        batch_index = batch_indices[i]
        start_index = start_time_indices[i]
        minibatch_before[i, :] = data[batch_index, start_index, :]
        minibatch_before_prime = data[batch_index, start_index + 1, :]
        minibatch_after[i, :]  = data[batch_index, start_index + L, :]
        minibatch_after_prime  = data[batch_index, start_index + (L+1), :]
    
    return minibatch_before, minibatch_before_prime, minibatch_after, minibatch_after_prime


def plot_graph(data, latent_dim, writer, step):

    plt.figure(figsize=(12, 8))  

    if latent_dim == 2: # directly plot
        plt.plot(data[:, 0], data[:, 1], 'o')  

    else: # PCA for higher dimension
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data) 
        for i in range(len(pca_data)):
            plt.scatter(pca_data[i, 0], pca_data[i, 1], color='red', alpha=0.5)

    # labeling
    plt.title('2D NumPy Data Plot')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')

    # add figure
    writer.add_figure('numpy_plot', plt.gcf(), step)

    # plot
    # plt.show()
