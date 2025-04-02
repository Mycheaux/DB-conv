import numpy as np
import scipy as sp




def havrda_charvat_entropy(predictions, labels, parameter_a):
    """
    Calculate the Havrda-Charvat entropy between predicted values and labels.

    Args:
        predictions (np.ndarray or torch.Tensor): Predicted values (shape: [batch_size, num_classes]).
        labels (np.ndarray or torch.Tensor): Ground truth labels (shape: [batch_size, num_classes]).
        parameter_a (float, optional): Parameter for the Havrda-Charvat entropy. Default is 2.0.

    Returns:
        float: Average Havrda-Charvat entropy across the batch.
    """
    assert predictions.shape == labels.shape, "Predictions and labels must have the same shape."

    # Normalize predictions and labels to probabilities
    predictions_prob = np.clip(predictions.detach().cpu().numpy(), 1e-10, 1.0 - 1e-10)
    labels_prob = np.clip(labels.detach().cpu().numpy(), 1e-10, 1.0 - 1e-10)

    # Calculate entropy term for each class
    entropy_term = -np.sum(predictions_prob * np.log(labels_prob), axis=1)

    # Compute the overall Havrda-Charvat entropy
    hc_entropy = np.mean((1.0 - np.exp(-parameter_a * entropy_term)) / parameter_a)

    return hc_entropy

def calc_dist(arr_a,arr_b, method:str):

    if method == "L1":
        distance = np.sum(np.absolute(arr_a - arr_b))
    if method == "L2":
        distance = (np.sum(np.square(arr_a-arr_b)))**0.5
    return distance

def k_twin(matrix_a:np.array,matrix_b:np.array, k:int, method:str):


    n_a = matrix_a.shape[0]
    n_b = matrix_b.shape[0]

    if n_a != n_b:
        print ("matrix dimention doesn't match")
        return

    dist_matrix_a = np.zeros((n_a, n_a))
    dist_matrix_b = np.zeros((n_b, n_b))

    for i in range (0,n_a):
        for j in range (0,n_a):
            dist_matrix_a[i,j] = calc_dist(matrix_a[i],matrix_a[j], method)
            np.fill_diagonal(dist_matrix_a, np.inf)
    # print (dist_matrix_a)

    for i in range (0,n_a):
        for j in range (0,n_a):
            dist_matrix_b[i,j] = calc_dist(matrix_b[i],matrix_b[j], method)
            np.fill_diagonal(dist_matrix_b, np.inf)
    # print (dist_matrix_b)

    k_closest_a = np.zeros((n_a,k))
    for i in range (0,n_a):
        k_closest_a[i] = np.argsort(dist_matrix_a[i])[:k]
    # print(k_closest_a)

    k_closest_b = np.zeros((n_b,k))
    for i in range (0,n_b):
        k_closest_b[i] = np.argsort(dist_matrix_b[i])[:k]
    # print(k_closest_b)

    total_count = 0
    for i in range (0, n_a ):
        # print(set(k_closest_a[i]))
        count = len(set(k_closest_a[i]).intersection(set(k_closest_b[i])))
        total_count += count
        # print(total_count)

    JI = (total_count/(n_a*k))

    return JI

def k_twin_ind(matrix_a:np.array,matrix_b:np.array, k:int, method:str):


    n_a = matrix_a.shape[0]
    n_b = matrix_b.shape[0]

    if n_a != n_b:
        print ("matrix dimention doesn't match")
        return

    dist_matrix_a = np.zeros((n_a, n_a))
    dist_matrix_b = np.zeros((n_b, n_b))

    for i in range (0,n_a):
        for j in range (0,n_a):
            dist_matrix_a[i,j] = calc_dist(matrix_a[i],matrix_a[j], method)
            np.fill_diagonal(dist_matrix_a, np.inf)
    # print (dist_matrix_a)

    for i in range (0,n_a):
        for j in range (0,n_a):
            dist_matrix_b[i,j] = calc_dist(matrix_b[i],matrix_b[j], method)
            np.fill_diagonal(dist_matrix_b, np.inf)
    # print (dist_matrix_b)

    k_closest_a = np.zeros((n_a,k))
    for i in range (0,n_a):
        k_closest_a[i] = np.argsort(dist_matrix_a[i])[:k]
    # print(k_closest_a)

    k_closest_b = np.zeros((n_b,k))
    for i in range (0,n_b):
        k_closest_b[i] = np.argsort(dist_matrix_b[i])[:k]
    # print(k_closest_b)

    total_count = 0
    JI_list = []
    for i in range (0, n_a ):
        # print(set(k_closest_a[i]))
        count = len(set(k_closest_a[i]).intersection(set(k_closest_b[i])))
        JI_ind = count/(k)
        JI_list.append(JI_ind)
    return JI_list