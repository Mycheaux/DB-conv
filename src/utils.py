import numpy as np
import os
import subprocess
import pkg_resources
import re
from pathlib import Path



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

    if k<= n_a:
        k_closest_a = np.zeros((n_a,k))
        for i in range (0,n_a):
            k_closest_a[i] = np.argsort(dist_matrix_a[i])[:k]
    else:
        k_closest_a = np.zeros((n_a,n_a))
        for i in range (0,n_a):
            k_closest_a[i] = np.argsort(dist_matrix_a[i])[:n_a]
        
    if k<= n_b:
        k_closest_b = np.zeros((n_b,k))
        for i in range (0,n_b):
            k_closest_b[i] = np.argsort(dist_matrix_b[i])[:k]
    else:
        k_closest_b = np.zeros((n_b,n_b))
        for i in range (0,n_b):
            k_closest_b[i] = np.argsort(dist_matrix_b[i])[:n_b]

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

    if k<= n_a:
        k_closest_a = np.zeros((n_a,k))
        for i in range (0,n_a):
            k_closest_a[i] = np.argsort(dist_matrix_a[i])[:k]
    else:
        k_closest_a = np.zeros((n_a,n_a))
        for i in range (0,n_a):
            k_closest_a[i] = np.argsort(dist_matrix_a[i])[:n_a]

    if k<= n_b:
        k_closest_b = np.zeros((n_b,k))
        for i in range (0,n_b):
            k_closest_b[i] = np.argsort(dist_matrix_b[i])[:k]
    else:
        k_closest_b = np.zeros((n_b,n_b))
        for i in range (0,n_b):
            k_closest_b[i] = np.argsort(dist_matrix_b[i])[:n_b]

    total_count = 0
    JI_list = []
    for i in range (0, n_a ):
        # print(set(k_closest_a[i]))
        count = len(set(k_closest_a[i]).intersection(set(k_closest_b[i])))
        JI_ind = count/(k)
        JI_list.append(JI_ind)
    return JI_list



def get_gpu_info():
    """
    Retrieves GPU information using the `nvidia-smi` command.
    Returns:
        str: GPU details as a string.
    """
    gpu_info = os.popen("nvidia-smi").read().strip()
    return gpu_info


def install_requirements():
    """
    Installs packages from requirements.txt only if they are not already installed,
    handling --index-url for specific packages like torch.
    """
    try:
        print("Checking and installing dependencies from requirements.txt...")
        
        # Read the requirements.txt file
        with open("requirements.txt", "r") as file:
            requirements = file.readlines()
        
        # Check each package in requirements.txt
        for requirement in requirements:
            package = requirement.strip()
            
            # Extract package name and version (if any)
            match = re.match(r"([a-zA-Z0-9\-]+(?:==[a-zA-Z0-9.]+)?)(.*)", package)
            if match:
                package_name = match.group(1)
                extra_args = match.group(2).strip()
                
                try:
                    # Check if the package is already installed
                    pkg_resources.require(package_name)
                    print(f"Package '{package_name}' is already installed.")
                except pkg_resources.DistributionNotFound:
                    # Install the package if not found
                    print(f"Installing missing package: {package}")
                    install_command = [os.sys.executable, "-m", "pip3", "install", package]
                    
                    # Append extra arguments if any (e.g., --index-url)
                    if extra_args:
                        install_command.extend(extra_args.split())
                    
                    subprocess.check_call(install_command)
                except pkg_resources.VersionConflict as e:
                    print(f"Version conflict for package '{package_name}': {e}. Attempting to resolve...")
                    install_command = [os.sys.executable, "-m", "pip", "install", "--upgrade", package]
                    
                    # Append extra arguments if any (e.g., --index-url)
                    if extra_args:
                        install_command.extend(extra_args.split())
                    
                    subprocess.check_call(install_command)
            else:
                print(f"Skipping invalid requirement: {package}")

        print("Dependency check and installation complete!")
    except Exception as e:
        print(f"Error during dependency installation: {e}")
        exit(1)
        
        


def create_directories(output_data_path, load_project_name, load_model_name, load_epoch_name):
    # Define the directory paths
    dir1 = Path(output_data_path) / load_project_name
    dir2 = dir1 / load_model_name
    dir3 = dir2 / load_epoch_name

    # Create the directories recursively
    dir1.mkdir(parents=True, exist_ok=True)
    dir2.mkdir(parents=True, exist_ok=True)
    dir3.mkdir(parents=True, exist_ok=True)

    print(f"Directories created:\n{dir1}\n{dir2}\n{dir3}")
