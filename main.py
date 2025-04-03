# from src.utils import install_requirements
# install_requirements()



import os
from src.train import train_model  # Assume train.py has a train_model() function
from src.test import run_tests 
 


import wandb# Assume test.py has a run_tests() function

def get_wandb_key():
    """Securely retrieves W&B API key from user input"""
    print("Weights & Biases API key is required for logging.")
    print("Get your key from: https://wandb.ai/authorize")
    return input("Enter your W&B API key: ").strip()

if __name__ == '__main__':
    # install_requirements() 
    # Set up W&B authentication
    wandb_key = get_wandb_key()
    os.environ['WANDB_API_KEY'] = wandb_key
    
    try:
        wandb.login(key=wandb_key)
        print("Successfully logged into Weights & Biases!")
    except Exception as e:
        print(f"Failed to log into W&B. Error: {e}")
        exit(1)  # Exit if login fails
    
    
    # Execute training workflow
    print("\n=== Starting Training ===")
    train_model()  # Calls trainer.fit() from your training script
    
    # Execute testing workflow
    print("\n=== Starting Testing ===")
    run_tests()    # Runs tests and saves output files``