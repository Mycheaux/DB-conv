To run this you need an weights and bias account, it will ask for the API.
Currentluy the requirement.txz is ignore, you can reactivate from `main.py`.
Please install correctversion of pytorch depending on availability of cuda devices/ change the requirement.txt accordingly.
You can use existing folders for input data and output model, or you can use any different location, change the path in config/data_path.yaml
Most hyperparapeters that are of interest to be tunes for most tasks can be found in config/congif.yaml and config/architecture.yaml
Other hyperparameters are in config/advanced_config.yaml
user can define their own mapper (forward DB- converter) and inverter (backward DB- converter) by replacing the model.py or changing 2nd line of `db-converter.py` file

