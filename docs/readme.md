## Bi-directional DB-converter 
### It uses two NN architectures trained in tandem

To run the app, you need a weights and bias account; it will ask for the API.

Currently, the `requirement.txt` is ignored; you can reactivate it from `main.py`.

Please install the correct version of pytorch depending on the availability of CUDA devices/ change the `requirement.txt` accordingly.

You can use existing folders for input data and output model, or you can use a different location, change the path in `config/data_path.yaml`

Most hyperparameters that are of interest to be tuned for most tasks can be found in `config/congif.yaml` and `config/architecture.yaml`

Other hyperparameters are in `config/advanced_config.yaml`

The user can define their own mapper (forward DB- converter) and inverter (backward DB- converter) by replacing the model.py or changing 2nd line of `db-converter.py` file

