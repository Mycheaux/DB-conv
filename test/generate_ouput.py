from src.read_config import read_config
from src.model import Mapper, Inverter
from src.utils  import create_directories
from test.test_data_loader import load_test_data
from test.load_model import load_mapper,load_inverter


import numpy as np

def run_test():
    arch_config = read_config('config/architecture.yaml')
    Mapper_input_size = arch_config.get('Mapper_input_size')
    Mapper_output_size = arch_config.get('Mapper_output_size')
    Mapper_learning_rate = arch_config.get('Mapper_learning_rate', 1e-3)
    Inverter_input_size = arch_config.get('Inverter_input_size')
    Inverter_output_size = arch_config.get('Inverter_input_size')
    Inverter_learning_rate = arch_config.get('Inverter_input_size', 1e-3)


    data_path_config = read_config('config/data_path.yaml')
    general_config = read_config('config/config.yaml')
    
    load_model_path = general_config.get('load_model_path', 'output/models')
    load_project_name = general_config.get('load_project_name','app-test')
    load_model_name = general_config.get('load_model_name','m0t0')
    load_epoch_name = general_config.get('load_epoch_name','last.ckpt')
    batch_size = general_config.get('batch_size', 24)
    #Input data path
    data_path =  data_path_config.get('data_path',"data/preprocessed")
    x_test_path = data_path + '/'+ data_path_config.get('x_test','za_test.npy')
    y_test_path = data_path + '/'+ data_path_config.get('y_test','zb_test.npy')
    #Output data path
    config = read_config('config/config.yaml')
    output_data_path = config.get('output_data_path')

    x_test_tt, y_test_tt = load_test_data(x_test_path, y_test_path, batch_size)

    checkpoint_path= str( load_model_path + "/"+ load_project_name + "/" + load_model_name  + "/" + load_epoch_name)
    print("Loading from >>>", checkpoint_path)

    m  = load_mapper(checkpoint_path,Mapper, Mapper_input_size, Mapper_output_size, Mapper_learning_rate)
    i = load_inverter(checkpoint_path, Inverter, Inverter_input_size, Inverter_output_size, Inverter_learning_rate)
    
    #Make directories
    create_directories(output_data_path, load_project_name, load_model_name, load_epoch_name)

    # Mapper outputs
    y_hat_test = m(x_test_tt).detach().cpu().numpy()
    y_hat_hat_test = i(m(x_test_tt)).detach().cpu().numpy()

    np.save(str(output_data_path + "/" + load_project_name + "/" + load_model_name+ "/" + load_epoch_name+ "/" + "DB-A_converted.npy"), y_hat_test)
    np.save(str(output_data_path + "/" + load_project_name + "/" + load_model_name+ "/" + load_epoch_name+ "/" + "DB-A_reconverted.npy"), y_hat_hat_test)
    
    np.savetxt(str(output_data_path + "/" + load_project_name + "/" + load_model_name+ "/" + load_epoch_name+ "/" + "DB-A_converted.csv"), y_hat_test, delimiter=',')
    np.savetxt(str(output_data_path + "/" + load_project_name + "/" + load_model_name+ "/" + load_epoch_name+ "/" + "DB-A_reconverted.csv"), y_hat_hat_test, delimiter=',')
    
    # Inverter outputs
    x_hat_test = i(y_test_tt).detach().cpu().numpy()
    x_hat_hat_test = m(i(y_test_tt)).detach().cpu().numpy()

    np.save(str(output_data_path + "/" + load_project_name + "/" + load_model_name+ "/" + load_epoch_name+ "/" + "DB-B_converted.npy"), x_hat_test)
    np.save(str(output_data_path + "/" + load_project_name + "/" + load_model_name+ "/" + load_epoch_name+ "/" + "DB-B_reconverted.npy"), x_hat_hat_test)
    
    np.savetxt(str(output_data_path + "/" + load_project_name + "/" + load_model_name+ "/" + load_epoch_name+ "/" + "DB-B_converted.csv"), x_hat_test, delimiter=',')
    np.savetxt(str(output_data_path + "/" + load_project_name + "/" + load_model_name+ "/" + load_epoch_name+ "/" + "DB-B_reconverted.csv"), x_hat_hat_test, delimiter=',') # add , fmt='%.6f' to save space 
    
    print ('Files are saved to output/<porject_name>/<model_name>/<epoch_name>')
    print ("saved in both npy and csv format")