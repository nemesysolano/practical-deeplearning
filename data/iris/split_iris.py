import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from multiprocessing import Pool, cpu_count
import functools
SIGNS = (-1, 1)

def generate_iris_feature(std, avg, sample):
    dev = std*0.10 * -1 if np.random.randint(-10, 10) < 0 else 1
    return np.round(avg + dev,2)

def generate_iris_sample(kernel_record):
    all_std = kernel_record['all_std']
    all_avg = kernel_record['all_avg']
    species = kernel_record['Species']
    return {
        'SepalLengthCm': generate_iris_feature(all_std[species]['SepalLengthCm'], all_avg[species]['SepalLengthCm'], kernel_record['SepalLengthCm']), 
        'SepalWidthCm': generate_iris_feature(all_std[species]['SepalWidthCm'], all_avg[species]['SepalWidthCm'], kernel_record['SepalWidthCm']), 
        'PetalLengthCm': generate_iris_feature(all_std[species]['PetalLengthCm'], all_avg[species]['PetalLengthCm'], kernel_record['PetalLengthCm']), 
        'PetalWidthCm': generate_iris_feature(all_std[species]['PetalWidthCm'], all_avg[species]['PetalWidthCm'], kernel_record['PetalWidthCm']), 
        'Species': species
    }

def create_kernel_record(record, all_std, all_avg):
    record['all_std'] = all_std
    record['all_avg'] = all_avg
    return record

def save_dataset(dataset, x_file_name, y_file_name):
    label_columns = 'Species'
    input_columns = list(filter(lambda column_name: not column_name.startswith('Species'), dataset.columns))
    x = dataset[input_columns]
    y = dataset[label_columns]
    np.save(x_file_name, x.to_numpy(), allow_pickle=True)
    np.save(y_file_name, y.to_numpy(), allow_pickle=True)

if __name__ == "__main__":
    # Loads and shuffles dataset
    iris = pd.read_csv("iris.csv")
    
    # Assuming iris is your DataFrame and it has been loaded properly, calculate std and avg for the input features
    # grouped by Species
    numeric_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    all_std = iris.groupby('Species')[numeric_columns].std().to_dict('index')
    all_avg = iris.groupby('Species')[numeric_columns].mean().to_dict('index')

    # Generate augmented dataset
    kernel_records=  [create_kernel_record(record, all_std, all_avg) for record in iris.to_dict('records')]
    pool = Pool(cpu_count())
    augmented_iris = shuffle(
        pd.concat(
            [
                iris, 
                *[shuffle(pd.DataFrame(pool.map(generate_iris_sample, kernel_records))) for i in range(1,10)]
            ], 
            axis = 0, 
            ignore_index= True
        )
    )
    

    # Split dataset
    train_size = int(len(augmented_iris) * 0.90)
    test_size = len(augmented_iris) - train_size
    train_iris = augmented_iris.iloc[:train_size]
    test_iris = augmented_iris.iloc[train_size:]

    save_dataset(train_iris, 'iris_train_features_augmented.npy', 'iris_train_labels_augmented.npy')
    save_dataset(test_iris, 'iris_test_features_augmented.npy', 'iris_test_labels_augmented.npy')
    save_dataset(iris,'iris_features.npy', 'iris_labels.npy')


    