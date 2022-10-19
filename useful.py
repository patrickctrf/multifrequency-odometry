import glob
import os
from math import sin, cos

import numpy
from matplotlib import pyplot as plt
from numpy import save, savez, concatenate, ones, where, array, load
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from ptk.timeseries import *
from ptk.utils import *


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_lr(optimizer, new_lr=0.01):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
        return


# Funcoes que nao estou mais usando mas que podem ser uteis em algum momento


# def fake_position(x):
#     """
# Generate a simulated one dimensional position for training on "1d dataset" an
# test if the neural network is able to generalize well.
#
# Given "x" values, calculate f(x) for this.
#
# It is supposed to be the output for the neural network.
#
#     :param x: Array of values to calculate f(x) = y.
#     :return: Array os respective position in "y" axis.
#     """
#
#     return cos(x) + 2 * sin(2 * x)
#
#
# def fake_acceleration(x):
#     """
# Generate a simulated one dimensional acceleration for training on "1d dataset" an test if the neural network is able to generalize well.
#
# Given "x" values, calculate f''(x) for this.
#
# It is supposed to be the input for the neural network.
#
#     :param x: Array of values to calculate f''(x) = y''.
#     :return: Array os respective acceleration in "y" axis.
#     """
#
#     return 4 * cos(2 * x) - sin(x)


def fake_position(x):
    """
Generate a simulated one dimensional position for training on "1d dataset" an
test if the neural network is able to generalize well.
Given "x" values, calculate f(x) for this.
It is supposed to be the output for the neural network.
    :param x: Array of values to calculate f(x) = y.
    :return: Array os respective position in "y" axis.
    """

    if x == 0:
        return 1

    return sin(x) / x


def fake_acceleration(x):
    """
Generate a simulated one dimensional acceleration for training on "1d dataset" an test if the neural network is able to generalize well.
Given "x" values, calculate f''(x) for this.
It is supposed to be the input for the neural network.
    :param x: Array of values to calculate f''(x) = y''.
    :return: Array os respective acceleration in "y" axis.
    """

    if x == 0:
        return -1 / 3

    return -((x ** 2 - 2) * sin(x) + 2 * x * cos(x)) / x ** 3


def plot_csv(csv_path="dataset-room2_512_16/mav0/mocap0/data.csv"):
    output_data = read_csv(csv_path)

    for key in output_data.columns[1:]:
        plt.close()
        output_data.plot(kind='scatter', x=output_data.columns[0], y=key, color='red')
        plt.savefig(key + ".png", dpi=200)
        plt.show()

    return


def plot_csv_xy(csv_path="dataset-room2_512_16/mav0/mocap0/data.csv", name=""):
    output_data = read_csv(csv_path)

    plt.close()
    output_data.plot(kind='line', x=output_data.columns[1],
                     y=output_data.columns[2], color='red', ylabel=output_data.columns[2],
                     title=name, legend=None)
    plt.savefig("x_vs_y_" + name + ".png", dpi=200)
    plt.show()

    return


def difference(dataset, interval=1):
    """
For a given dataset, calculates the difference between each sample and the
previous one, for both input and output values.

    :param dataset: Dataset to difference from.
    :param interval: Which sample to compare (usually 1).
    :return: New dataset composed of differences.
    """
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return array(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    """
If you are working with an output composed by differences of this time in
relation to the previous time, this function is useful beacause it takes the
array with the history and sums the given difference (yhat) to the last value in
history (or the value "-interval" in history).

It's more reasonable to use interval as 1, but I implemented it this way because
it gets more general.

    :param history: Array of historical values in the output series.
    :param yhat: A difference to be summed with the previous value.
    :param interval: Which past value is this difference (yhat) related to (usually, interval = 1).
    :return: An output representing the current value of the measuring being tracked, no more the difference of this value.
    """
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
    """
Simples rescale train and test data (separately) and returns scaled data. Note
that the scaling process is done for EACH FEATURE. Even if you have different
range for each feature, the scaling procces is done separately.
This scaler SAVES data_min, data_max and feature range for later unscaling, or
even rescaling newer data.
Equation details: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

Also returns the scaler object used to further inverse scaling.

    :param train: Train data to be rescaled (array or matrix).
    :param test: Test data to be rescaled (array or matrix).
    :return: Scaler, train scaled and test scaled.
    """
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
    """
Since you "scale" object saves data_min and data_max, as well as feature_range,
you just need to pass it to this function, togheter with the object you want to
inverse scale (yhat) too.

    :param scaler: Scaler object from scipy, previously calculated in the same dataset being worked here.
    :param X: Matrix line used as input to generate this output prediction (yhat).
    :param yhat: Predicted output for your network.
    :return: Only the "yhat" data rescaled.
    """
    # We have input features in X information. We need to add "y" info
    # (prediction, not ground truth or reference) before unscaling (because scaler was made with whole dataset, X and y).
    # We call it "row" because it's a array (there's no columns). We'll reshape
    # it into a matrix line ahead in this function.
    new_row = [x for x in X] + [yhat]
    array = numpy.array(new_row)
    # Array is vertical, but scaler expects a matrix or matrix line, and that's
    # why we reshape it.
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def alinha_dataset_tum(imu_data, ground_truth, impossible_value=-444444):
    """
Funcao especifica para abrir e alinhar IMU e ground truth no formato de dataset
da TUM. Mto chata mds

    :param imu_data: Array-like (2D) with samples from IMU.
    :param ground_truth: Array-like (2D) with samples from ground truth.
    :param impossible_value: We define an impossible value for we to distinguish where we left unfilled elements.
    :return: IMU original input and ground truth with unfilled values.
    """
    # Initialize aux matrix with impossible value.
    aux_matrix = ones((imu_data.shape[0], ground_truth.shape[1])) * impossible_value

    # Now we try to align ground truth with IMU timestamp (undersampling).
    for sample in ground_truth:
        _, index = find_nearest(imu_data[:, 0], sample[0])
        aux_matrix[index] = sample

    return imu_data, aux_matrix


def format_dataset(dataset_directory="dataset-room2_512_16", enable_asymetrical=True, sampling_window_size=10, file_format="npy"):
    """
A utilidade desta função eh fazer o split do dataset no formato de serie
temporal e convete-lo para arquivos numpy (NPZ), de forma que a classe que criamos
para dataset no PyTorch possa opera-lo sem estouro de memoria RAM. Pois se
abrissemos ele inteiro de uma vez, aconteceria overflow de memoria.

O formato de dataset esperado eh o dataset visual-inercial da TUM.

    :param dataset_directory: Diretorio onde se encontra o dataset (inclui o nome da sequencia).
    :param sampling_window_size: (ignorado) O tamanho da janela de entrada na serie temporal (Se for simetrica. Do contrario, ignorado).
    :param enable_asymetrical: (ignorado) Se a serie eh assimetrica.
    :return: Arrays X e y completos.
    :param file_format: NPY (deafult): 1 Arquivo para X e outro para Y. NPZ: Cada linha das matrizes X e Y gera 1 arquivo dentro dos NPZ de saida.
    """
    counter = 0
    for interval in tqdm(range(1, 201)[::-1], desc="Split do dataset"):
        # Opening dataset.
        input_data = read_csv(dataset_directory + "/mav0/imu0/data.csv").to_numpy()
        output_data = read_csv(dataset_directory + "/mav0/mocap0/data.csv").to_numpy()

        # ===============DIFF=======================================================
        # Precisamos restaurar o time para alinhar os dados depois do "diff"
        original_ground_truth_timestamp = output_data[:, 0]

        # inutil agora, mas deixarei aqui pra nao ter que refazer depois
        original_imu_timestamp = input_data[:, 0]

        # Queremos apenas a VARIACAO de posicao a cada instante.
        output_data = difference(output_data, interval=interval)
        # Restauramos a referencia de time original.
        output_data[:, 0] = original_ground_truth_timestamp[interval:]
        # ===============fim-de-DIFF================================================

        # features without timestamp (we do not scale timestamp)
        input_features = input_data[:, 1:]
        output_features = output_data[:, 1:]

        # Scaling data
        input_scaler = StandardScaler()
        input_features = input_scaler.fit_transform(input_features)
        output_scaler = MinMaxScaler()
        output_features = output_scaler.fit_transform(output_features)

        # Replacing scaled data (we kept the original TIMESTAMP)
        input_data[:, 1:] = input_features
        output_data[:, 1:] = output_features

        # A IMU e o ground truth nao sao coletados ao mesmo tempo, precisamos alinhar.
        x, y = alinha_dataset_tum(input_data, output_data)

        # Depois de alinhado, timestamp nao nos importa mais. Vamos descartar
        x = x[:, 1:]
        y = y[:, 1:]

        # Divido x e y em diferentes conjuntos de mesmo tamanho e alinhados
        x_chunks = split_into_chunks(x, 10 ** 9)
        y_chunks = split_into_chunks(y, 10 ** 9)

        # Vou concatenando os dados ja "splittados" nestes arrays
        X_array = None
        y_array = None

        # A partir daqui, o bagulho fica louco. Vou dividir a entrada em pequenos
        # datasets e fazer o split deles individualente para ter mais granularidade
        # no tamanho das sequencias e nao resultar tambem em senquencias muito
        # longas (com o tamanho do dataset inteiro).
        for x, y in list(zip(x_chunks, y_chunks)):
            # Fazemos o carregamento correto no formato de serie temporal
            X, y = timeseries_split(data_x=x, data_y=y, enable_asymetrical=False, sampling_window_size=interval)
            # Um ajuste na dimensao do y pois prevemos so o proximo passo.
            y = y.reshape(-1, 7)

            X = X.astype("float32")
            y = y.astype("float32")

            # Agora jogamos fora os valores onde nao ha ground truth e serviram apenas
            # para fazermos o alinhamento e o dataloader correto.
            samples_validas = where(y > -44000, True, False)
            X = X[samples_validas[:, 0]]
            y = y[samples_validas[:, 0]]

            if X_array is None and y_array is None:
                X_array = X.copy()
                y_array = y.copy()
            else:
                X_array = concatenate((X_array, X.copy()))
                y_array = concatenate((y_array, y.copy()))

        # Apena para manter o padrao de nomenclatura
        X = X_array
        y = y_array

        keys_to_concatenate = ["arr_" + str(i) for i in range(counter, X.shape[0] + counter)]
        counter += X.shape[0]

        if file_format.lower() == "npy":
            with open("x_data.npy", "wb") as x_file, open("y_data.npy", "wb") as y_file:
                save(x_file, X)
                save(y_file, y)
        else:
            with open("tmp_x/x_data" + str(counter) + ".npz", "wb") as x_file, open("tmp_y/y_data" + str(counter) + ".npz", "wb") as y_file:
                # Asterisco serve pra abrir a LISTA como se fosse *args.
                # Dois asteriscos serviriam pra abrir um DICIONARIO como se fosse **kwargs.
                savez(x_file, **dict(zip(keys_to_concatenate, X)))
                savez(y_file, **dict(zip(keys_to_concatenate, y)))

    return X, y


def join_npz_files(files_origin_path="./", output_file="./x_data.npz"):
    with open(output_file, "wb") as file:
        npfiles = glob.glob(os.path.normpath(files_origin_path) + "/" + "*.npz")
        npfiles.sort()
        all_arrays = []
        for i, npfile in enumerate(npfiles):
            npz_file = load(npfile)
            files_names = npz_file.files
            all_arrays.extend([npz_file[file_name] for file_name in files_names])  # , mmap_mode="r"
        savez(file, *all_arrays)
    return
