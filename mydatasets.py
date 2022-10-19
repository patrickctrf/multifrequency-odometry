import itertools
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import torch
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from torch import from_numpy, cat
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Specifying which modules to import when "import *" is called over this module.
# Also avoiding to import the smae things this module imports
__all__ = ["AsymetricalTimeseriesDataset", "BatchTimeseriesDataset", "ParallelBatchTimeseriesDataset", "CustomDataLoader"]

from ptk.utils.numpytools import find_nearest, axis_angle_into_quaternion, rotation_matrix_into_axis_angle, axis_angle_into_rotation_matrix, quaternion_into_axis_angle


class AsymetricalTimeseriesDataset(Dataset):

    def __init__(self, x_csv_path, y_csv_path, max_window_size=200, min_window_size=10, noise=None, convert_first=False, device=torch.device("cpu"), shuffle=True,
                 reference_x_csv_path="dataset-files/V1_01_easy/mav0/imu0/data.csv", reference_y_csv_path="dataset-files/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv"):
        super().__init__()
        self.input_data = read_csv(x_csv_path).to_numpy()
        self.output_data = read_csv(y_csv_path).to_numpy()
        self.min_window_size = min_window_size
        self.convert_first = convert_first
        self.device = device
        self.noise = noise

        # =========SCALING======================================================
        # features without timestamp (we do not scale timestamp)
        input_features = self.input_data[:, 1:]
        output_features = self.output_data[:, 1:]

        # Scaling data
        self.input_scaler, self.output_scaler = \
            AsymetricalTimeseriesDataset.get_reference_scaler(
                reference_x_csv_path, reference_y_csv_path
            )
        input_features = self.input_scaler.transform(input_features)
        output_features = self.output_scaler.transform(output_features)

        # Replacing scaled data (we kept the original TIMESTAMP and quaternions)
        self.input_data[:, 1:] = input_features
        self.output_data[:, 1:4] = output_features[:, :3]
        # =========end-SCALING==================================================

        # Save timestamps for syncing samples.
        self.input_timestamp = self.input_data[:, 0]
        self.output_timestamp = self.output_data[:, 0]

        # Throw out timestamp, we are not going to RETURN this.
        self.input_data = self.input_data[:, 1:]
        self.output_data = self.output_data[:, 1:]

        # We are calculating the number of samples that each window size produces.
        # We discount the window SIZE from the total number of samples (timeseries).
        # P.S.: It MUST use OUTPUT shape, because unlabeled data doesnt not help us.
        # P.S. 2: The first window_size is min_window_size, NOT 1.
        n_samples_per_window_size = np.ones((max_window_size - min_window_size,)) * self.output_data.shape[0] - np.arange(min_window_size + 1, max_window_size + 1)

        # Now, we know the last index where we can sample for each window size.
        # Concatenate element [0] in the begining to avoid error on first indices.
        self.last_window_sample_idx = np.hstack((np.array([0]), np.cumsum(n_samples_per_window_size))).astype("int")

        self.length = int(n_samples_per_window_size.sum())
        self.indices = np.arange(self.length)

        self.shuffle_array = np.arange(self.length)
        if shuffle is True: np.random.shuffle(self.shuffle_array)

        return

    def __getitem__(self, idx):
        """
Get itens from dataset according to idx passed. The return is in numpy arrays.

        :param idx: Index or slice to return.
        :return: 2 elements or 2 lists (x,y) values, according to idx.
        """

        # If we receive an index, return the sample.
        # Else, if receiving an slice or array, return an slice or array from the samples.
        if not isinstance(idx, slice):

            if idx >= len(self) or idx < 0:
                raise IndexError('Index out of range')

            # shuffling indices before return
            idx = self.shuffle_array[idx]

            argwhere_result = np.argwhere(self.last_window_sample_idx < idx)
            window_size = self.min_window_size + (argwhere_result[-1][0] if argwhere_result.size != 0 else 0)

            window_start_idx = idx - self.last_window_sample_idx[(argwhere_result[-1][0] if argwhere_result.size != 0 else 0)]

            _, x_start_idx = find_nearest(self.input_timestamp, self.output_timestamp[window_start_idx])
            _, x_finish_idx = find_nearest(self.input_timestamp, self.output_timestamp[window_start_idx + window_size])

            x = self.input_data[x_start_idx: x_finish_idx + 1]
            # Initialize y with proper shape. Neither quaternion or position are calculated this way anymore
            y = self.output_data[window_start_idx + window_size] - self.output_data[window_start_idx]

            # Calculate quaternion variation.Converts into rotation matriz first
            y[3:] = \
                axis_angle_into_quaternion(
                    *rotation_matrix_into_axis_angle(
                        # multiplication between current orientation matrix and
                        # inverse of PREVIOUS orientation matrix must give us
                        # orientation variation.
                        np.matmul(
                            # inverse (r.transpose) matrix of previous quaternion
                            axis_angle_into_rotation_matrix(
                                *quaternion_into_axis_angle(
                                    self.output_data[window_start_idx][3:]
                                )
                            ).T,
                            # Current orientation matrix (from quaternion)
                            axis_angle_into_rotation_matrix(
                                *quaternion_into_axis_angle(
                                    self.output_data[window_start_idx + window_size][3:]
                                )
                            )
                        )
                    )
                )

            # Position variation procedure. First, get body-frame orientation
            # This matrix describes the convertion from reference (or ground
            # truth) frame to IMU (or body) frame
            rotation_matrix_body_to_ref = axis_angle_into_rotation_matrix(
                *quaternion_into_axis_angle(
                    self.output_data[window_start_idx][3:]
                )
            ).T

            # Position variation seen from body frame.
            y[:3] = \
                np.matmul(rotation_matrix_body_to_ref,
                          self.output_data[window_start_idx + window_size][:3]) - \
                np.matmul(rotation_matrix_body_to_ref,
                          self.output_data[window_start_idx][:3])

            # We add gaussian noise to data, if configured to.
            if self.noise is not None:
                x = x + np.random.normal(loc=self.noise[0], scale=self.noise[1], size=x.shape)
                y = y + np.random.normal(loc=self.noise[0], scale=self.noise[1], size=y.shape)

            # If we want to convert into torch tensors first
            if self.convert_first is True:
                return from_numpy(x.astype("float32")).to(self.device), \
                       from_numpy(y.astype("float32")).to(self.device)
            else:
                return x, y
        else:
            # If we received a slice(e.g., 0:10:-1) instead an single index.
            return self.__getslice__(idx)

    def __getslice__(self, slice_from_indices):
        return list(zip(*[self[i] for i in self.indices[slice_from_indices]]))

    def __len__(self):
        return self.length

    @staticmethod
    def get_reference_scaler(x_csv_path="dataset-room1_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room1_512_16/mav0/mocap0/data.csv"):
        """
    We need to work with fixed scalers, so we can reproduce results. Thus
    we need to keep them saved or to recalculate them over a referecne
    dataset. In both cases, this method will help you.

        :return: (input_scaler, output_scaler)
        """

        input_data = read_csv(x_csv_path).to_numpy()
        output_data = read_csv(y_csv_path).to_numpy()

        # =========SCALING======================================================
        # features without timestamp (we do not scale timestamp)
        input_features = input_data[:, 1:]
        output_features = output_data[:, 1:]

        # Scaling data
        input_scaler = StandardScaler()
        input_scaler.fit(input_features)
        output_scaler = StandardScaler()
        output_scaler.fit(output_features)

        return input_scaler, output_scaler


class BatchTimeseriesDataset(Dataset):
    def __init__(self, x_csv_path, y_csv_path, max_window_size=200, min_window_size=10, shuffle=True, batch_size=1, noise=None,
                 reference_x_csv_path="dataset-files/V1_01_easy/mav0/imu0/data.csv", reference_y_csv_path="dataset-files/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv"):
        super().__init__()
        self.batch_size = batch_size

        self.base_dataset = \
            AsymetricalTimeseriesDataset(x_csv_path=x_csv_path,
                                         y_csv_path=y_csv_path,
                                         max_window_size=max_window_size,
                                         min_window_size=min_window_size,
                                         convert_first=True,
                                         device=torch.device("cpu"),
                                         shuffle=False,
                                         noise=noise,
                                         reference_x_csv_path=reference_x_csv_path,
                                         reference_y_csv_path=reference_y_csv_path)

        try:
            tabela = np.load("dataset-cache/" + str(max_window_size) + str(min_window_size) +
                             str(x_csv_path).replace("/", "_").replace(".", "_") +
                             "_tabela_elementos_dataset.npy")

        except FileNotFoundError as e:
            print("Indexing dataset samples:")
            tabela = np.zeros((len(self.base_dataset),))
            i = 0
            for element in tqdm(self.base_dataset):
                tabela[i] = element[0].shape[0]
                i = i + 1
            np.save("dataset-cache/" + str(max_window_size) + str(min_window_size) +
                    str(x_csv_path).replace("/", "_").replace(".", "_") +
                    "_tabela_elementos_dataset.npy", tabela)

        # dict_count = Counter(tabela)
        # ocorrencias = array(list(dict_count.values()))

        # Os arrays nesta lista contem INDICES para elementos do dataset com
        # mesmo comprimento.
        self.lista_de_arrays_com_mesmo_comprimento = []

        print("Loading sample info...")
        # grupos de arrays com mesmo comprimento.
        # Eles serao separados em batches, entao alguns
        # batches sao de arrays com mesmo comprimento que outros. Alguns
        # batches serao um pouco maiores, pois a quantidade de elementos com o
        # mesmo tamanho talvez nao seja um multiplo inteiro do batch_size
        # escolhido
        for i in tqdm(range(tabela.min().astype("int"), tabela.max().astype("int") + 1)):
            if np.where(tabela == i)[0].shape[0] // self.batch_size + (np.where(tabela == i)[0].shape[0] % self.batch_size > 0) <= 1:
                # Se nao houver nenhuma sample daquele tamanho, pule a iteracao
                # Se houver so 1 sample, pularemos tambem, porque a Batchnorm
                # buga e um batch com so 1 sample eh instavel demais
                pass
            else:
                self.lista_de_arrays_com_mesmo_comprimento.extend(
                    np.array_split(np.where(tabela == i)[0],
                                   np.where(tabela == i)[0].shape[0] // self.batch_size + (np.where(tabela == i)[0].shape[0] % self.batch_size > 0))
                )

        self.length = len(self.lista_de_arrays_com_mesmo_comprimento)

        self.shuffle_array = np.arange(self.length)

        if shuffle is True:
            np.random.shuffle(self.shuffle_array)

        return

    def __getitem__(self, idx):
        """
Get itens from dataset according to idx passed. The return is in numpy arrays.

        :param idx: Index to return.
        :return: 2 elements (batches), according to idx.
        """

        # If we are shuffling indices, we do it here. Else, we'll just get the
        # same index back
        idx = self.shuffle_array[idx]

        # concatenate tensor in order to assemble batches
        x_batch = \
            cat([self.base_dataset[dataset_element_idx][0].unsqueeze(0)
                 for dataset_element_idx
                 in self.lista_de_arrays_com_mesmo_comprimento[idx]], 0)
        y_batch = \
            cat([self.base_dataset[dataset_element_idx][1].unsqueeze(0)
                 for dataset_element_idx
                 in self.lista_de_arrays_com_mesmo_comprimento[idx]], 0)

        return x_batch, y_batch

    def __len__(self):
        return self.length


def _load_data_for_parallel_dataset(base_dataset, array_idx):
    return base_dataset[array_idx][0].unsqueeze(0), base_dataset[array_idx][1].unsqueeze(0)


class ParallelBatchTimeseriesDataset(Dataset):
    def __init__(self, x_csv_path, y_csv_path, max_window_size=200, min_window_size=10, shuffle=True, batch_size=1, noise=None,
                 reference_x_csv_path="dataset-files/V1_01_easy/mav0/imu0/data.csv", reference_y_csv_path="dataset-files/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                 n_threads=10):
        super().__init__()
        self.batch_size = batch_size

        # Thread pool assembles batches
        self.n_threads = n_threads

        self.base_dataset = \
            AsymetricalTimeseriesDataset(x_csv_path=x_csv_path,
                                         y_csv_path=y_csv_path,
                                         max_window_size=max_window_size,
                                         min_window_size=min_window_size,
                                         convert_first=True,
                                         device=torch.device("cpu"),
                                         shuffle=False,
                                         noise=noise,
                                         reference_x_csv_path=reference_x_csv_path,
                                         reference_y_csv_path=reference_y_csv_path)

        try:
            tabela = np.load("dataset-cache/" + str(max_window_size) + str(min_window_size) +
                             str(x_csv_path).replace("/", "_").replace(".", "_") +
                             "_tabela_elementos_dataset.npy")

        except FileNotFoundError as e:
            print("Indexing dataset samples:")
            tabela = np.zeros((len(self.base_dataset),))
            i = 0
            for element in tqdm(self.base_dataset):
                tabela[i] = element[0].shape[0]
                i = i + 1
            np.save("dataset-cache/" + str(max_window_size) + str(min_window_size) +
                    str(x_csv_path).replace("/", "_").replace(".", "_") +
                    "_tabela_elementos_dataset.npy", tabela)

        # dict_count = Counter(tabela)
        # ocorrencias = array(list(dict_count.values()))

        # Os arrays nesta lista contem INDICES para elementos do dataset com
        # mesmo comprimento.
        self.lista_de_arrays_com_mesmo_comprimento = []

        print("Loading sample info...")
        # grupos de arrays com mesmo comprimento.
        # Eles serao separados em batches, entao alguns
        # batches sao de arrays com mesmo comprimento que outros. Alguns
        # batches serao um pouco maiores, pois a quantidade de elementos com o
        # mesmo tamanho talvez nao seja um multiplo inteiro do batch_size
        # escolhido
        for i in tqdm(range(tabela.min().astype("int"), tabela.max().astype("int") + 1)):
            if np.where(tabela == i)[0].shape[0] // self.batch_size + (np.where(tabela == i)[0].shape[0] % self.batch_size > 0) < 1 or np.where(tabela == i)[0].shape[0] == 1:
                # Se nao houver nenhuma sample daquele tamanho, pule a iteracao
                # Se houver so 1 sample, pularemos tambem, porque a Batchnorm
                # buga e um batch com so 1 sample eh instavel demais
                pass
            else:
                self.lista_de_arrays_com_mesmo_comprimento.extend(
                    np.array_split(np.where(tabela == i)[0],
                                   np.where(tabela == i)[0].shape[0] // self.batch_size + (np.where(tabela == i)[0].shape[0] % self.batch_size > 0))
                )

        self.length = len(self.lista_de_arrays_com_mesmo_comprimento)

        self.shuffle_array = np.arange(self.length)

        if shuffle is True:
            np.random.shuffle(self.shuffle_array)

        return

    def __getitem__(self, idx):
        """
Get itens from dataset according to idx passed. The return is in numpy arrays.

        :param idx: Index to return.
        :return: 2 elements (batches), according to idx.
        """

        # If we are shuffling indices, we do it here. Else, we'll just get the
        # same index back
        idx = self.shuffle_array[idx]

        self.pool = ThreadPool(self.n_threads)
        dataset_elements = self.pool.starmap(_load_data_for_parallel_dataset,
                                             zip(itertools.repeat(self.base_dataset),
                                                 self.lista_de_arrays_com_mesmo_comprimento[idx])
                                             )

        x_list, y_list = list(zip(*dataset_elements))

        # concatenate tensor in order to assemble batches
        x_batch = cat(x_list, 0)
        y_batch = cat(y_list, 0)

        return x_batch, y_batch

    def __len__(self):
        return self.length


class CustomDataLoader(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.dataloader = DataLoader(*args, **kwargs)
        self.iterable = None

    def __iter__(self):
        self.iterable = iter(self.dataloader)
        return self

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        next_sample = next(self.iterable)
        # Discard batch dimension, since dataloader is going to add this anyway
        return next_sample[0].view(next_sample[0].shape[1:]), \
               next_sample[1].view(next_sample[1].shape[1:])
