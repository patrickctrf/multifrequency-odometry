import csv
import math
import time
from threading import Thread, Event

import torch
from torch import nn, movedim, absolute
from torch.cuda.amp import autocast, GradScaler
from torch.nn import Sequential, Conv1d, AvgPool1d
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm

from activations import BnActivation
from losses import *
from mydatasets import ParallelBatchTimeseriesDataset, CustomDataLoader
from ptk.utils import DataManager
from useful import get_lr

# When importing every models from this module, make sure only models are
# imported
__all__ = ["InertialModule", "IMUHandler", ]


class LSTMLatentFeatures(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, n_lstm_units=1, bidirectional=False):
        """
This class implements the classical LSTM with 1 or more cells (stacked LSTM). It
receives sequences and returns the predcition at the end of each one.

There is a fit() method to train this model according to the parameters given in
the class initialization. It follows the sklearn header pattern.

This is also an sklearn-like estimator and may be used with any sklearn method
designed for classical estimators. But, when using GPU as PyTorch device, you
CAN'T use multiple sklearn workers (n_jobs), beacuse it raises an serializtion
error within CUDA.

        :param input_size: Input dimension size (how many features).
        :param hidden_layer_size: How many features there will be inside each LSTM.
        :param output_size: Output dimension size (how many features).
        :param n_lstm_units: How many stacked LSTM cells (or units).
        :param epochs: The number of epochs to train. The final model after
        train will be the one with best VALIDATION loss, not necessarily the
        model found after whole "epochs" number.
        :param training_batch_size: Size of each mini-batch during training
        process. If number os samples is not a multiple of
        "training_batch_size", the final batch will just be smaller than the
        others.
        :param validation_percent: The percentage of samples reserved for
        validation (cross validation) during training inside fit() method.
        :param bidirectional: If the LSTM units will be bidirectional.
        :param device: PyTorch device, such as torch.device("cpu") or
        torch.device("cuda:0").
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.n_lstm_units = n_lstm_units
        if bidirectional:
            self.bidirectional = 1
            self.num_directions = 2
        else:
            self.bidirectional = 0
            self.num_directions = 1

        self.lstm = nn.LSTM(input_size, self.hidden_layer_size,
                            batch_first=True, num_layers=self.n_lstm_units,
                            bidirectional=bool(self.bidirectional), )

        self.dense_network = Sequential(
            nn.Linear(self.num_directions * self.hidden_layer_size, 128),
            BnActivation(128, ),
            nn.Linear(128, self.output_size)
        )
        # We train using multiple inputs (mini_batch), so we let this cell ready
        # to be called.
        # self.hidden_cell_zeros = (torch.zeros((self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size), device=self.device),
        #                           torch.zeros((self.num_directions * self.n_lstm_units, self.training_batch_size, self.hidden_layer_size), device=self.device))
        self.hidden_cell_zeros = None

        self.hidden_cell_output = None

        return

    def forward(self, input_seq):
        """
Classic forward method of every PyTorch model, as fast as possible. Receives an
input sequence and returns the prediction for the final step.

        :param input_seq: Input sequence of the time series.
        :return: The prediction in the end of the series.
        """

        lstm_out, self.hidden_cell_output = self.lstm(input_seq, self.hidden_cell_zeros) 

        # All batch size, whatever sequence length, forward direction and
        # lstm output size (hidden size).
        # We only want the last output of lstm (end of sequence), that is
        # the reason of '[:,-1,:]'.
        return lstm_out.view(input_seq.shape[0], -1, self.num_directions * self.hidden_layer_size)[:, -1, :] 


class ResBlock(nn.Module):
    def __init__(self, n_input_channels=6, n_output_channels=7,
                 kernel_size=7, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros'):
        """
    ResNet-like block, receives as arguments the same that PyTorch's Conv1D
    module.
        """
        super(ResBlock, self).__init__()

        self.feature_extractor = \
            Sequential(
                nn.Conv1d(n_input_channels, n_output_channels, kernel_size,
                          stride, kernel_size // 2 * dilation, dilation,
                          groups, bias, padding_mode),
                BnActivation(n_output_channels), 
                nn.Conv1d(n_output_channels, n_output_channels, kernel_size,
                          stride, kernel_size // 2 * dilation,
                          dilation, groups, bias, padding_mode),
            )

        self.skip_connection = \
            Sequential(
                nn.Conv1d(n_input_channels, n_output_channels, 1,
                          stride, padding, dilation, groups, bias, padding_mode)
            )

        self.activation = BnActivation(n_output_channels)

    def forward(self, input_seq):
        return self.activation(self.feature_extractor(input_seq) + self.skip_connection(input_seq))


class Conv1DFeatureExtractor(nn.Module):
    def __init__(self, input_size=1, output_size=8):
        """
This class implements the classical LSTM with 1 or more cells (stacked LSTM). It
receives sequences and returns the predcition at the end of each one.

There is a fit() method to train this model according to the parameters given in
the class initialization. It follows the sklearn header pattern.

This is also an sklearn-like estimator and may be used with any sklearn method
designed for classical estimators. But, when using GPU as PyTorch device, you
CAN'T use multiple sklearn workers (n_jobs), beacuse it raises an serializtion
error within CUDA.

        :param input_size: Input dimension size (how many features).
        :param hidden_layer_size: How many features there will be inside each LSTM.
        :param output_size: Output dimension size (how many features).
        :param n_lstm_units: How many stacked LSTM cells (or units).
        :param epochs: The number of epochs to train. The final model after
        train will be the one with best VALIDATION loss, not necessarily the
        model found after whole "epochs" number.
        :param training_batch_size: Size of each mini-batch during training
        process. If number os samples is not a multiple of
        "training_batch_size", the final batch will just be smaller than the
        others.
        :param validation_percent: The percentage of samples reserved for
        validation (cross validation) during training inside fit() method.
        :param bidirectional: If the LSTM units will be bidirectional.
        :param device: PyTorch device, such as torch.device("cpu") or
        torch.device("cuda:0").
        """
        super().__init__()
        self.input_size = input_size

        # Half of the outputs are summed and the other half is Avg pooling.
        self.output_size = output_size

        # ATTENTION: You cannot change this anymore, since we added a sum layer
        # and it casts conv outputs to 1 feature per channel
        pooling_output_size = 1

        n_base_filters = 512
        n_output_features = 6 * n_base_filters
        self.feature_extractor = \
            Sequential(
                Conv1d(input_size, 1 * n_base_filters, (3,), dilation=(2,), stride=(3,)), AvgPool1d(2, 2), BnActivation(1 * n_base_filters),
                ResBlock(1 * n_base_filters, 2 * n_base_filters, ), AvgPool1d(2, 2),
                ResBlock(2 * n_base_filters, 3 * n_base_filters, ), AvgPool1d(2, 2),
                ResBlock(3 * n_base_filters, 4 * n_base_filters, ), AvgPool1d(2, 2),
                ResBlock(4 * n_base_filters, 5 * n_base_filters, ), AvgPool1d(2, 2),
                ResBlock(5 * n_base_filters, n_output_features, ),
            )

        self.sum_layer = SumLayer()
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(pooling_output_size)

        self.dense_network = Sequential(
            nn.Flatten(),
            nn.Linear(2 * pooling_output_size * n_output_features, 128),
            BnActivation(128, ),
            nn.Linear(128, self.output_size),
        )

        return

    def forward(self, input_seq):
        """
    Classic forward method of every PyTorch model, as fast as possible. Receives an
    input sequence and returns the prediction for the final step.

        :param input_seq: Input sequence of the time series.
        :return: The prediction in the end of the series.
        """

        # As features (px, py, pz, qw, qx, qy, qz) sao os "canais" da
        # convolucao e precisam vir no meio para o pytorch
        input_seq = movedim(input_seq, -2, -1) 

        input_seq = self.feature_extractor(input_seq)

        return torch.cat((
            self.sum_layer(input_seq),
            self.adaptive_pooling(input_seq),
        ), dim=1)


class _AttentionLayer(torch.nn.Module):

    def __init__(self, embedding_dim: int, max_seq_length: int = 300):
        """
        Implements the Self-attention, decoder-only."

        Args:
            max_seq_length (int): Size of the sequence to consider as context for prediction.
            embedding_dim (int): Dimension of the embedding layer for each word in the context.
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # hidden_size for MLP
        hidden_size = 2048

        # Linear projections
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)
        self.w_0 = nn.Linear(embedding_dim, embedding_dim)

        # output MLP
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            BnActivation(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, embedding_dim),
        )

        self.activation = nn.LeakyReLU()

        # cast to probabilities
        self.softmax = nn.Softmax(dim=-1)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Matriz triangular de mascara, convertida para Booleano
        # Onde vale 1, o valor deve ser substituida por um valor negativo alto no tensor de scores.
        self.casual_mask = torch.ones((max_seq_length, max_seq_length), ).triu(diagonal=1) == 1.0

    def forward(self, x_embeddings):
        k = self.w_k(x_embeddings)
        v = self.w_v(x_embeddings) 
        q = self.w_q(x_embeddings)

        scores = torch.matmul(q, k.transpose(1, 2))

        probabilities = self.softmax(scores)

        e = self.w_0(self.norm1(x_embeddings + torch.matmul(probabilities, v)))

        logits = self.mlp(e)

        return self.norm2(self.activation(logits + e))


class Transformer(torch.nn.Module):

    def __init__(self, dim: int, n_layers: int, max_seq_length: int = 300, input_size: int = 6, output_size: int = 7):
        """
        Implements the Self-attention, decoder-only."

        Args:
            input_size (int): Size of the input features.
            max_seq_length (int): Size of the sequence to consider as context for prediction.
            dim (int): Dimension of the embedding layer for each word in the context.
            n_layers (int): number of self-attention layers.
        """
        # Escreva seu código aqui.
        super().__init__()
        embedding_dim = dim
        self.embedding_dim = embedding_dim

        # tokens (words indexes) embedding and positional embedding
        self.c_embedding = nn.Sequential(
            nn.Linear(input_size, embedding_dim),
            BnActivation(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.p_embedding = nn.Embedding(max_seq_length, embedding_dim)

        self.attention = nn.Sequential(*[_AttentionLayer(embedding_dim=embedding_dim) for _ in range(n_layers)])

        self.CLS = nn.Parameter(torch.randn((1, 1, input_size,)))

        self.dense_network = nn.Linear(embedding_dim, output_size)

    def forward(self, inputs):
        # Precisa adicionar o token CLS no inicio de cada sequencia
        inputs = torch.cat((self.CLS.repeat(inputs.shape[0], 1, 1), inputs), dim=1)

        positional_indexes = torch.arange(inputs.shape[1], device=inputs.device).view(1, -1) 

        input_embeddings = self.c_embedding(inputs)

        positional_embeddings = self.p_embedding(positional_indexes.repeat(inputs.shape[0], 1))

        x_embeddings = positional_embeddings + input_embeddings

        logits = self.attention(x_embeddings)

        return logits[:, 0]


class InertialModule(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, n_lstm_units=1, epochs=150, training_batch_size=64, validation_percent=0.2, bidirectional=False,
                 device=torch.device("cpu"),
                 use_amp=True):
        """
This class implements the classical LSTM with 1 or more cells (stacked LSTM). It
receives sequences and returns the predcition at the end of each one.

There is a fit() method to train this model according to the parameters given in
the class initialization. It follows the sklearn header pattern.

This is also an sklearn-like estimator and may be used with any sklearn method
designed for classical estimators. But, when using GPU as PyTorch device, you
CAN'T use multiple sklearn workers (n_jobs), beacuse it raises an serializtion
error within CUDA.

        :param input_size: Input dimension size (how many features).
        :param hidden_layer_size: How many features there will be inside each LSTM.
        :param output_size: Output dimension size (how many features).
        :param n_lstm_units: How many stacked LSTM cells (or units).
        :param epochs: The number of epochs to train. The final model after
        train will be the one with best VALIDATION loss, not necessarily the
        model found after whole "epochs" number.
        :param training_batch_size: Size of each mini-batch during training
        process. If number os samples is not a multiple of
        "training_batch_size", the final batch will just be smaller than the
        others.
        :param validation_percent: The percentage of samples reserved for
        validation (cross validation) during training inside fit() method.
        :param bidirectional: If the LSTM units will be bidirectional.
        :param device: PyTorch device, such as torch.device("cpu") or
        torch.device("cuda:0").
        """
        super().__init__()
        self.creation_time = time.time()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.training_batch_size = training_batch_size
        self.epochs = epochs
        self.validation_percent = validation_percent
        self.n_lstm_units = n_lstm_units
        if bidirectional:
            self.bidirectional = 1
            self.num_directions = 2
        else:
            self.bidirectional = 0
            self.num_directions = 1
        self.device = device
        self.use_amp = use_amp  # Automatic Mixed Precision (float16 and float32)

        # Proporcao entre dados de treino e de validacao
        self.train_percentage = 1 - self.validation_percent

        self.loss_function = None
        self.optimizer = None

        print("creation_time: ", self.creation_time)

        # n_base_filters = 8
        # n_output_features = 8
        # self.feature_extractor = \
        #     LSTMLatentFeatures(input_size=input_size,
        #                        hidden_layer_size=hidden_layer_size,
        #                        output_size=output_size,
        #                        n_lstm_units=n_lstm_units,
        #                        bidirectional=bidirectional)

        self.feature_extractor = Conv1DFeatureExtractor(input_size=input_size,
                                                        output_size=output_size)

        # self.feature_extractor = Transformer(dim=256, n_layers=2,
        #                                      input_size=input_size,
        #                                      output_size=output_size)

        # Assim nao precisamos adaptar a rede densa a uma saida de CNN ou LSTM,
        # ja pegamos a rede adaptada do proprio extrator de
        # features (seja lstm ou CNN)
        self.dense_network = self.feature_extractor.dense_network

        return

    def forward(self, input_seq):
        """
Classic forward method of every PyTorch model, as fast as possible. Receives an
input sequence and returns the prediction for the final step.

        :param input_seq: Input sequence of the time series.
        :return: The prediction in the end of the series.
        """

        output_seq = self.feature_extractor(input_seq)

        predictions = self.dense_network(output_seq)

        pos = predictions[:, 0:3]
        quat = torch.nn.functional.normalize(predictions[:, 3:7])

        return torch.cat((pos, quat), dim=1)

    def fit(self):
        """
This method contains the customized script for training this estimator. Data is
obtained with PyTorch's dataset and dataloader classes for memory efficiency
when dealing with big datasets. Otherwise loading the whole dataset would
overflow the memory.

        :return: Trained model with best validation loss found (it uses checkpoint).
        """
        print("creation_time: ", self.creation_time)
        self.train()
        self.to(self.device)
        # =====DATA-PREPARATION=================================================
        euroc_v1_01_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/V1_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                             n_threads=2,
                                                             min_window_size=200, max_window_size=201, batch_size=self.training_batch_size, shuffle=False, noise=None)

        # Esse daqui gera NAN no treino e na validacao, melhor nao usar
        euroc_v2_01_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/V2_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                             n_threads=2,
                                                             min_window_size=200, max_window_size=201, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_v2_02_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/V2_02_medium/mav0/imu0/data.csv",
                                                             y_csv_path="dataset-files/V2_02_medium/mav0/state_groundtruth_estimate0/data.csv", n_threads=2,
                                                             min_window_size=200, max_window_size=201, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_v2_03_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/V2_03_difficult/mav0/imu0/data.csv",
                                                             y_csv_path="dataset-files/V2_03_difficult/mav0/state_groundtruth_estimate0/data.csv", n_threads=2,
                                                             min_window_size=200, max_window_size=201, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_v1_02_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/V1_02_medium/mav0/imu0/data.csv",
                                                             y_csv_path="dataset-files/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv", n_threads=2,
                                                             min_window_size=200, max_window_size=201, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_v1_03_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/V1_03_difficult/mav0/imu0/data.csv",
                                                             y_csv_path="dataset-files/V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv", n_threads=2,
                                                             min_window_size=200, max_window_size=201, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_mh1_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/MH_01_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                           n_threads=2,
                                                           min_window_size=200, max_window_size=201, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_mh2_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/MH_02_easy/mav0/imu0/data.csv", y_csv_path="dataset-files/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv",
                                                           n_threads=2,
                                                           min_window_size=200, max_window_size=201, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_mh3_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/MH_03_medium/mav0/imu0/data.csv",
                                                           y_csv_path="dataset-files/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv", n_threads=2,
                                                           min_window_size=200, max_window_size=201, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_mh4_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/MH_04_difficult/mav0/imu0/data.csv",
                                                           y_csv_path="dataset-files/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv", n_threads=2,
                                                           min_window_size=200, max_window_size=201, batch_size=self.training_batch_size, shuffle=False, noise=None)

        euroc_mh5_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-files/MH_05_difficult/mav0/imu0/data.csv",
                                                           y_csv_path="dataset-files/MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv", n_threads=2,
                                                           min_window_size=200, max_window_size=201, batch_size=self.training_batch_size, shuffle=False, noise=None)

        dummy_dataset = Subset(euroc_v1_01_dataset, range(1))

        # room1_tum_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-room1_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room1_512_16/mav0/mocap0/data.csv", n_threads=2,
        #                                                    min_window_size=40, max_window_size=201, batch_size=self.training_batch_size, shuffle=False, noise=None)
        #
        # room2_tum_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-room2_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room2_512_16/mav0/mocap0/data.csv", n_threads=2,
        #                                                    min_window_size=40, max_window_size=201, batch_size=self.training_batch_size, shuffle=False)
        #
        # room3_tum_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-room3_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room3_512_16/mav0/mocap0/data.csv", n_threads=2,
        #                                                    min_window_size=200, max_window_size=201, batch_size=self.training_batch_size, shuffle=False, noise=None)
        #
        # room4_tum_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-room4_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room4_512_16/mav0/mocap0/data.csv", n_threads=2,
        #                                                    min_window_size=200, max_window_size=201, batch_size=self.training_batch_size, shuffle=False)
        #
        # room5_tum_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-room5_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room5_512_16/mav0/mocap0/data.csv", n_threads=2,
        #                                                    min_window_size=150, max_window_size=200, batch_size=self.training_batch_size, shuffle=False, noise=None)
        #
        # room6_tum_dataset = ParallelBatchTimeseriesDataset(x_csv_path="dataset-room6_512_16/mav0/imu0/data.csv", y_csv_path="dataset-room6_512_16/mav0/mocap0/data.csv", n_threads=2,
        #                                                    min_window_size=150, max_window_size=200, batch_size=self.training_batch_size, shuffle=False)

        # # Diminuir o dataset para verificar o funcionamento de scripts
        # room1_tum_dataset = Subset(room1_tum_dataset, arange(int(len(room1_tum_dataset) * 0.001)))

        # train_dataset = Subset(room1_tum_dataset, arange(int(len(room1_tum_dataset) * self.train_percentage)))
        # val_dataset = Subset(room1_tum_dataset, arange(int(len(room1_tum_dataset) * self.train_percentage), len(room1_tum_dataset)))

        train_dataset = ConcatDataset([euroc_v1_01_dataset,
                                       euroc_v1_02_dataset,
                                       euroc_mh4_dataset])
        val_dataset = ConcatDataset([euroc_v2_02_dataset, euroc_mh3_dataset])

        train_loader = CustomDataLoader(dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=10, multiprocessing_context='spawn')
        val_loader = CustomDataLoader(dataset=val_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=10, multiprocessing_context='spawn')

        # =====fim-DATA-PREPARATION=============================================

        epochs = self.epochs
        best_validation_loss = 999999
        if self.loss_function is None: self.loss_function = PosAndAngleLoss()
        if self.optimizer is None: self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, )
        scaler = GradScaler(enabled=self.use_amp)
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: max(math.exp(-epoch), 0.1))

        f = open("loss_log.csv", "w")
        w = csv.writer(f)
        w.writerow(["epoch", "training_loss", "val_loss"])

        tqdm_bar = tqdm(range(epochs))
        for i in tqdm_bar:
            self.train()
            train_manager = DataManager(train_loader, device=self.device, buffer_size=2)
            val_manager = DataManager(val_loader, device=self.device, buffer_size=2)
            training_loss = 0.0
            validation_loss = 0.0
            ponderar_losses = 0.0
            self.optimizer.zero_grad()

            for j, (X, y) in enumerate(train_manager):
                # # Precisamos resetar o hidden state do LSTM a cada batch, ou
                # # ocorre erro no backward(). O tamanho do batch para a cell eh
                # # simplesmente o tamanho do batch em y ou X (tanto faz).
                # self.feature_extractor.hidden_cell_zeros = (torch.zeros((self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size), device=self.device),
                #                                             torch.zeros((self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size), device=self.device))
                with autocast(enabled=self.use_amp):
                    y_pred = self(X)
                    # O peso do batch no calculo da loss eh proporcional ao seu
                    # tamanho.
                    single_loss = self.loss_function(y_pred, y) * X.shape[0] / 1e6
                scaler.scale(single_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

                # print("t GPU: ", '{:f}'.format(time.time() - t))
                # t = time.time()
                # We need detach() to no accumulate gradient. Otherwise, memory
                # overflow will happen.
                # We divide by the size of batch because we dont need to
                # compensate batch size when estimating average training loss,
                # otherwise we woul get an explosive and incorrect loss value.
                training_loss += single_loss.detach()

                ponderar_losses += X.shape[0] / 1e6

            # Tira a media ponderada das losses.
            training_loss = training_loss / ponderar_losses

            ponderar_losses = 0.0

            # Nao precisamos perder tempo calculando gradientes da loss durante
            # a validacao
            with torch.no_grad():
                # validando o modelo no modo de evaluation
                self.eval()
                for j, (X, y) in enumerate(val_manager):
                    # # Precisamos resetar o hidden state do LSTM a cada batch, ou
                    # # ocorre erro no backward(). O tamanho do batch para a cell eh
                    # # simplesmente o tamanho do batch em y ou X (tanto faz).
                    # self.feature_extractor.hidden_cell_zeros = (torch.zeros((self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size), device=self.device),
                    #                                             torch.zeros((self.num_directions * self.n_lstm_units, X.shape[0], self.hidden_layer_size), device=self.device))

                    y_pred = self(X)
                    single_loss = self.loss_function(y_pred, y) * X.shape[0] / 1e6

                    validation_loss += single_loss.detach()

                    ponderar_losses += X.shape[0] / 1e6

            # Tira a media ponderada das losses.
            validation_loss = validation_loss / ponderar_losses

            # Checkpoint to best models found.
            if best_validation_loss > validation_loss:
                # Update the new best loss.
                best_validation_loss = validation_loss
                self.eval()
                torch.save(self, "best_model.pth")
                torch.save(self.state_dict(), "best_model_state_dict.pth")

            tqdm_bar.set_description(f'epoch: {i:1} train_loss: {training_loss.item():10.10f}' +
                                     f' val_loss: {validation_loss.item():10.10f}' +
                                     f' LR: {get_lr(self.optimizer):1.7f}')
            w.writerow([i, training_loss.item(), validation_loss.item()])
            f.flush()

            # scheduler.step()
        f.close()

        self.eval()

        # At the end of training, save the final model.
        torch.save(self, "last_training_model.pth")

        # Update itself with BEST weights foundfor each layer.
        self.load_state_dict(torch.load("best_model_state_dict.pth"))

        self.eval()

        # Returns the best model found so far.
        return torch.load("best_model.pth")


class IMUHandler(nn.Module, Thread):

    def __init__(self, sampling_window_size=200, imu_input_size=6,
                 position_output_size=7, device=torch.device("cpu"), dtype=torch.float32):
        """
This module receives IMU samples and uses the InertialModule to predict our
current position at real time.

Our estimated position takes into account the estimated position
at sampling_window_size IMU samples before, summed with the displacement
predicted by InertialModule after all those IMU samples. It is an arithmetic
summation of each displacement and doesnt takes into account any estimated
trajectory for the object being tracked.

ATENTION: You need to call load_feature_extractor() after instantiating this
class, or your InertialModule predictor will just be a random output. You also
need to call this object.start() to begin updating positions in another thread.

        :param sampling_window_size: (Integer) How many IMU samples to calculate the displacement. Default = 200.
        :param imu_input_size: (Integer) How many features IMU give to us. Default = 6 (ax, ay, az, wx, wq, wz).
        :param position_output_size: (Integer) How many features we use to represent our position annotation. Default = 7 (px, py, pz, qw, qx, qy, qz).
        """
        nn.Module.__init__(self)
        Thread.__init__(self)
        self.sampling_window_size = sampling_window_size
        self._imu_input_size = imu_input_size
        self._position_output_size = position_output_size
        self.device = device
        self.dtype = dtype

        # Event synchronizer for position updates. We'll only estimate a new
        # position if new IMU samples have arrived.
        self._imu_samples_arrived = Event()

        # Indicate for clients that our position predictions have been updated.
        self.new_predictions_arrived = Event()

        # A control flag ordering this thread to stop.
        self.stop_flag = False

        # Here we are gonna store previous IMU samples and our estimated
        # positions at that time. Also, we need a timestamp info to synchronize
        # them. These buffers will grow as new samples and predictions arrive.
        self.predictions_buffer = torch.zeros((1, 1 + position_output_size,),
                                              dtype=self.dtype,
                                              device=self.device,
                                              requires_grad=False)
        # First dimension will always be 1, because of batch dimension.
        self.imu_buffer = torch.zeros((1, 1, 1 + imu_input_size,),
                                      dtype=self.dtype,
                                      device=self.device,
                                      requires_grad=False)

        # This avoids recalculating these thresholds.
        self._imu_buffer_size_limit = 2 * sampling_window_size
        self._imu_buffer_reduction_threshold = int(1.5 * sampling_window_size)

        # ========DEFAULT-PREDICTOR-FOR-COMPATIBILITY===========================
        # Here we define a default InertialModule component, but your are
        # strongly encouraged to load_feature_extractor(), since the modules
        # defined here are not trained yet.
        pooling_output_size = 100
        n_base_filters = 72
        n_output_features = 200
        self.feature_extractor = \
            Sequential(
                Conv1d(imu_input_size, 1 * n_base_filters, (7,)), nn.PReLU(), nn.BatchNorm1d(1 * n_base_filters),
                Conv1d(1 * n_base_filters, 2 * n_base_filters, (7,)), nn.PReLU(), nn.BatchNorm1d(2 * n_base_filters),
                Conv1d(2 * n_base_filters, 3 * n_base_filters, (7,)), nn.PReLU(), nn.BatchNorm1d(3 * n_base_filters),
                Conv1d(3 * n_base_filters, 4 * n_base_filters, (7,)), nn.PReLU(), nn.BatchNorm1d(4 * n_base_filters),
                Conv1d(4 * n_base_filters, n_output_features, (7,)), nn.PReLU(), nn.BatchNorm1d(n_output_features)
            )
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(pooling_output_size)
        self.dense_network = Sequential(
            nn.Linear(pooling_output_size * n_output_features, 72), nn.PReLU(), nn.BatchNorm1d(72), nn.Dropout(p=0.5),
            nn.Linear(72, 32), nn.PReLU(),
            nn.Linear(32, position_output_size)
        )
        # ========end-of-DEFAULT-PREDICTOR-FOR-COMPATIBILITY====================

        return

    def forward(self, input_seq):
        """
Classic forward method of every PyTorch model. However,
you are not expected to call this method in this model. We have a thread that
updates our position and returns it to with get_current_position().

        :param input_seq: Input sequence of the time series.
        :return: The prediction in the end of the series.
        """

        # As features (px, py, pz, qw, qx, qy, qz) sao os "canais" da
        # convolucao e precisam vir no meio para o pytorch
        input_seq = movedim(input_seq, -2, -1)

        input_seq = self.feature_extractor(input_seq)
        input_seq = self.adaptive_pooling(input_seq)

        predictions = self.dense_network(input_seq.view(input_seq.shape[0], -1))

        return predictions

    def load_feature_extractor(self, freeze_pretrained_model=True):
        """
Here you may load a pretrained InertialModule to make predictions. By default,
it freezes all InertialModule layers.

        :param freeze_pretrained_model: Whenever to freeze pretrained InertialModule layers. Default = True.
        :return: Void.
        """
        model = torch.load("best_model.pth")
        # model.load_state_dict(torch.load("best_model_state_dict.pth"))

        self.feature_extractor = model.feature_extractor
        # Aproveitamos para carregar tambem a camada densa de previsao de uma vez
        self.dense_network = model.dense_network

        if freeze_pretrained_model is True:
            # Congela todas as camadas do extrator para treinar apenas as camadas
            # seguintes
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

            # Estamos congelndo tambem a camada de previsao, apenas para aproveitar a chamada da funcao
            for param in self.dense_network.parameters():
                param.requires_grad = False

        return

    def get_current_position(self, ):
        """
Returns latest position update generated by this model. We also receive a
timestamp indicating the time that position was estimated.

        :return: Tuple with (Tensor (1x7) containing latest position update; Respective position time reference).
        """
        if self.predictions_buffer.shape[0] > 3 * self.sampling_window_size:
            self.predictions_buffer = \
                self.predictions_buffer[-2 * self.sampling_window_size:]

        self.new_predictions_arrived.clear()
        # Return the last position update and drop out the associated timestamp.
        return self.predictions_buffer[-1, 1:], self.predictions_buffer[-1, 0]

    def set_initial_position(self, position, timestamp=time.time()):
        """
    This Handler calculates position according to displacement from initial
    position. It gives you absolute position estimatives if the initial
    position is set according to your global reference.

        :param position: Starting position (px, py, pz, qw, qx, qy, qz).
        """

        position = torch.tensor(position, device=self.device,
                                dtype=self.dtype, requires_grad=False)

        timestamp = torch.tensor(timestamp, device=self.device,
                                 dtype=self.dtype, requires_grad=False)

        self.predictions_buffer[0, 1:] = position
        self.predictions_buffer[0, 0:1] = timestamp

        return

    def push_imu_sample_to_buffer(self, sample, timestamp):
        """
Stores a new IMU sample to the buffer, so it will be taken into account in next
predictions.

        :param sample: List or iterable containig 6 channels of IMU sample (ax, ay, az, wx, wq, wz).
        :return: Void.
        """

        if self.imu_buffer.shape[1] > self._imu_buffer_size_limit:
            self.imu_buffer = \
                self.imu_buffer[0:, -self._imu_buffer_reduction_threshold:, :]

        self.imu_buffer = \
            torch.cat((self.imu_buffer,
                       torch.tensor([timestamp] + list(sample),
                                    dtype=self.dtype, requires_grad=False,
                                    device=self.device).view(1, 1, -1)
                       ), dim=1)

        # Informs this thread that new samples have been generated.
        self._imu_samples_arrived.set()

        return

    # This class is not intended to be used as training for the InertialModule
    # first stage, so we disable grad for faster computation.
    @torch.no_grad()
    def run(self):

        # We only can start predicting positions after we have the
        # sampling_window_size IMU samples collected.
        while self.imu_buffer.shape[1] < self.sampling_window_size:
            time.sleep(1)

        # Put pytorch module into evaluation mode (batchnorm and dropout need).
        self.eval()

        # If this thread is not stopped, continue updating position.
        while self.stop_flag is False:
            # Wait for IMU new readings
            self._imu_samples_arrived.wait()

            # We check the timestamp of the last sampling_window_size reading and
            # search for the closest timestamp in prediction buffer.
            prediction_closest_timestamp, prediction_idx = \
                IMUHandler.find_nearest(
                    self.predictions_buffer[:, 0],
                    self.imu_buffer[0, -self.sampling_window_size, 0]
                )

            # We also need the prediction's closest reading, that may not be
            # the one at sampling_window_size before
            imu_closest_timestamp, imu_idx = \
                IMUHandler.find_nearest(
                    self.imu_buffer[0, :, 0],
                    prediction_closest_timestamp
                )

            # We need to now for what time we are calculating position.
            current_timestamp = self.imu_buffer[0:, -1, 0:1]

            # Registers that there are no more NEW imu readings to process.
            self._imu_samples_arrived.clear()

            self.predictions_buffer = \
                torch.cat((
                    self.predictions_buffer,
                    torch.cat((
                        current_timestamp,
                        self.predictions_buffer[prediction_idx, 1:] +
                        self(self.imu_buffer[0:, imu_idx:, 1:])
                    ), dim=1)
                ))

            self.new_predictions_arrived.set()

    @staticmethod
    def find_nearest(tensor_to_search, value):
        """
This method takes 1 tensor as first argument and a value to find the element
in array whose value is the closest. Returns the closest value element and its
index in the original array.

        :param tensor_to_search: Reference tensor.
        :param value: Value to find closest element.
        :return: Tuple (Element value, element index).
        """
        idx = (absolute(tensor_to_search - value)).argmin()
        return tensor_to_search[idx], idx


class _MoveDimModule(nn.Module):
    def __init__(self, source=-2, destination=-1):
        super().__init__()
        self.source = source
        self.destination = destination

    def forward(self, input_seq):
        return movedim(input_seq, self.source, self.destination)


class SumLayer(nn.Module):
    def __init__(self, ):
        """
    This layer aims to sum the last dimension of a tensor.
        """
        super(SumLayer, self).__init__()

    def forward(self, input_seq):
        return torch.sum(input_seq, dim=-1, keepdim=True)
