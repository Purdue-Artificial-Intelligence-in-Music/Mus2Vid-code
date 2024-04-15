import unittest
import torch
import random
from models.CNN import *
from models.GInv_structures import *
from models.CNN import *
from models.GInv_Linear import *
from models.GInv_LSTM import *
from models.GInv_RNN import *
from models.LSTM import *
from models.RNN import *
from dataset_classes.DEAM_CQT_sliding import *

class TestModels(unittest.TestCase):
    def test_no_crash_on_init(self):
        input_size = 5
        hidden_size = 10
        num_layers = 3
        m = MLP(input_size=input_size, layer_sizes=[hidden_size], output_dim=hidden_size)
        m = RNN_model(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        m = LSTM_model(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        m = GInvariantMLP(input_size=input_size, layer_sizes=[hidden_size], output_dim=hidden_size)
        m = GInvariantRNN_Model(input_dim=input_size, hidden_dim=hidden_size)
        m = GInvariantLSTM_Model(input_dim=input_size, hidden_dim=hidden_size)
        m = GInvariantRNN_Model(input_dim=input_size, hidden_dim=hidden_size, num_layers=num_layers)
        m = GInvariantLSTM_Model(input_dim=input_size, hidden_dim=hidden_size, num_layers=num_layers)
        annot_path = "deam_dataset/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv"
        audio_path = "deam_dataset/DEAM_audio/MEMD_audio/"
        transform_path = "transforms/"
        transform_name = "cqt"
        train_dataset = DEAM_CQT_Dataset_Sliding(annot_path=annot_path, audio_path=audio_path, save_files=True, transform_path=transform_path, transform_name=transform_name, train=True)
        (data, target) = train_dataset.__getitem__(1)
        m = CNN_model(input_size = data.shape, hidden_size = 30, out_size = (data.shape[0], 1))

    def test_no_crash_on_run(self):
        annot_path = "deam_dataset/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv"
        audio_path = "deam_dataset/DEAM_audio/MEMD_audio/"
        transform_path = "transforms/"
        transform_name = "testing"
        d = DEAM_CQT_Dataset_Sliding(annot_path=annot_path, audio_path=audio_path, save_files=True, transform_path=transform_path, transform_name=transform_name, train=True)
        # Run each of the models, reshaping input as necessary and make sure they don't crash

    def test_GInv_MLP_invariance(self):
        # Test the GInv_MLP for frequency shift invariance
        model = GInvariantMLP(input_size=20, layer_sizes=[10], output_dim=5)
        for s in range(20):
            input_data = torch.randn((5, 20))
            output_basic = model.forward(input_data)
            for q in range(20):
                data = torch.roll(input_data, shifts=q, dims=1)
                output = model.forward(data)
                assert torch.allclose(output_basic, output, atol=1e-5)
        pass
            

    def test_GInv_RNN_invariance(self):
        # Test the GInv_RNN for frequency shift invariance
        model = GInvariantRNN_Model(input_dim=20, hidden_dim=25, num_layers=1)
        for s in range(20):
            input_data = torch.randn((5, 10, 20))
            output_basic = model.forward(input_data)
            for q in range(20):
                data = torch.roll(input_data, shifts=q, dims=2)
                output = model.forward(data)
                assert torch.allclose(output_basic, output, atol=1e-5)
        model = GInvariantRNN_Model(input_dim=20, hidden_dim=25, num_layers=3)
        for s in range(20):
            input_data = torch.randn((5, 10, 20))
            output_basic = model.forward(input_data)
            for q in range(20):
                data = torch.roll(input_data, shifts=q, dims=2)
                output = model.forward(data)
                assert torch.allclose(output_basic, output, atol=1e-5)
        pass

    def test_GInv_LSTM_invariance(self):
        # Test the GInv_LSTM for frequency shift invariance
        model = GInvariantLSTM_Model(input_dim=20, hidden_dim=25, num_layers=1)
        for s in range(20):
            input_data = torch.randn((5, 10, 20))
            output_basic = model.forward(input_data)
            for q in range(20):
                data = torch.roll(input_data, shifts=q, dims=2)
                output = model.forward(data)
                assert torch.allclose(output_basic, output, atol=1e-5)
        model = GInvariantLSTM_Model(input_dim=20, hidden_dim=25, num_layers=3)
        for s in range(20):
            input_data = torch.randn((5, 10, 20))
            output_basic = model.forward(input_data)
            for q in range(20):
                data = torch.roll(input_data, shifts=q, dims=2)
                output = model.forward(data)
                assert torch.allclose(output_basic, output, atol=1e-5)
        pass


if __name__ == '__main__':
    unittest.main()