import torch
from dataset_classes.DEAM_CQT import DEAM_CQT_Dataset
from dataset_classes.DEAM_CQT_sliding import DEAM_CQT_Dataset_Sliding
from models.GInv_LSTM import GInvariantLSTM_Model
from models.LSTM import LSTM_model
from models.CNN import CNN_model
import copy
import librosa
import numpy as np
import matplotlib.pyplot as plt

N_EPOCHS = 100
BATCH_SIZE = 30
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HIDDEN_SIZE = 30
NUM_LAYERS = 3
ANNOT_PATH = "deam_dataset/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv"
AUDIO_PATH = "deam_dataset/DEAM_audio/MEMD_audio/"
TRANSFORM_PATH = "transforms/"
TRANSFORM_NAME = "cqt"
LEARNING_RATE = 1e-5    # decrease
DROPOUT = 0.3
MAX_NONDEC_EPOCHS = 40

def chroma_cqt(y, sr, hop_length):
    return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

def cqt(y, sr, hop_length):
    return np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length))

TRANSFORM_FUNC = cqt
DATASET_CLASS = DEAM_CQT_Dataset_Sliding
DATASET_NAME = "sliding"

def evaluate(model, test_loader, criterion, device):
    model.eval()
    avg_loss = 0.0
    num_vals = float(len(test_loader))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = torch.squeeze(model(data))        
            avg_loss += criterion(output, target) / num_vals
    return avg_loss

def train(model,
         *,
         train_loader,
         test_loader,
         optimizer,
         criterion,
         scheduler,
         num_epochs,
         device):
    
    best_model = None
    best_acc = 0.0
    nondecreasing_acc = 0
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        model.train()
        n = 0
        loss_sum = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = torch.squeeze(model(data))        
            loss = criterion(output, target)
            with torch.no_grad():
                loss_sum += loss
                n += 1
            loss.backward()
            optimizer.step()

        loss_sum /= n
        scheduler.step()
        accuracy = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train loss: {loss_sum:.4f}, Test loss: {accuracy:.4f}')
        train_loss.append(loss_sum.to("cpu").item())
        test_loss.append(accuracy.to("cpu"))
        

        if best_model is None or accuracy < best_acc:
            best_acc = accuracy
            best_model = copy.deepcopy(model)
            nondecreasing_acc = 0
        else:
            nondecreasing_acc += 1 

        if nondecreasing_acc > MAX_NONDEC_EPOCHS:
            break
        

    print('Training finished!')

    return best_model, train_loss, test_loss


def main():
    train_dataset = DATASET_CLASS(annot_path=ANNOT_PATH, audio_path=AUDIO_PATH, save_files=True, transform_path=TRANSFORM_PATH, transform_name=TRANSFORM_NAME, transform_func=TRANSFORM_FUNC, train=True)
    test_dataset = DATASET_CLASS(annot_path=ANNOT_PATH, audio_path=AUDIO_PATH, save_files=True, transform_path=TRANSFORM_PATH, transform_name=TRANSFORM_NAME, transform_func=TRANSFORM_FUNC, train=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_size = train_dataset.__getitem__(0)[0].shape[1]
    (data, target) = train_dataset.__getitem__(320)
    model = CNN_model(input_size = data.shape, hidden_size = 30, out_size = (data.shape[0], 1))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
    #     torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.pow(0.9, 0.2)),
    #     torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01),
    #     torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # ], milestones=[10, 11])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95) # increase gamma

    model = model.to(DEVICE)

    best_model, train_loss, test_loss = train(model,
          train_loader=train_loader,
          test_loader=test_loader,
          criterion=criterion,
          optimizer=optimizer,

          num_epochs=N_EPOCHS,
          device=DEVICE,
          scheduler=scheduler)

    print("Evaluating on test set")
    accuracy = evaluate(best_model, test_loader, criterion, DEVICE)
    print(f'Final test set loss: {accuracy:.4f}')

    (chroma, annots) = test_dataset.__getitem__(0)
    print(model(chroma))
    print(annots)

    torch.save(best_model, TRANSFORM_NAME+"_"+DATASET_NAME+"_model.pt")

    vals = np.arange(1, len(train_loss)+1)
    plt.plot(vals, train_loss, 'r')
    plt.plot(vals, test_loss, 'g')
    plt.title("Training with transform="+TRANSFORM_NAME+" and dataset="+DATASET_NAME)
    plt.xlabel("Epoch #")
    plt.savefig(TRANSFORM_NAME+"_"+DATASET_NAME+".png")


if __name__ == "__main__":
    main()