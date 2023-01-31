import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim import lr_scheduler
import copy
import math
import pickle
from model.BiLSTM import BiLSTM


def feature_label(dataset):
    all_dataset = []
    for i in range(len(dataset["features"])):
        all_dataset.append((dataset["features"][i], dataset["labels"][i]))
    return all_dataset


def sort_data(data_set):
    indices = sorted(range(len(data_set)),
                     key=lambda k: len(data_set[k][0][0]),
                     reverse=True)
    data_set = [data_set[i] for i in indices]
    return data_set, indices


def split_data(all_dataset):
    data_size = len(all_dataset)
    dataset_sizes = {
        "train": int(data_size * 0.8),
        "test": int(data_size * 0.2),
    }

    train_set, test_set = data.random_split(all_dataset, [dataset_sizes["train"], dataset_sizes["test"]])

    sorted_train, train_indices = sort_data(train_set)
    sorted_test, test_indices = sort_data(test_set)

    def pad_tensor(vec, pad, dim):
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size).cuda()], dim=dim)

    def collate_fn(instances):
        max_len = max(map(lambda x: x[0].shape[1], instances))
        batch = []
        for (x, y) in instances:
            batch.append((pad_tensor(x, pad=max_len, dim=1), y))

        f = list(map(lambda x: x[0], batch))
        l = list(map(lambda x: x[1], batch))
        features = torch.stack(f, dim=0)
        labels = torch.Tensor(l)
        return (features, labels)

    dataloaders = {
        "train": torch.utils.data.DataLoader(sorted_train, batch_size=BATCHSIZE, num_workers=0, drop_last=True,
                                             collate_fn=collate_fn),
        "test": torch.utils.data.DataLoader(sorted_test, batch_size=BATCHSIZE, num_workers=0, drop_last=True,
                                            collate_fn=collate_fn), }

    return dataloaders, dataset_sizes


def train(model, dataloaders, dataset_sizes, num_epochs, checkpoint=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, 10)

    outputlist = {'train': {'loss': [], 'acc': []}, 'test': {'loss': [], 'acc': []}}

    if checkpoint is None:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = math.inf
        best_acc = 0.0
    else:
        print(
            f'Test loss: {checkpoint["best_test_loss"]}, Test accuracy: {checkpoint["best_test_accuracy"]}')
        model.load_state_dict(checkpoint['model_state_dict'])
        best_model_wts = copy.deepcopy(model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_loss = checkpoint['best_test_loss']
        best_acc = checkpoint['best_test_accuracy']

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('--' * 10)

        for phase in ["train", "test"]:
            running_loss = 0.0
            running_corrects = 0

            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # [20, t, 512]
                inputs = inputs.permute(0, 2, 1)
                inputs = inputs.to(device)
                labels = torch.tensor([i.long() for i in labels])
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward propogation and optimize in 'train' mode
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            # record the loss and accuracy for visualization
            outputlist[phase]['loss'].append(epoch_loss)
            # outputlist[phase]['acc'].append(epoch_acc.cpu())
            outputlist[phase]['acc'].append(epoch_acc)

            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # output test result
            if phase == "test" and epoch_acc > best_acc:
                print(f'New best model found!')
                print(f'New record accuracy: {epoch_acc}, Previous record accuracy: {best_acc}')
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                if epoch_acc > 0.50:
                    check_point_path = OUTPUT + "epoch" + str(epoch) + ".pth"

                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_test_loss': best_loss,
                                'best_test_accuracy': best_acc,
                                'scheduler_state_dict': scheduler.state_dict(),
                                }, check_point_path)

    print('Best test Accuracy: {:.4f} Best test loss: {:.4f}'.format(best_acc, best_loss))

    # load and return the best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss, best_acc, outputlist


if __name__ == '__main__':
    INPUT = '../output/audio_feature.pkl'
    OUTPUT = '../checkpoints/bilstm/'
    NUM_CLASSES = 4
    BATCHSIZE = 20
    EPOCHS = 40

    INPUT_DIM = 512
    HIDDEN_1 = 256
    HIDDEN_2 = 128
    HIDDEN_3 = 64
    DROPOUT = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))
    print(torch.cuda.get_device_name(0))

    torch.manual_seed(101)

    dataset = pickle.load(open(INPUT, 'rb'))
    # print(len(dataset["features"]))

    all_dataset = feature_label(dataset)

    dataloaders, dataset_sizes = split_data(all_dataset)
    print('Training set:', dataset_sizes["train"], 'Testing set:', dataset_sizes["test"])

    model = BiLSTM(
        input_size=INPUT_DIM,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        out_size=NUM_CLASSES,
        device = device
    )

    best_model, best_test_loss, best_test_acc, outputlist = train(model, dataloaders, dataset_sizes, EPOCHS)
    torch.save(best_model.module, OUTPUT)