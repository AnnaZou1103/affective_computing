from utils.get_data import get_dataloader
import torch
from torch import nn
from model.common_model import MLP, MMDL, Identity
from model.common_fusion import Concat
from model.BiLSTM import BiLSTM
from timm.models.swin_transformer_v2 import swinv2_tiny_window16_256
from utils.performance import AUPRC, f1_score, accuracy, eval_affect
import time

softmax = nn.Softmax()

def getallparams(li):
    params = 0
    for module in li:
        for param in module.parameters():
            params += param.numel()
    return params

def deal_with_objective(objective, pred, truth, args):
    """Alter inputs depending on objective function, to deal with different objective arguments."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if type(objective) == nn.CrossEntropyLoss:
        if len(truth.size()) == len(pred.size()):
            truth1 = truth.squeeze(len(pred.size())-1)
        else:
            truth1 = truth
        return objective(pred, truth1.long().to(device))
    elif type(objective) == nn.MSELoss or type(objective) == nn.modules.loss.BCEWithLogitsLoss or type(objective) == nn.L1Loss:
        return objective(pred, truth.float().to(device))
    else:
        return objective(pred, truth, args)

def train(
        encoders, fusion, head, train_dataloader, valid_dataloader, total_epochs, additional_optimizing_modules=[],
        early_stop=False, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None,
        input_to_float=True, clip_val=8,):
    """
    Handle running a simple supervised training loop.

    :param encoders: list of modules, unimodal encoders for each input modality in the order of the modality input data.
    :param fusion: fusion module, takes in outputs of encoders in a list and outputs fused representation
    :param head: classification or prediction head, takes in output of fusion module and outputs the classification or prediction results that will be sent to the objective function for loss calculation
    :param total_epochs: maximum number of epochs to train
    :param additional_optimizing_modules: list of modules, include all modules that you want to be optimized by the optimizer other than those in encoders, fusion, head (for example, decoders in MVAE)
    :param early_stop: whether to stop early if valid performance does not improve over 7 epochs
    :param task: type of task, currently support "classification","regression","multilabel"
    :param optimtype: type of optimizer to use
    :param lr: learning rate
    :param weight_decay: weight decay of optimizer
    :param objective: objective function, which is either one of CrossEntropyLoss, MSELoss or BCEWithLogitsLoss or a custom objective function that takes in three arguments: prediction, ground truth, and an argument dictionary.
    :param auprc: whether to compute auprc score or not
    :param save: the name of the saved file for the model with current best validation performance
    :param validtime: whether to show valid time in seconds or not
    :param objective_args_dict: the argument dictionary to be passed into objective function. If not None, at every batch the dict's "reps", "fused", "inputs", "training" fields will be updated to the batch's encoder outputs, fusion module output, input tensors, and boolean of whether this is training or validation, respectively.
    :param input_to_float: whether to convert input to float type or not
    :param clip_val: grad clipping limit
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MMDL(encoders, fusion, head).to(device)

    additional_params = []
    for m in additional_optimizing_modules:
        additional_params.extend(
            [p for p in m.parameters() if p.requires_grad])
    op = optimtype([p for p in model.parameters() if p.requires_grad] +
                    additional_params, lr=lr, weight_decay=weight_decay)
    bestvalloss = 10000
    bestacc = 0
    bestf1 = 0
    patience = 0

    def _processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp

    for epoch in range(total_epochs):
        totalloss = 0.0
        totals = 0
        model.train()
        for j in train_dataloader:
            op.zero_grad()
            model.train()
            out = model([_processinput(i).to(device) for i in j[:-1]])
            if not (objective_args_dict is None):
                objective_args_dict['reps'] = model.reps
                objective_args_dict['fused'] = model.fuseout
                objective_args_dict['inputs'] = j[:-1]
                objective_args_dict['training'] = True
                objective_args_dict['model'] = model
            loss = deal_with_objective(
                objective, out, j[-1], objective_args_dict)

            totalloss += loss * len(j[-1])
            totals += len(j[-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            op.step()
        print("Epoch " + str(epoch) + " train loss: " + str(totalloss / totals))

        validstarttime = time.time()
        if validtime:
            print("train total: " + str(totals))
        model.eval()
        with torch.no_grad():
            totalloss = 0.0
            pred = []
            true = []
            pts = []

            for j in valid_dataloader:
                model.train()
                out = model([_processinput(i).to(device)
                              for i in j[:-1]])

                if not (objective_args_dict is None):
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = False
                loss = deal_with_objective(
                    objective, out, j[-1], objective_args_dict)
                totalloss += loss * len(j[-1])

                pred.append(torch.argmax(out, dim=1))
                true.append(j[-1])
                if auprc:
                    sm = softmax(out)
                    pts += [(sm[i][1].item(), j[-1][i].item())
                            for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0)
        true = torch.cat(true, 0)
        totals = true.shape[0]
        valloss = totalloss / totals

        acc = accuracy(true, pred)
        print("Epoch " + str(epoch) + " valid loss: " + str(valloss) +
              " acc: " + str(acc))
        if acc > bestacc:
            patience = 0
            bestacc = acc
            print("Saving Best")
            torch.save(model, save)
        else:
            patience += 1

        if early_stop and patience > 7:
            break
        if auprc:
            print("AUPRC: " + str(AUPRC(pts)))
        validendtime = time.time()
        if validtime:
            print("valid time:  " + str(validendtime - validstarttime))
            print("Valid total: " + str(totals))

def single_test(
        model, test_dataloader, is_packed=False,
        criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True):
    """Run single test for model.
    Args:
        model (nn.Module): Model to test
        test_dataloader (torch.utils.data.Dataloader): Test dataloader
        is_packed (bool, optional): Whether the input data is packed or not. Defaults to False.
        criterion (_type_, optional): Loss function. Defaults to nn.CrossEntropyLoss().
        task (str, optional): Task to evaluate. Choose between "classification", "multiclass", "regression", "posneg-classification". Defaults to "classification".
        auprc (bool, optional): Whether to get AUPRC scores or not. Defaults to False.
        input_to_float (bool, optional): Whether to convert inputs to float before processing. Defaults to True.
    """

    def _processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp

    with torch.no_grad():
        totalloss = 0.0
        pred = []
        true = []
        pts = []
        for j in test_dataloader:
            model.eval()
            if is_packed:
                out = model([[_processinput(i).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                              for i in j[0]], j[1]])
            else:
                out = model([_processinput(i).float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                             for i in j[:-1]])
            if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss or type(criterion) == torch.nn.MSELoss:
                loss = criterion(out, j[-1].float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

            # elif type(criterion) == torch.nn.CrossEntropyLoss:
            #     loss=criterion(out, j[-1].long().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

            elif type(criterion) == nn.CrossEntropyLoss:
                if len(j[-1].size()) == len(out.size()):
                    truth1 = j[-1].squeeze(len(out.size()) - 1)
                else:
                    truth1 = j[-1]
                loss = criterion(out, truth1.long().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            else:
                loss = criterion(out, j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            totalloss += loss * len(j[-1])
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            elif task == "posneg-classification":
                prede = []
                oute = out.cpu().numpy().tolist()
                for i in oute:
                    if i[0] > 0:
                        prede.append(1)
                    elif i[0] < 0:
                        prede.append(-1)
                    else:
                        prede.append(0)
                pred.append(torch.LongTensor(prede))
            true.append(j[-1])
            if auprc:
                # pdb.set_trace()
                sm = softmax(out)
                pts += [(sm[i][1].item(), j[-1][i].item())
                        for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0)
        true = torch.cat(true, 0)
        totals = true.shape[0]
        testloss = totalloss / totals
        if auprc:
            print("AUPRC: " + str(AUPRC(pts)))
        if task == "classification":
            print("acc: " + str(accuracy(true, pred)))
            return {'Accuracy': accuracy(true, pred)}
        elif task == "multilabel":
            print(" f1_micro: " + str(f1_score(true, pred, average="micro")) +
                  " f1_macro: " + str(f1_score(true, pred, average="macro")))
            return {'micro': f1_score(true, pred, average="micro"), 'macro': f1_score(true, pred, average="macro")}
        elif task == "regression":
            print("mse: " + str(testloss.item()))
            return {'MSE': testloss.item()}
        elif task == "posneg-classification":
            trueposneg = true
            accs = eval_affect(trueposneg, pred)
            acc2 = eval_affect(trueposneg, pred, exclude_zero=False)
            print("acc: " + str(accs) + ', ' + str(acc2))
            return {'Accuracy': accs}

if __name__ == '__main__':
    audio_input_dim = 512
    HIDDEN_1 = 256
    HIDDEN_2 = 128
    mlp_em_dim = 256
    class_num = 6
    model_save_path = 'checkpoints/multimodal/best.pt'

    EPOCHS = 20
    model_lr = 1e-4
    weight_decay = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloader('output')

    audio_model = BiLSTM(input_size=audio_input_dim, hidden_1=HIDDEN_1, hidden_2=HIDDEN_2)
    image_model = swinv2_tiny_window16_256(pretrained=True)
    em_dim = HIDDEN_2 + image_model.head.in_features
    image_model.head = Identity()

    encoders = [audio_model.to(device), image_model.to(device)]

    head = MLP(em_dim, mlp_em_dim, class_num).to(device)

    fusion = Concat().to(device)

    train(encoders, fusion, head, train_loader, val_loader, EPOCHS, task="classification", optimtype=torch.optim.AdamW,
          early_stop=True, lr=model_lr, save=model_save_path, weight_decay=weight_decay,
          objective=torch.nn.CrossEntropyLoss())