import sys
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import anyconfig
sys.path.append("..")
import  torch.nn.functional as F
from data.datalorder import GarbageData
from data.datalorderS import GarbageData2
from model.SwimTransformer import *
from model.main import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings
import json
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')
with open("D:/dataset_garb/jupyter_code/name_to_id48.json",'r') as load_f:
    name_to_id = json.load(load_f)
with open("D:/dataset_garb/jupyter_code/id_to_name48.json",'r') as load_f:
    id_to_name = json.load(load_f)
with open("D:/dataset_garb/jupyter_code/detail_to_big48.json",'r') as load_f:
    detail_to_big = json.load(load_f)
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_train_data():
    root = "D:/dataset_garb/"  # 根目录

    cfg_path = os.path.join(root + "code/config/cfg.yaml")
    assert (os.path.exists(cfg_path))
    config = anyconfig.load(open(cfg_path, 'rb'))
    train_csv_path = config['train_csv_path']
    val_csv_path = config['val_csv_path']

    train_dataset = GarbageData2(csv_path_train=train_csv_path, mode='train')
    val_dataset = GarbageData2(csv_path_val=val_csv_path, mode='valid')
    print(len(val_dataset))
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=28,
            sampler=train_dataset.weight_sampler,
            shuffle=False,
            num_workers=0
        )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    return train_loader,val_loader
def load_train_only():
    root = "D:/dataset_garb/"  # 根目录

    cfg_path = os.path.join(root + "code/config/cfg.yaml")
    assert (os.path.exists(cfg_path))
    config = anyconfig.load(open(cfg_path, 'rb'))
    train_csv_path = config['train_csv_path']
    val_csv_path = config['val_csv_path']

    train_dataset = GarbageData2(csv_path_train=train_csv_path, mode='train')
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
        sampler=train_dataset.weight_sampler,
        shuffle=False,
        num_workers=0
    )
    return train_loader
def map_to_big(logits):

    big_logits=torch.ones(logits.shape,dtype=torch.int64)
    for index,item in enumerate(logits.cpu()):
        big_logits[index]=detail_to_big[str(item.item())]
    return big_logits
def get_config():
    root = "D:/dataset_garb/"  # 根目录

    cfg_path = os.path.join(root + "code/config/cfg.yaml")
    assert (os.path.exists(cfg_path))
    config = anyconfig.load(open(cfg_path, 'rb'))
    return config
def get_root():
    return "D:/dataset_garb/"
def val(currentnum):
    train_loader, val_loader = load_train_data()
    num_classes = 48
    my_model = get_res_model_my(num_classes=num_classes,res_layer=50,model_path="D:/dataset_garb/code/weight/gabtest3_50_withval_cbamAA_au3.pth")
    device = get_device()
    my_model=my_model.to(device)
    my_model.device = device
    cfg=get_config()
    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    my_model.eval()
    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []
    val_label = []

    val_biglabel=[]
    val_bigpredict=[]




    val_predict = []
    valid_accs_big=[]
    valid_accs_big2=[]
    logits_list=[]
    # Iterate the validation set by batches.
    for batch in tqdm(val_loader):
        imgs, labels, big_labels,path_ = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            _, _, _, _, logits,big_logits = my_model(imgs.to(device))
        logits_list.append(torch.squeeze(logits,dim=0).cpu())
        big_labels_logits=map_to_big(logits.argmax(dim=-1) )
        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        acc2=(big_labels_logits == big_labels).float().mean()
        acc2_another = (big_logits.argmax(dim=-1) == big_labels.to(device)).float().mean()
        path_=np.asarray(path_)
        select=np.asarray(torch.tensor(logits.argmax(dim=-1) != labels.to(device)).cpu())

        with open("D:/dataset_garb/code/log/error2.txt",'a') as f:
            for item in path_[select]:
                f.write(item+"\n")
        val_label.append(labels)
        val_biglabel.append(big_labels)

        val_predict.append(logits.argmax(dim=-1))
        val_bigpredict.append(big_labels_logits)

        # Record the loss and accuracy.
        valid_accs.append(acc)
        valid_accs_big.append(acc2)
        valid_accs_big2.append(acc2_another)


    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_acc_big=sum(valid_accs_big) / len(valid_accs_big)
    valid_acc_big2 = sum(valid_accs_big2) / len(valid_accs_big2)
    valid_acc = sum(valid_accs) / len(valid_accs)
    all_label = torch.cat(val_label, -1)
    all_predict = torch.cat(val_predict, -1)

    all_biglabel=torch.cat(val_biglabel,-1)
    all_bigpredict=torch.cat(val_bigpredict,-1)

    all_predict = all_predict.detach().cpu().numpy()
    all_label = all_label.detach().cpu().numpy()

    all_biglabel=all_biglabel.detach().cpu().numpy()
    all_bigpredict=all_bigpredict.detach().cpu().numpy()

    the_label = list(range(48))
    the_label2=list(range(4))


    report = classification_report(all_label, all_predict, labels=the_label, output_dict=True)
    matrix = confusion_matrix(all_label, all_predict)

    report2 = classification_report(all_biglabel, all_bigpredict, labels=the_label2, output_dict=True)
    matrix2 = confusion_matrix(all_biglabel,all_bigpredict)


    print(report)
    print(matrix)

    print(report2)
    print(matrix2)

    matrix=np.asarray(matrix)
    # logits_list=np.asarray(torch.stack(logits_list).numpy(),dtype=np.float32)
    np.savetxt("D:/dataset_garb/code/log/matrix{}.txt".format(str(currentnum)),matrix,fmt="%d",delimiter=",")
    # np.savetxt("D:/dataset_garb/code/log/logits{}.txt".format(str(currentnum)),  logits_list, fmt="%f", delimiter=",")
    with open("D:/dataset_garb/code/log/report{}.json".format(str(currentnum)),'w') as f:
        json.dump(report,f)
        with open("D:/dataset_garb/code/log/reportbig{}.json".format(str(currentnum)), 'w') as f:
            json.dump(report2, f)
    # Print the information.
    print(f"[ Valid | acc = {valid_acc:.5f}")
    print(f"[ Valid | acc_bigclass = {valid_acc_big:.5f}")
    print(f"[ Valid | acc_bigclass2 = {valid_acc_big2:.5f}")


def train(currenttrainid,valid = False):

    if(valid==False):
        train_loader=load_train_only()
    else:
        train_loader,val_loader=load_train_data()
    num_classes=48
    # my_model=get_st(num_classes) #swin transformer
    reslayer=50
    my_model=res_model_pretrain(num_classes=num_classes,res_layer=reslayer)
#    my_model_teacher = get_res_model_my(num_classes=num_classes,res_layer=50,model_path="D:/dataset_garb/code/weight/gabtest3_50_withval_withbigloss.pth")
    device = get_device()
    my_model=my_model.to(device)
    # my_model_teacher=my_model_teacher.to(device)
    my_model.device = device
    cfg=get_config()
    save_model_path=get_root()+cfg["model_path"]

    #分类问题选择交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    #初始化优化器
    learning_rate=float(cfg['l_r'])
    weight_decay=float(cfg['weight_decay'])
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # The number of training epochs.
    n_epochs = cfg['num_epoch']


    best_acc = 0.0
    for epoch in range(n_epochs):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        my_model.train()
        # my_model_teacher.train()
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        train_label=[]
        train_predict =[]
        # Iterate the training set by batches.
        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels,big_labels,imgpath,train_logits = batch
            imgs=imgs['image']
            imgs = imgs.to(device)
            labels = labels.to(device)
            big_labels=big_labels.to(device)
            train_label.append(labels)
            # Forward the data. (Make sure data and model are on the same device.)
            _,_,_,_,logits,big_logits = my_model(imgs)

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = 0.5*criterion(logits, labels)+0.5*criterion(big_logits,big_labels)
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
            # Compute the gradients for parameters.
            loss.backward()
            # Update the parameters with computed gradients.
            optimizer.step()
            predict=logits.argmax(dim=-1)
            train_predict.append(predict)

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels).float().mean()


            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        # The average loss and accuracy of the training set is the average of the recorded values.

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)



        with open("D:/dataset_garb/code/log/train_loss{}.txt".format(currenttrainid),'a') as f:
            f.write(str(train_loss)+"\n")
        with open("D:/dataset_garb/code/log/train_acc{}.txt".format(currenttrainid),'a') as f:
            f.write(str(train_acc)+"\n")


        train_label=tuple(train_label)
        train_predict=tuple(train_predict)
        all_label=torch.cat(train_label,-1)
        all_predict=torch.cat(train_predict,-1)
        all_predict=all_predict.detach().cpu().numpy()
        all_label = all_label.detach().cpu().numpy()
        the_label=list(range(num_classes))

        report = classification_report(all_label, all_predict, labels=the_label, output_dict=True)
        matrix = confusion_matrix(all_label, all_predict)
        print(report)
        print(matrix)
        # print(all_label.shape)
        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # if train_acc > best_acc:
        #     best_acc = train_acc
        #     torch.save(my_model.state_dict(), save_model_path)
        #     print('saving model with acc {:.3f}'.format(best_acc))
        if(valid==True):

            # ---------- Validation ----------
            # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
            my_model.eval()
            # These are used to record information in validation.
            valid_loss = []
            valid_accs = []
            val_label=[]
            val_predict=[]
            valid_accs_big = []

            # Iterate the validation set by batches.
            for batch in tqdm(val_loader):
                imgs, labels ,big_labels,path_= batch

                # We don't need gradient in validation.
                # Using torch.no_grad() accelerates the forward process.
                with torch.no_grad():
                    _,_,_,_,logits,big_logits = my_model(imgs.to(device))
                big_labels_logits = map_to_big(logits.argmax(dim=-1))
                # We can still compute the loss (but not the gradient).
                loss = criterion(logits, labels.to(device))

                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
                acc2 = (big_labels_logits == big_labels).float().mean()
                val_label.append(labels)
                val_predict.append(logits.argmax(dim=-1))
                # Record the loss and accuracy.
                valid_loss.append(loss.item())
                valid_accs.append(acc)
                valid_accs_big.append(acc2)

            # The average loss and accuracy for entire validation set is the average of the recorded values.
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)
            valid_acc_big = sum(valid_accs_big) / len(valid_accs_big)
            with open("D:/dataset_garb/code/log/val_loss{}.txt".format(currenttrainid), 'a') as f:
                f.write(str(valid_loss) + "\n")
            with open("D:/dataset_garb/code/log/val_acc{}.txt".format(currenttrainid), 'a') as f:
                f.write(str(valid_acc) + "\n")

            all_label = torch.cat(val_label, -1)
            all_predict = torch.cat(val_predict, -1)
            all_predict = all_predict.detach().cpu().numpy()
            all_label = all_label.detach().cpu().numpy()
            the_label=list(range(num_classes))
            report = classification_report(all_label, all_predict, labels=the_label, output_dict=True)
            matrix = confusion_matrix(all_label, all_predict)
            print(report)
            print(matrix)
            # Print the information.
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}, acc_bigclass = {valid_acc_big:.5f}")


            # if the model improves, save a checkpoint at this epoch
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(my_model.state_dict(),save_model_path)
                print('saving model with acc {:.3f}'.format(best_acc))

    pass
if __name__ == '__main__':
    """
    test
    """
    load_train_data()
    print(get_device())
    cfg=get_config()
    save_model_path = get_root()+cfg["model_path"]
    print(save_model_path)
    assert save_model_path
    val(10)