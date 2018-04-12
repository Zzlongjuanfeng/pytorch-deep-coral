"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim
import torch

import params
import os
import time
from utils import make_variable, save_model, cal_confusion_mat
from logger import Logger



def train_src(encoder, classifier, src_data_loader, tgt_data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        [{'params': encoder.parameters(), 'lr': 5e-5},
         {'params': classifier.parameters()}],
        lr=params.learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))
    logger = Logger(os.path.join(params.model_root, 'enevts'))
    for epoch in range(params.num_epochs_pre):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, labels_src), (images_tgt, _)) in data_zip:
            t0 = time.time()
            # make images and labels variable
            total_step = epoch * len_data_loader + step + 1

            images_src = make_variable(images_src)
            labels_src = make_variable(labels_src.squeeze_())
            images_tgt = make_variable(images_tgt)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for classification and CORAL
            feat_src, preds_src = classifier(encoder(images_src))
            feat_tgt, preds_tgt = classifier(encoder(images_tgt))

            class_loss = criterion(preds_src, labels_src)
            CORAL_loss_fc2 = CORAL(preds_src, preds_tgt)
            CORAL_loss_fc1 = CORAL(feat_src, feat_tgt)
            CORAL_loss = (CORAL_loss_fc1 + CORAL_loss_fc2) / 2.0
            DAN_loss_fc2 = DAN(preds_src, preds_tgt)
            DAN_loss_fc1 = DAN(feat_src, feat_tgt)
            DAN_loss = (DAN_loss_fc2 + DAN_loss_fc1) / 2.0

            # loss = class_loss + CORAL_loss * params.CORAL_weight
            loss = class_loss + DAN_loss * params.DAN_weight
            # optimize source classifier
            loss.backward()
            optimizer.step()
            t1 = time.time()

            batch_time = t1 - t0
            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: class_loss={:.6f} CORAL_loss={:.6f} "
                      "DAN_loss={:.6f} loss={:.6f} batch time={:.5f}"
                      .format(epoch + 1, params.num_epochs_pre,
                              step + 1, len_data_loader,
                              class_loss.data[0], CORAL_loss.data[0],
                              DAN_loss.data[0],
                              loss.data[0], batch_time))
                info = {"train_loss": loss.data[0],
                        'class_loss': class_loss.data[0],
                        'coral_loss': CORAL_loss.data[0],
                        'DAN_loss': DAN_loss.data[0]}
                for tag, value in info.items():
                    logger.scalar_summary(tag=tag, value=value, step=total_step)


        # eval model on train set
        if ((epoch + 1) % params.eval_epoch_pre == 0):
            loss, acc = eval_src(encoder, classifier, tgt_data_loader)
            encoder.train()
            classifier.train()
            logger.scalar_summary(tag='acc', value=acc, step=total_step)

        # save model parameters
        if ((epoch + 1) % params.save_epoch_pre == 0):
            save_model(encoder, "CORAL-source-encoder-{}.pt".format(epoch + 1))
            save_model(
                classifier, "CORAL-source-classifier-{}.pt".format(epoch + 1))

    # save final model
    save_model(encoder, "CORAL-source-encoder-final.pt")
    save_model(classifier, "CORAL-source-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0
    con_matrix = torch.zeros((params.classes, params.classes))
    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels)

        _, preds = classifier(encoder(images))
        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()
        con_matrix += cal_confusion_mat(pred=pred_cls, label=labels.data)

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    pred_ = torch.sum(con_matrix, dim=1)
    label_ = torch.sum(con_matrix, dim=0)
    for i in range(con_matrix.shape[0]):
        pred_[i] = con_matrix[i, i] / float(pred_[i])
        label_[i] = con_matrix[i, i] / float(label_[i])

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    print(con_matrix)
    print("pred recall:{}".format(pred_.unsqueeze(dim=1).t()))
    print("class accuracy:{}".format(label_.unsqueeze(dim=1).t()))
    return loss, acc


def CORAL(source, target):
    d = source.data.shape[1]
    n_src = float(source.data.shape[0])
    n_tgt = float(target.data.shape[0])
    # source covariance
    xm = source - torch.mean(source, 0, keepdim=True)
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm) / (n_src - 1)

    # target covariance
    xmt = target - torch.mean(target, 0, keepdim=True)
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt) / (n_tgt - 1)

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)

    return loss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val) #/len(kernel_val)


def DAN(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)