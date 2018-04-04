"""Main script for ADDA."""

import os
import params
from core import eval_src, train_src
from models import Discriminator, Classifier, ResNet34Encoder
from utils import get_data_loader, init_model, init_random_seed
from datasets.visda import get_visda

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader = get_visda(root=params.data_root, sub_dir='train', split='train')
    src_data_loader_eval = get_visda(root=params.data_root, sub_dir='train', split='test')

    tgt_data_loader = get_visda(root=params.data_root, sub_dir='validation', split='train')
    tgt_data_loader_eval = get_visda(root=params.data_root, sub_dir='validation', split='test')

    # load models
    src_encoder = init_model(net=ResNet34Encoder(),
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=Classifier(),
                                restore=params.src_classifier_restore)

    # train source model
    # print("=== Training classifier for source domain ===")
    # print(">>> Source Encoder <<<")
    # print(src_encoder)
    # print(">>> Source Classifier <<<")
    # print(src_classifier)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    # print(">>> domain adaption <<<")
    eval_src(src_encoder, src_classifier, tgt_data_loader_eval)