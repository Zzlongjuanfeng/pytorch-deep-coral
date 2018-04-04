"""Main script for ADDA."""

import os
import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, Classifier, ResNet34Encoder
from utils import get_data_loader, init_model, init_random_seed
from datasets.visda import get_visda

os.environ["CUDA_VISIBLE_DEVICES"] = '1,0'

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
    tgt_encoder = init_model(net=ResNet34Encoder(),
                             restore=params.tgt_encoder_restore)
    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)

    # train source model
    # print("=== Training classifier for source domain ===")
    # print(">>> Source Encoder <<<")
    # print(src_encoder)
    # print(">>> Source Classifier <<<")
    # print(src_classifier)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    # print("=== Training encoder for target domain ===")
    # print(">>> Target Encoder <<<")
    # print(tgt_encoder)
    # print(">>> Critic <<<")
    # print(critic)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)