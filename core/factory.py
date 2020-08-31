# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from paddlerec.core.utils import envs

trainer_abs = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "trainers")
trainers = {}


def trainer_registry():
    trainers["SingleTrainer"] = os.path.join(trainer_abs, "single_trainer.py")
    trainers["ClusterTrainer"] = os.path.join(trainer_abs,
                                              "cluster_trainer.py")
    trainers["CtrCodingTrainer"] = os.path.join(trainer_abs,
                                                "ctr_coding_trainer.py")
    trainers["CtrModulTrainer"] = os.path.join(trainer_abs,
                                               "ctr_modul_trainer.py")
    trainers["TDMSingleTrainer"] = os.path.join(trainer_abs,
                                                "tdm_single_trainer.py")
    trainers["TDMClusterTrainer"] = os.path.join(trainer_abs,
                                                 "tdm_cluster_trainer.py")
    trainers["OnlineLearningTrainer"] = os.path.join(
        trainer_abs, "online_learning_trainer.py")
    # Definition of procedure execution process
    trainers["CtrCodingTrainer"] = os.path.join(trainer_abs,
                                                "ctr_coding_trainer.py")
    trainers["CtrModulTrainer"] = os.path.join(trainer_abs,
                                               "ctr_modul_trainer.py")
    trainers["GeneralTrainer"] = os.path.join(trainer_abs,
                                              "general_trainer.py")


trainer_registry()


class TrainerFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def _build_trainer(yaml_path):
        print(envs.pretty_print_envs(envs.get_global_envs()))
        #打印运行时出现的那个参数的表格，所有参数都在环境变量中了

        train_mode = envs.get_trainer()
        trainer_abs = trainers.get(train_mode, None)
        #调general_trainer.py初始化trainer

        if trainer_abs is None:
            if not os.path.isfile(train_mode):
                raise IOError("trainer {} can not be recognized".format(
                    train_mode))
            trainer_abs = train_mode
            train_mode = "UserDefineTrainer"

        trainer_class = envs.lazy_instance_by_fliename(trainer_abs, train_mode)
        #trainer_class：<class 'general_trainer.GeneralTrainer'>
        trainer = trainer_class(yaml_path)
        return trainer

    @staticmethod
    def create(config):
        _config = envs.load_yaml(config)
        '''
        _config的内容：
        {'phase': [{'model': '{workspace}/model.py', 'thread_num': 1, 'name': 'phase_train', 'dataset_name': 'data1'}, {'model': '{workspace}/model.py', 'thread_num': 1, 'name': 'phase_infer', 'dataset_name': 'dataset_infer'}], 
        'hyper_parameters': {'cnn_dim': 128, 'optimizer': {'learning_rate': 0.001, 'class': 'Adagrad'}, 'max_len': 100, 'hid_dim': 96, 'emb_dim': 128, 'cnn_filter_size1': 1, 'cnn_filter_size2': 2, 'dict_dim': 33257, 'class_dim': 2, 'cnn_filter_size3': 3, 'is_sparse': False}, 
        'workspace': 'models/contentunderstanding/classification', 
        'runner': [{'init_model_path': '', 'phases': 'phase_train', 'save_inference_feed_varnames': [], 'name': 'train_runner', 'save_checkpoint_path': 'increment', 'epochs': 16, 'save_inference_path': 'inference', 'save_checkpoint_interval': 1, 'save_inference_fetch_varnames': [], 'print_interval': 10, 'save_inference_interval': 1, 'device': 'cpu', 'class': 'train'}, 
            {'init_model_path': 'increment/14', 'phases': 'phase_infer', 'name': 'infer_runner', 'device': 'cpu', 'class': 'infer', 'print_interval': 1}], 
        'mode': ['train_runner', 'infer_runner'], 
        'dataset': [{'type': 'DataLoader', 'data_converter': '{workspace}/reader.py', 'data_path': '{workspace}/data/train', 'name': 'data1', 'batch_size': 10}, 
            {'type': 'DataLoader', 'data_converter': '{workspace}/reader.py', 'data_path': '{workspace}/data/test', 'name': 'dataset_infer', 'batch_size': 2}]}
        '''
        envs.set_global_envs(_config)
        trainer = TrainerFactory._build_trainer(config)
        return trainer


# server num, worker num
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("need a yaml file path argv")
    trainer = TrainerFactory.create(sys.argv[1])
    trainer.run()
