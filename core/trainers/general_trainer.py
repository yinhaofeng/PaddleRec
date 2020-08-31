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
"""
General Trainer, applicable to many situations: Single/Cluster/Local_Cluster + PS/COLLECTIVE
"""
from __future__ import print_function

import os

from paddlerec.core.utils import envs
from paddlerec.core.trainer import Trainer, EngineMode, FleetMode


class GeneralTrainer(Trainer):
    """
    Trainer for various situations.
    """

    def __init__(self, config=None):
        Trainer.__init__(self, config)
        self.processor_register()
        self.abs_dir = os.path.dirname(os.path.abspath(__file__))
        self.runner_env_name = "runner." + self._context["runner_name"]

    def processor_register(self):
        print("processor_register begin")
        self.regist_context_processor('uninit', self.instance)
        self.regist_context_processor('network_pass', self.network)
        self.regist_context_processor('startup_pass', self.startup)
        self.regist_context_processor('train_pass', self.runner)
        self.regist_context_processor('terminal_pass', self.terminal)
        #regist_context_processor在基类trainer中，按照注册顺序加入Trainer基类中名为status_processor的字典

    def instance(self, context):
        #通过环境变量启动paddle分布式的实例，执行在模型训练前的所有操作。用户可以在这里进行下载数据，import不同的包，配置环境变量等操作。instance的官方实现位于instance.py。
        #您需要继承InstanceBase并命名为Instance，完成instance的实现，通过上下文信息字典context拿到模型所需信息，及保存相关配置。
        instance_class_path = envs.get_global_env(
            self.runner_env_name + ".instance_class_path", default_value=None)
        if instance_class_path:
            instance_class = envs.lazy_instance_by_fliename(
                instance_class_path, "Instance")(context)
        else:
            if self.engine == EngineMode.SINGLE:
                instance_class_name = "SingleInstance"
            elif self.fleet_mode == FleetMode.PSLIB:
                instance_class_name = "PslibInstance"
            elif self.fleet_mode == FleetMode.PS:
                instance_class_name = "PSInstance"
            elif self.fleet_mode == FleetMode.COLLECTIVE:
                instance_class_name = "CollectiveInstance"
            else:
                raise ValueError("Instance Init Error")
            instance_path = os.path.join(self.abs_dir, "framework",
                                         "instance.py")
            #instance基类的文件地址
            instance_class = envs.lazy_instance_by_fliename(
                instance_path, instance_class_name)(context)
            #根据文件地址和类名，生成一个文件中的类。然后用context作为参数得到该类的对象

        instance_class.instance(context)
        #将context中的status转换为network_pass,fleet转换为使用的fleet

    '''
    完整的context字典：
    {'status': 'uninit', 'is_exit': False, 
    'phases': [{'model': '{workspace}/model.py', 'thread_num': 1, 'name': 'phase_train', 'dataset_name': 'data1'}],
    'exe': <paddle.fluid.executor.Executor object at 0x7f304de96d10>, 'is_pslib': False, 'is_infer': False, 
    'dataset': [{'type': 'DataLoader', 'data_converter': '{workspace}/reader.py', 'data_path': '{workspace}/data/train', 'name': 'data1', 'batch_size': 10}, 
        {'type': 'DataLoader', 'data_converter': '{workspace}/reader.py', 'data_path': '{workspace}/data/test', 'name': 'dataset_infer', 'batch_size': 2}], 
    'engine': 1, 'fleet_mode': 'ps', 'place': <paddle.fluid.core_avx.CPUPlace object at 0x7f304de2ca40>, 
    'env': {'phase': [{'model': '{workspace}/model.py', 'thread_num': 1, 'name': 'phase_train', 'dataset_name': 'data1'}, 
            {'model': '{workspace}/model.py', 'thread_num': 1, 'name': 'phase_infer', 'dataset_name': 'dataset_infer'}], 
            'hyper_parameters': {'cnn_dim': 128, 'optimizer': {'learning_rate': 0.001, 'class': 'Adagrad'}, 'max_len': 100, 'hid_dim': 96, 'emb_dim': 128, 'cnn_filter_size1': 1,
                'cnn_filter_size2': 2, 'dict_dim': 33257, 'class_dim': 2, 'cnn_filter_size3': 3, 'is_sparse': False}, 
            'workspace': 'models/contentunderstanding/classification', 
            'runner': [{'init_model_path': '', 'phases': 'phase_train', 'save_inference_feed_varnames': [], 
                'name': 'train_runner', 'save_checkpoint_path': 'increment', 'epochs': 16, 'save_inference_path': 'inference', 'save_checkpoint_interval': 1, 
                'save_inference_fetch_varnames': [], 'print_interval': 10, 'save_inference_interval': 1, 'device': 'cpu', 'class': 'train'}, 
                {'init_model_path': 'increment/14', 'phases': 'phase_infer', 'name': 'infer_runner', 'device': 'cpu', 'class': 'infer', 'print_interval': 1}], 
            'mode': ['train_runner', 'infer_runner'], 
            'dataset': [{'type': 'DataLoader', 'data_converter': '{workspace}/reader.py', 'data_path': '{workspace}/data/train', 'name': 'data1', 'batch_size': 10}, 
                {'type': 'DataLoader', 'data_converter': '{workspace}/reader.py', 'data_path': '{workspace}/data/test', 'name': 'dataset_infer', 'batch_size': 2}]},
    'config_yaml': 'models/contentunderstanding/classification/config.yaml', 'device': 'CPU', 'is_fleet': False, 'runner_name': 'train_runner'}
    '''

    def network(self, context):
        #根据模型组网生成训练的program.您需要继承NetworkBase并命名为Network，完成build_network的实现，
        #通过上下文信息字典context拿到模型所需信息，并在context中保存模型的program与scope信息
        network_class_path = envs.get_global_env(
            self.runner_env_name + ".network_class_path", default_value=None)
        if network_class_path:
            network_class = envs.lazy_instance_by_fliename(network_class_path,
                                                           "Network")(context)
        else:
            if self.engine == EngineMode.SINGLE:
                network_class_name = "SingleNetwork"
            elif self.fleet_mode == FleetMode.PSLIB:
                network_class_name = "PslibNetwork"
            elif self.fleet_mode == FleetMode.PS:
                network_class_name = "PSNetwork"
            elif self.fleet_mode == FleetMode.COLLECTIVE:
                network_class_name = "CollectiveNetwork"
            else:
                raise ValueError("NetWork Init Error")
            network_path = os.path.join(self.abs_dir, "framework",
                                        "network.py")
            #instance基类的文件地址
            network_class = envs.lazy_instance_by_fliename(
                network_path, network_class_name)(context)
            #根据文件地址和类名，生成一个文件中的类。然后用context作为参数得到该类的对象

        network_class.build_network(context)
        #将context中的status转换为startup_pass,

    def startup(self, context):
        #初始化模型组网中的各个参数，以及加载模型
        #startup执行网络参数的初始化，或者模型的热启动，主要功能是执行exe.run(fluid.default_startup_program())
        startup_class_path = envs.get_global_env(
            self.runner_env_name + ".startup_class_path", default_value=None)
        if startup_class_path:
            startup_class = envs.lazy_instance_by_fliename(startup_class_path,
                                                           "Startup")(context)
        else:
            if self.engine == EngineMode.SINGLE and context["is_infer"]:
                startup_class_name = "SingleInferStartup"
            elif self.engine == EngineMode.SINGLE and not context["is_infer"]:
                startup_class_name = "SingleStartup"
            elif self.fleet_mode == FleetMode.PS or self.fleet_mode == FleetMode.PSLIB:
                startup_class_name = "PSStartup"
            elif self.fleet_mode == FleetMode.COLLECTIVE:
                startup_class_name = "CollectiveStartup"
            else:
                raise ValueError("Startup Init Error")
            startup_path = os.path.join(self.abs_dir, "framework",
                                        "startup.py")
            startup_class = envs.lazy_instance_by_fliename(
                startup_path, startup_class_name)(context)
        startup_class.startup(context)

    def runner(self, context):
        #会根据环境分别调用dataset与dataloader进行训练的流程。runner是运行的主要流程，主要功能是reader的运行，网络的运行，指标的打印以及模型的保存
        runner_class_path = envs.get_global_env(
            self.runner_env_name + ".runner_class_path", default_value=None)
        if runner_class_path:
            runner_class = envs.lazy_instance_by_fliename(runner_class_path,
                                                          "Runner")(context)
        else:
            if self.engine == EngineMode.SINGLE and context["is_infer"]:
                runner_class_name = "SingleInferRunner"
            elif self.engine == EngineMode.SINGLE and not context["is_infer"]:
                runner_class_name = "SingleRunner"
            elif self.fleet_mode == FleetMode.PSLIB:
                runner_class_name = "PslibRunner"
            elif self.fleet_mode == FleetMode.PS:
                runner_class_name = "PSRunner"
            elif self.fleet_mode == FleetMode.COLLECTIVE:
                runner_class_name = "CollectiveRunner"
            else:
                raise ValueError("Runner Init Error")
            runner_path = os.path.join(self.abs_dir, "framework", "runner.py")
            runner_class = envs.lazy_instance_by_fliename(
                runner_path, runner_class_name)(context)
        runner_class.run(context)

    def terminal(self, context):
        #停止worker，以及执行模型训练后的所有操作
        #terminal主要进行分布式训练结束后的stop worker，以及其他需要在模型训练完成后进行的工作，比如数据整理，模型上传等等
        terminal_class_path = envs.get_global_env(
            self.runner_env_name + ".terminal_class_path", default_value=None)
        if terminal_class_path:
            terminal_class = envs.lazy_instance_by_fliename(
                terminal_class_path, "Terminal")(context)
            terminal_class.terminal(context)
        else:
            terminal_class_name = "TerminalBase"
            if self.engine != EngineMode.SINGLE and self.fleet_mode != FleetMode.COLLECTIVE:
                terminal_class_name = "PSTerminal"

            terminal_path = os.path.join(self.abs_dir, "framework",
                                         "terminal.py")
            terminal_class = envs.lazy_instance_by_fliename(
                terminal_path, terminal_class_name)(context)
        terminal_class.terminal(context)
        context['is_exit'] = True
