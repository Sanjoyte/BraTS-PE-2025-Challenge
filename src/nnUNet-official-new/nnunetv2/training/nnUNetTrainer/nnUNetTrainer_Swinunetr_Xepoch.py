from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import *


import os
import torch
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count

from typing import Tuple, Union, List
from monai.networks.nets import SwinUNETR

# /mnt/h_public/yyx/code/nnUNet-official-new/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py


### 继承关系
## nnUNetTrainer --> nnUNetTrainer_Swinunetr --> nnUNetTrainer_Swinunetr_Xepochs
## 基类--修改模型架构、深度监督--修改epochs(epoch1005加载预训练权重)

class nnUNetTrainer_Swinunetr( nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        """
        # self.print_to_log_file("======= the architecture of the network is set as SwinUNETR =========")
        print("Note: ======= the architecture of the network is set as SwinUNETR =========")


        print("==============arch_init_kwargs: ", arch_init_kwargs)
        
        model = SwinUNETR(
                img_size=(96, 160, 160),
                in_channels=num_input_channels,
                out_channels=num_output_channels,
                feature_size=48,
                use_checkpoint=True,
            )
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        # Deliberately a no-op: SwinUNETR has no deep-supervision heads, so the
        # base trainer's `mod.decoder.deep_supervision = enabled` would fail.
        self.print_to_log_file("======= disable the deep_supervision mode =========")

        pass


class nnUNetTrainer_Swinunetr_500epochs(nnUNetTrainer_Swinunetr):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = 500

class nnUNetTrainer_Swinunetr_1000epochs(nnUNetTrainer_Swinunetr):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = 1000


class nnUNetTrainer_Swinunetr_1001epochs(nnUNetTrainer_Swinunetr):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = 1001


class nnUNetTrainer_Swinunetr_1002epochs(nnUNetTrainer_Swinunetr):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = 1002


class nnUNetTrainer_Swinunetr_1003epochs(nnUNetTrainer_Swinunetr):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = 1003



class nnUNetTrainer_Swinunetr_1005epochs(nnUNetTrainer_Swinunetr):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = 1005
        self.initial_lr = 1e-3 

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        """
        # self.print_to_log_file("======= the architecture of the network is set as SwinUNETR =========")
        print("Note: ======= the architecture of the network is set as SwinUNETR =========")
        
        model = SwinUNETR(
                img_size=(128, 128, 128),
                in_channels=num_input_channels,
                out_channels=num_output_channels,
                feature_size=48,
                use_checkpoint=True,
            )

        print("Note: ======= patch_size=[128,128,128]. You need to use -p nnUNetPlansPatch =========")

        # 2. 加载预训练权重 / load the transfer-learning weights.
        # The path used to be hardcoded to the authors' own machine, so on any
        # other machine the load below failed and the model silently fell back to
        # random initialization — quietly dropping the BraTS 2021 transfer
        # learning this trainer exists for. It comes from the environment now
        # (PEDS_SWIN_PRETRAINED in .env); leave that unset to keep the old
        # random-init behaviour deliberately.
        weight_path = os.environ.get("PEDS_SWIN_PRETRAINED", "").strip()

        if not weight_path:
            print("Note: PEDS_SWIN_PRETRAINED is not set — training SwinUNETR "
                  "from random initialization (no BraTS 2021 transfer learning).")
            return model

        print("weight_path: ", weight_path)

        # 方法1：直接加载权重（推荐）
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
            
            # 检查checkpoint的结构
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # 处理可能的键名不匹配问题
            model_dict = model.state_dict()
            
            # 过滤掉不匹配的键
            filtered_dict = {}
            for k, v in state_dict.items():
                # 移除可能的 'module.' 前缀
                key = k.replace('module.', '') if k.startswith('module.') else k
                if key in model_dict and v.shape == model_dict[key].shape:
                    filtered_dict[key] = v
                else:
                    print(f"jump off the key: {k} -> {key}")
            
            # 加载权重
            model.load_state_dict(filtered_dict, strict=False)
            print(f"成功加载权重，匹配的参数数量: {len(filtered_dict)}/{len(model_dict)}")
            
        except Exception as e:
            print(f"权重加载失败: {e}")
            print("将使用随机初始化的权重")
        
        return model



        
