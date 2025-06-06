
import argparse
import warnings
from typing import List

import torch
from networkx import descendants
import numpy as np
from termcolor import colored
import transformers
# HACK: monkey patch for disable flash attn op, which will case 'unsupported symbol' error in onnx
# transformers.utils.is_flash_attn_2_available = lambda: False

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.model.policy import BasePolicy, Gr00tPolicy, unsqueeze_dict_values
from gr00t.utils.eval import calc_mse_for_single_trajectory

warnings.simplefilter("ignore", category=FutureWarning)


def get_module_input(dataset: LeRobotSingleDataset, modality_keys: List[str], policy: Gr00tPolicy, ):
    n1: GR00T_N1 = policy.model
    traj_id = 0
    step_count = 0
    data_point = dataset.get_step_data(traj_id, step_count)
    is_batch = policy._check_state_is_batched(data_point)
    if not is_batch:
        data_point = unsqueeze_dict_values(data_point)
    normalized_input = policy.apply_transforms(data_point)

    backbone_inputs, action_inputs = n1.prepare_input(normalized_input)
    backbone_inputs = (
        backbone_inputs.pixel_values, 
        backbone_inputs.input_ids,
        backbone_inputs.attention_mask,
    )
    return backbone_inputs, action_inputs


"""
Example command:

python scripts/eval_policy.py --host localhost --port 5555 --plot
    --modality_keys right_arm right_hand
    --steps 250
    --trajs 1000
    --action_horizon 16
    --video_backend decord
    --dataset_path demo_data/robot_sim.PickNPlace/
    --embodiment_tag gr1
    --data_config gr1_arms_waist
provide --model_path to load up the model checkpoint in this script.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_config",
        type=str,
        default="gr1_arms_only",
        choices=list(DATA_CONFIG_MAP.keys()),
        help="data config name",
    )
    parser.add_argument("--video_backend", type=str, default="torchvision_av")
    parser.add_argument("--dataset_path", type=str, default="demo_data/robot_sim.PickNPlace/")
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="new_",
    )
    ## When using a model instead of client-server mode.
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="[Optional] Path to the model checkpoint directory, this will disable client server mode.",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps if model_path is provided",
        default=4,
    )
    args = parser.parse_args()

    data_config = DATA_CONFIG_MAP[args.data_config]

    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    modality_keys = [k.replace('state.', '') for k in data_config.state_keys]

    policy: Gr00tPolicy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device="cpu",
        export_mode=True,
    )
    
    modality = policy.get_modality_config()
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
    )

    n1: GR00T_N1 = policy.model
    n1.backbone
    n1.action_head
    backbone_inputs, action_inputs = get_module_input(dataset, modality_keys, policy)
    # n1.backbone._export_mode = True
    torch.onnx.export(n1.backbone, backbone_inputs, 'backbone.onnx', dynamo=True)
