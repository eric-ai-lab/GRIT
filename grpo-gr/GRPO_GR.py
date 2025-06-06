
import os, sys
if 'WANDB_PROJECT' not in os.environ:
    os.environ["WANDB_PROJECT"] = "grpo_tool_undefined"  # name your W&B project

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trl'))

from rewards import (
    gpt_score_reward, gpt_score_reward_1, bleu_score_reward, answer_format_reward,repetitive_reward,
    grounded_region_specific_thinking_format_reward_think_rethink,
    think_and_rethink_format_reward, 
    
)
from GRPO_GRTrainer import GRPOGRTrainer

from accelerate import Accelerator
from relationReasoningDataset import RelationReasoningDataset
accelerator = Accelerator()
import shutil
import numpy as np

from transformers import (
    HfArgumentParser
)
from dataclasses import dataclass, field
from trl import (
    ModelConfig,
    GRPOConfig, 
    ScriptArguments,
    get_peft_config,
)

@dataclass
class VLToolGRPOConfig(GRPOConfig):
    """
    Extended arguments for VLTool GRPO training script, including InternVL3-specific parameters.
    """
    eval_only: bool = field(default=False, metadata={"help": "Whether to only run evaluation."})
    setting: str = field(default="rr_use_external_grounding_tool", metadata={"help": "Experiment setting."})
    project_root_path: str = field(default="", metadata={"help": "The root path of the project."})
    python_path_for_dino: str = field(default="/usr/local/anaconda3/envs/grpo_env/bin/python", metadata={"help": "The conda environment path."})
    train_data_path: str = field(default="", metadata={"help": "Path to training data."})
    train_image_folder_path: str = field(default="", metadata={"help": "Path to training images."})
    eval_data_path: str = field(default="", metadata={"help": "Path to evaluation data."})
    eval_image_folder_path: str = field(default="", metadata={"help": "Path to evaluation images."})
    max_turns: int = field(default=2, metadata={"help": "Max conversational turns."})
    tool_port_starting_num: int = field(default=8020, metadata={"help": "Starting port number for tools."})
    
    # InternVL3-specific parameters
    downsample_ratio: float = field(default=0.5, metadata={"help": "Downsample ratio for vision inputs."})
    mlp_path: str = field(default=None, metadata={"help": "Path to pretrained MLP projector."})
    force_image_size: int = field(default=448, metadata={"help": "Forced resize image dimension."})
    dynamic_image_size: bool = field(default=True, metadata={"help": "Enable dynamic image resizing."})
    vision_select_layer: int = field(default=-1, metadata={"help": "Selected vision model layer for extraction."})
    min_dynamic_patch: int = field(default=8, metadata={"help": "Minimum dynamic patch size."})
    max_dynamic_patch: int = field(default=6, metadata={"help": "Maximum dynamic patch size."})
    use_thumbnail: bool = field(default=True, metadata={"help": "Use thumbnails instead of full images."})
    ps_version: str = field(default='v2', metadata={"help": "Position embedding scheme version."})
    drop_path_rate: float = field(default=0.1, metadata={"help": "Drop path rate for stochastic depth in ViT."})
    use_backbone_lora: int = field(default=0, metadata={"help": "LoRA rank for vision backbone."})
    use_llm_lora: int = field(default=0, metadata={"help": "LoRA rank for LLM."})
    freeze_backbone: bool = field(default=True, metadata={"help": "Freeze vision backbone parameters."})
    freeze_llm: bool = field(default=False, metadata={"help": "Freeze LLM parameters."})
    unfreeze_vit_layers: int = field(default=0, metadata={"help": "Number of ViT layers to unfreeze from the end."})
    freeze_mlp: bool = field(default=False, metadata={"help": "Freeze MLP layers."})
    unfreeze_lm_head: bool = field(default=False, metadata={"help": "Unfreeze the language model head."})
    conv_style: str = field(default="internvl2_5", metadata={"help": "Convolution style for the model."})
    down_sample_ratio: float = field(default=0.5, metadata={"help": "Downsample ratio for the model."})

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, VLToolGRPOConfig, ModelConfig)) 
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    if accelerator.is_main_process:
        
        shutil.rmtree(training_args.output_dir, ignore_errors=True)
        
    peft_config = get_peft_config(model_args)
    prompt_suffix = ''
    prompt = ''
    if script_args.dataset_name == 'rr':

        request_token = "<object>"
        call_token = "</object>"
        
        assert training_args.max_turns == 1
    
        if 'add_grounded_reasoning' in training_args.setting:
            prompt = ""
            if '_think_rethink' in training_args.setting:
                prompt_suffix = ''' First, think between <think> and </think> while output necessary coordinates needed to answer the question in JSON with key 'bbox_2d'. Then, based on the thinking contents and coordinates, rethink between <rethink> </rethink> and then answer the question after <answer>.\n'''


            if '_icl' in training_args.setting:
                prompt_suffix = ''' First, think between <think> and </think> while output necessary coordinates needed to answer the question in JSON with key 'bbox_2d'. Then, based on the thinking contents and coordinates, rethink between <rethink> </rethink> and then answer the question after <answer>.\n'''
                prompt_suffix += '''\nExample: Question: How many chairs are there?\n<think> Let me check them out \n{\n  \"bbox_2d\": [142, 314, 299, 447],\n  \"label\": \"chair\",\n  \"bbox_2d\": [0, 439, 58, 531],\n  \"label\": \"chair\",\n  \"bbox_2d\": [222, 260, 299, 324],\n  \"label\": \"chair\",\n  \"bbox_2d\": [285, 325, 336, 430],\n  \"label\": \"chair\",\n  \"bbox_2d\": [316, 329, 336, 387],\n  \"label\": \"chair\"\n}\n \n</think>\n<rethink>\nBased on the bounding box coordinates, there are 5 chairs in the picture. \n</rethink>\n<answer>\n5\n</answer>'''
                if 'intern' in training_args.setting:
                    prompt_suffix = ''' First, think between <think> and </think> while output necessary coordinates needed to answer the question between <box> and </box>. Then, based on the thinking contents and coordinates, rethink between <rethink> </rethink> and then answer the question after <answer>.\n'''
                    prompt_suffix += '''\nExample: Question: How many chairs are there?\n<think> Let me check them out <ref>chair</ref> <box>[[142, 314, 299, 447], [0, 439, 58, 531], [222, 260, 299, 324], [285, 325, 336, 430], [316, 329, 336, 387]]</box> </think>\n<rethink>\nBased on the bounding box coordinates, there are 5 chairs in the picture. \n</rethink>\n<answer>\n5\n</answer>'''
                
        elif 'SFT' in training_args.setting:
            prompt_suffix = "\nFirst, output a lists of all necessary coordinates (bbox_2d) needed to answer the question, then answer the question after <answer>.\n"
        else:
            prompt_suffix = ''
            prompt = ''

        
        tools = ['BoundingboxBrushTool:bbox-brush']
        
        
        
        if ',' in training_args.train_data_path:
            assert ',' in training_args.train_image_folder_path, "Please provide the same number of image folders as the number of train data paths."
            train_dataset = {}
            train_data_paths = training_args.train_data_path.split(',')
            train_image_folder_paths = training_args.train_image_folder_path.split(',')

            train_dataset = RelationReasoningDataset(train_data_paths, train_image_folder_paths, prompt = prompt, prompt_suffix = prompt_suffix)
        else:
            train_dataset = RelationReasoningDataset(training_args.train_data_path, training_args.train_image_folder_path, prompt = prompt, prompt_suffix = prompt_suffix)
        
        if ',' in training_args.eval_data_path:
            assert ',' in training_args.eval_image_folder_path, "Please provide the same number of image folders as the number of eval data paths."
            eval_dataset = {}
            eval_data_paths = training_args.eval_data_path.split(',')
            eval_image_folder_paths = training_args.eval_image_folder_path.split(',')
            for i, eval_data_path in enumerate(eval_data_paths):
                eval_dataset[eval_data_path] = RelationReasoningDataset(eval_data_path, eval_image_folder_paths[i], prompt = prompt, prompt_suffix = prompt_suffix)
        else:
            eval_dataset = RelationReasoningDataset(training_args.eval_data_path, training_args.eval_image_folder_path, prompt = prompt, prompt_suffix = prompt_suffix, limits=None) #30 if not training_args.eval_only else 
        
        #################
        ### Reward functions
        #################
        REWARD_FUNCS_REGISTRY = {
            "answer_gpt_accuracy": gpt_score_reward,
            "answer_blue_score": bleu_score_reward,
            "answer_format_reward": answer_format_reward,
            "repetitive_reward": repetitive_reward,
        }

        if '_think_rethink' in training_args.setting:
            REWARD_FUNCS_REGISTRY["JSON_format_reward"] = grounded_region_specific_thinking_format_reward_think_rethink
            REWARD_FUNCS_REGISTRY["think_format_reward"] = think_and_rethink_format_reward

        if 'vanilla_zeroshot' in training_args.setting:
            REWARD_FUNCS_REGISTRY["answer_gpt_accuracy"] = gpt_score_reward_1

        reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in REWARD_FUNCS_REGISTRY.keys()]



    else:
        raise ValueError(f"Dataset name in {training_args.setting} is not supported.")    
    
    end_flag = False
    if os.path.exists(training_args.output_dir):
        checkpoint_list = [d for d in os.listdir(training_args.output_dir) if d.endswith('end_of_training.txt')]
        if len(checkpoint_list) > 0:
            print(f"Training has been finished. Please remove {training_args.output_dir} to continue training.")
            end_flag = True
        

    if not end_flag:
        ################
        # Training
        ################
        trainer = accelerator.prepare(GRPOGRTrainer( 
            args=training_args,
            model=  model_args.model_name_or_path, 
            tool_names = tools,
            request_token = request_token,
            call_token = call_token,
            reward_funcs= reward_funcs, 
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,   
        ))

        if training_args.eval_only:
            trainer.evaluate()
        else:
            trainer.train()

        # save a mark for the end of training
        with open(os.path.join(training_args.output_dir, "end_of_training.txt"), "w") as f:
            f.write("Training finished.\n")
            
        
