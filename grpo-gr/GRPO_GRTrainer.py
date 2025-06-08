
import json
import wandb
import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch
import numpy as np
import re
import base64
from PIL import Image
import io
import time
import datetime
import random
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
    load_tool

)
from vision_process import process_vision_info
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

import types

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

from internVL_src import *


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source: Sized, repeat_count: int):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)

    def __iter__(self):
        indexes = [idx for idx in torch.randperm(self.num_samples).tolist() for _ in range(self.repeat_count)]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count

class GRPOGRTrainer(Trainer):
    """
    Trainer for the GRPO-GR method.
    Please refer to the paper [GRIT: Teaching MLLMs to Think with Images]. It is adapted from the GRPO.

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"message"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        tool_names: list[str],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 256*28*28, #12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2", #"sdpa",
        tool_model_id: Optional[str] = 'IDEA-Research/grounding-dino-base',
        request_token: Optional[str] = "<object>",
        call_token: Optional[str] = "</object>",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        def _load_optimizer_and_scheduler(self, checkpoint_path):
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")

            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.accelerator.device))
                print("Optimizer state loaded from checkpoint.")

            if os.path.exists(scheduler_path):
                self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.accelerator.device))
                print("Scheduler state loaded from checkpoint.")

        # if isinstance(model, str):
        assert isinstance(model, str)
        model_id = model
        self.model_id = model_id
        
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        
        if "qwen" in model_id.lower():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        elif "internvl" in model_id.lower():
            config = InternVLChatConfig.from_pretrained(model_id)
            config.vision_config.drop_path_rate = args.drop_path_rate
            if config.llm_config.model_type == 'internlm2':
                config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
                print('Using flash_attention_2 for InternLM')
            else:
                config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
                print('Using flash_attention_2 for LLaMA')
            config.template = args.conv_style
            config.select_layer = args.vision_select_layer
            config.dynamic_image_size = args.dynamic_image_size
            config.use_thumbnail = args.use_thumbnail
            config.ps_version = args.ps_version
            config.min_dynamic_patch = args.min_dynamic_patch
            config.max_dynamic_patch = args.max_dynamic_patch

            model = InternVLChatModel.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, config=config)
            patch_size = model.config.vision_config.patch_size
            if model.config.vision_config.image_size != args.force_image_size:
                print(f'Resizing position embedding from '
                            f'{model.config.vision_config.image_size} '
                            f'to {args.force_image_size}...')
                model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                        new_size=args.force_image_size,
                                                        patch_size=patch_size)
                model.config.vision_config.image_size = args.force_image_size
            model.config.force_image_size = args.force_image_size
            model.num_image_token = int((args.force_image_size // patch_size) ** 2 * (args.down_sample_ratio ** 2))
            
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "qwen" in model_id.lower():
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "internvl" in model_id.lower():
                self.ref_model = InternVLChatModel.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, config=config)
                if self.ref_model.config.vision_config.image_size != args.force_image_size:
                    print(f'Resizing position embedding from '
                                f'{self.ref_model.config.vision_config.image_size} '
                                f'to {args.force_image_size}...')
                    self.ref_model.vision_model.resize_pos_embeddings(old_size=self.ref_model.config.vision_config.image_size,
                                                            new_size=args.force_image_size,
                                                            patch_size=patch_size)
                    self.ref_model.config.vision_config.image_size = args.force_image_size
                self.ref_model.config.force_image_size = args.force_image_size
                self.ref_model.num_image_token = int((args.force_image_size // patch_size) ** 2 * (args.down_sample_ratio ** 2))        
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "qwen" in model_id.lower(): #or "Aria" in model_id:
                try:
                    processing_class = AutoProcessor.from_pretrained(model_id)
                except:
                    processing_class = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
                
                processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels
            elif "internvl" in model_id.lower():
                # processing_class = AutoTokenizer.from_pretrained(model_id, padding_side="left",trust_remote_code=True, use_fast=False)

                # Load pretrained model, tokenizer, and image processor
                tokenizer_path = model_id
                print(f'Loading Tokenizer: {tokenizer_path}')
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
                tokenizer.tokenizer_path = tokenizer_path
                tokenizer.model_max_length = args.max_prompt_length
                token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                            QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                            REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
                num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
                img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
                tcs_loader = None
                processing_class = tokenizer
                
                if num_new_tokens > 0:
                    model.language_model.resize_token_embeddings(len(tokenizer))
                    self.ref_model.language_model.resize_token_embeddings(len(tokenizer))

                    output_embeddings = model.language_model.get_output_embeddings().weight.data
                    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                    output_embeddings[-num_new_tokens:] = output_embeddings_avg

                    model.config.llm_config.vocab_size = len(tokenizer)
                    model.language_model.config.vocab_size = len(tokenizer)      
                    self.ref_model.config.llm_config.vocab_size = len(tokenizer)
                    self.ref_model.language_model.config.vocab_size = len(tokenizer)      
                model.img_context_token_id = img_context_token_id
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                self.ref_model.img_context_token_id = img_context_token_id
                self.ref_model.language_model.config.use_cache = False
                self.ref_model.vision_model.gradient_checkpointing = True
                self.ref_model.vision_model.encoder.gradient_checkpointing = True
                    
        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "message". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        assert self.use_vllm == False
        
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=processing_class.pad_token_id,
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
                
        self.request_token = request_token
        self.call_token = call_token
        self.response_token = "<response>"
        self.submit_token = "<answer_end>" # not used
        self.max_turns = self.args.max_turns
        self.max_tool_response = 30
        self.tool_model_id = tool_model_id 

        self.process_uinque_id = int(self.accelerator.process_index) + self.args.tool_port_starting_num
        assert len(tool_names) == 1, "Only one tool is supported for now"
        self.tool_func = tool_names[0].split(":")[0]
        self.tools = {}
        for tool_name in tool_names:
            assert len(tool_name.split(":")) == 2
            tool_func = tool_name.split(":")[0]
            tool_path = tool_name.split(":")[1]
            self.tools[tool_func] = load_tool(f"./custom_tools/{tool_path}", port=self.process_uinque_id, project_root_path=self.args.project_root_path, python_path_for_dino=self.args.python_path_for_dino)

        
        to_phrase = "Wait, I need to think again. "
        from_phrase = "<rethink>\n"
        if "qwen" in model_id.lower():
            self.eos_token_id = processing_class.tokenizer.eos_token_id
            self.pad_token_id = processing_class.tokenizer.pad_token_id
        elif "internvl" in model_id.lower():
            self.eos_token_id = processing_class.convert_tokens_to_ids(EOS_TOKEN)
            self.pad_token_id = processing_class.convert_tokens_to_ids(END_OF_TEXT_TOKEN)

        self.is_train = True

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["message"]

    # We need a custom sampler that samples the same prompt multiple times
    def _get_train_sampler(self) -> Sampler:
        return RepeatRandomSampler(self.train_dataset, self.num_generations)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatRandomSampler(eval_dataset, 2) # set minimum num_generation for eval, making the eval faster #self.num_generations)

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        if "qwen" in self.model_id.lower():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        elif "internvl" in self.model_id.lower():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_flags=image_grid_thw)
        logits = outputs.logits
        logits = logits[:, -logits_to_keep-1:-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens
     
    def parse_tool_call(self, text):
        """
        Parse request string. Expected format: <request_token>query<end_token>, query could be (x1,y1),(x2,y2) or just str.
        """
        # result = re.search(f"(?<={self.request_token}).*?(?={self.call_token})", text, re.DOTALL)

        # # if we can't find a <request>/<call> span we return none
        # if result is None:
        #     return None
        # else:
        #     extracted_text = result.group()

        extracted_text = text
        # if we can't find a bbox span we return none
        target_str = re.search(f"(?<={self.request_token}).*?(?={self.call_token})", extracted_text, re.DOTALL)
        if target_str is None:
            return None
        else:
            target_str = target_str.group()
            return target_str
        

    def parse_grounding(self, text):
        """
        Parse grounding string
        """
        # result = re.search(f"(?<={self.request_token}).*?(?={self.call_token})", text, re.DOTALL)

        # # if we can't find a <request>/<call> span we return none
        # if result is None:
        #     return None
        # else:
        #     extracted_text = result.group()

    
        pattern = r'\b\d+,\s*\d+,\s*\d+,\s*\d+\b'
        matches = re.findall(pattern, text)

        if len(matches)>0:
            return matches
        else:
            return None
        
    def multi_modal_get_item(self, data_items):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.args.force_image_size,
                                    pad2square=False, normalize_type='imagenet')

        # Ensure the first conversation contains an image placeholder
        # if '<image>' not in data_item['question']:
        #     data_item['question'] = '<image>\n' + data_item['question']

        pixel_values = []
        processed_conversations = []
        num_patches = []
        images = []
        for data_item in data_items:
            image_path = [data_item['message'][i]['content'][k]['image'] for i in range(len(data_item['message'])) for k in range(len(data_item['message'][i]['content'])) if 'image' in data_item['message'][i]['content'][k]]
            num_image = len(image_path)
            # not using tcs_loader, use PIL
            # Ensure that there is only one patch, otherwise, we need apply dynamic image size 
            assert len(image_path) == 1, f'The number of patches should be 1, but got {len(image_path)}.'
            
            pixel_value, image = load_image(image_path[0], self.is_train)
            images.append(image)
            pixel_values.append(pixel_value)

            # Ensure that there is only one patch, otherwise, we need apply dynamic image size 
            num_patches.append(pixel_value.size(0))
            # assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

            # Select the appropriate preprocessing function based on the template name
            # preprocess_function = preprocess_internvl2_5



            # Preprocess the conversations and generate the return dictionary
            conversations = [
                {'from': 'human', 'value': '<image>\n' + data_item['message'][0]['content'][-1]['text']},
                # {'from': 'gpt', 'value': data_item['gt_answer']},
            ]
            
            
            template_name = self.args.conv_style
            # sources = [deepcopy(chosen_conversations)]
            num_image_token_list = [self.model.num_image_token * num_patches[-1]]
            
            if conversations[0]['from'] == 'system':
                system_prompt = conversations[0]['value']
                conversations = conversations[1:]  # remove system prompt
            else:
                conv = get_conv_template(template_name)
                system_prompt = conv.system_message
                # system_prompt = None
            
            # if not text_only:
            new_conversations = []
            current_image_idx = 0
            for conversation in conversations:
                if conversation['from'] == 'human':
                    image_cnt = conversation['value'].count('<image>')
                    for i in range(image_cnt):
                        if current_image_idx == num_image:
                            break
                        image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}{IMG_END_TOKEN}'
                        conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                        current_image_idx += 1
                new_conversations.append(conversation)
            conversations = new_conversations
            assert current_image_idx == num_image, f'{current_image_idx} != {num_image}'

            batches, roles = [], []
            if system_prompt is not None:
                batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
                roles.append('system')
            for conversation in conversations:
                if conversation['from'] == 'human':
                    batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
                    roles.append('human')
                elif conversation['from'] == 'gpt':
                    batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
                    roles.append('gpt')
                else:
                    raise NotImplementedError
            processed_conversations.append(''.join(batches))
        
        chosen_ret = self.processing_class(
                    processed_conversations,
                    padding=True,
                    padding_side='left',
                    return_tensors="pt",
                    max_length=self.processing_class.model_max_length,
                    truncation=True,
                )
        
        


        # Create the final return dictionary
        ret = dict(
            input_ids=chosen_ret['input_ids'],
            # chosen_labels=chosen_ret['labels'][0],
            attention_mask=chosen_ret['attention_mask'],
            # rejected_input_ids=rejected_ret['input_ids'][0],
            # rejected_labels=rejected_ret['labels'][0],
            # rejected_attention_mask=rejected_ret['attention_mask'][0],
            pixel_values=torch.stack([sect for sects in pixel_values for sect in sects],dim=0),
            image_grid_thw=torch.tensor([1 for n in num_patches for i in range(n)], dtype=torch.long),
        )
        return ret, images

    
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]], is_train = True) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        self.is_train = is_train
        
        total_completion_mask = torch.zeros(len(inputs), self.max_completion_length, device=device)
        total_completion_ids = torch.full(
            (len(inputs), self.max_completion_length), self.processing_class.pad_token_id, device=device
        )
        total_completion_pointers = np.zeros((len(inputs),), dtype=int)
        finish_flags = np.zeros((len(inputs),), dtype=bool)
        
        # Potentially support multiple turns, but we only use one turn for now i.e. max_turns = 1
        j=-1
        while True:
            j+=1
            
            if j == 0:
                original_prompts = [x["message"] for x in inputs]
                inputs_tobe_updated_each_turn = inputs.copy()
            if "qwen" in self.model_id.lower(): 
                # Process inputs
                text = [self.processing_class.apply_chat_template(
                    inp['message'], 
                    tokenize=False, 
                    add_generation_prompt=True
                ) for inp in inputs_tobe_updated_each_turn]
                image_inputs, video_inputs = [], []
                            
                # process_vision_info(inp['message']) for inp in inputs]
                # process the image_inputs to be a list for image(s) in each conversation
                image_inputs, video_inputs = [], []
                for inp in inputs_tobe_updated_each_turn:
                    image_input, video_input = process_vision_info(inp['message'])
                    if image_input:
                        image_inputs.append(image_input)
                    if video_input:
                        video_inputs.append(video_input)
                
                for i in range(len(text)):
                    if len(text[i].split('<|im_end|>\n<|im_start|>assistant\n')) > 2:
                        t_list = text[i].split('<|im_end|>\n<|im_start|>assistant\n')[:-1]
                        text[i] = '<|im_end|>\n<|im_start|>assistant\n'.join(t_list[:2]) + ''.join(t_list[2:])
                
                video_list = [vi  for vid_inps in video_inputs for vi in vid_inps]
                if len(video_list) == 0:
                    video_list = None
                image_list = [ii  for img_inps in image_inputs for ii in img_inps]
                if len(image_list) == 0:
                    image_list = None
                    
                prompt_inputs = self.processing_class(
                    text=text,
                    images=image_list,
                    videos=video_list,
                    padding=True,
                    padding_side='left',
                    return_tensors="pt"
                )
            elif "internvl" in self.model_id.lower():
                prompt_inputs, image_inputs = self.multi_modal_get_item(inputs_tobe_updated_each_turn)

            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            latest_prompt_ids, latest_prompt_mask, latest_pixel_values = \
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"], prompt_inputs.get("pixel_values")
            
            latest_image_grid_thw = prompt_inputs.get("image_grid_thw")
            
            
            if self.max_turns == j:
                break
            

            if j == 0:
                if latest_prompt_ids.size(1) > self.max_prompt_length:
                    # we do not trip the input in case the image token is in complete
                    print(f"Warning: prompt length {latest_prompt_ids.size(1)} exceeds the maximum prompt length {self.max_prompt_length}. Truncating.")
                original_prompt_ids = latest_prompt_ids.clone()
                original_prompt_mask = latest_prompt_mask.clone()
                
                if self.generation_config.stop_strings is None:
                    self.generation_config.stop_strings = ['</answer>']
                elif self.call_token not in self.generation_config.stop_strings and self.submit_token not in self.generation_config.stop_strings:
                    self.generation_config.stop_strings += ['</answer>']
                grounded_images = []
                
            p_mask = latest_prompt_mask
                
            start_time = time.time()
            # Generate completions does not use vLLM but only regular generation, need more work to support vLLM
            if "qwen" in self.model_id.lower():
                with torch.no_grad():


                    if self.is_train:
                        
                        prompt_completion_ids = self.model.generate(
                            **prompt_inputs, generation_config=self.generation_config, tokenizer=self.processing_class.tokenizer
                        ) # 
                    else:
                        eval_generation_config = deepcopy(self.generation_config)
                        eval_generation_config.temperature = 0.001
                        eval_generation_config.do_sample = True
                        eval_generation_config.top_k = 1
                        eval_generation_config.top_p = 0.
                        prompt_completion_ids = self.model.generate(
                                **prompt_inputs, tokenizer=self.processing_class.tokenizer,
                                generation_config=eval_generation_config,
                            ) # 
                        
                # Compute prompt length and extract completion ids
                prompt_length = latest_prompt_ids.size(1)
                if not (latest_prompt_ids == prompt_completion_ids[:, :prompt_length]).all():
                    print("Prompt mismatch, unexpected.")
                completion_ids = prompt_completion_ids[:, prompt_length:]
                
            elif "internvl" in self.model_id.lower():
                with torch.no_grad():
                    if self.is_train:
                        
                        prompt_completion_ids = self.model.generate(
                                input_ids=prompt_inputs["input_ids"], 
                                attention_mask=prompt_inputs["attention_mask"], 
                                pixel_values=prompt_inputs.get("pixel_values"), 
                                generation_config=self.generation_config, 
                                tokenizer=self.tokenizer
                            ) # 
                    else:
                        eval_generation_config = deepcopy(self.generation_config)
                        eval_generation_config.temperature = 0.001
                        eval_generation_config.do_sample = True
                        eval_generation_config.top_k = 1
                        eval_generation_config.top_p = 0.
                        prompt_completion_ids = self.model.generate(
                                input_ids=prompt_inputs["input_ids"], 
                                attention_mask=prompt_inputs["attention_mask"], 
                                pixel_values=prompt_inputs.get("pixel_values"), 
                                generation_config=eval_generation_config, 
                                tokenizer=self.tokenizer
                            )

                completion_ids = prompt_completion_ids

            time_taken = time.time() - start_time
            print(f"\n[Timer]: \nTime taken for generation: {time_taken:.2f} seconds.\n")

    

            start_time = time.time()

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            is_pad = completion_ids == self.pad_token_id
            is_eos = is_eos | is_pad
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            
            
            
            for i in range(len(completion_ids)):
                if total_completion_pointers[i] < self.max_completion_length and not finish_flags[i]:
                    # add the new generated completion_id to the total_completion_ids, add the completion mask to the total_completion_mask
                    left_tk_space = min(self.max_completion_length, total_completion_pointers[i] + eos_idx[i]) - total_completion_pointers[i]
                    total_completion_ids[i, total_completion_pointers[i]:(total_completion_pointers[i] + left_tk_space)] = completion_ids[i, :left_tk_space]
                    total_completion_mask[i, total_completion_pointers[i]:(total_completion_pointers[i] + left_tk_space)] = completion_mask[i, :left_tk_space]
                    total_completion_pointers[i] += eos_idx[i]
                    completion = self.processing_class.decode(completion_ids[i][:eos_idx[i]], skip_special_tokens=True)
                    if 'assistant' != inputs_tobe_updated_each_turn[i]['message'][-1]['role']: # we need to use the chat format because the model can easily process the images
                        inputs_tobe_updated_each_turn[i]['message'].append({'role': 'assistant', 'content': []})
                    inputs_tobe_updated_each_turn[i]['message'][-1]['content'] += [{"type": "text", "text": completion}]
                    print(i, '_', j, completion)
                    # Call tool and set mask for each sample in a batch after one rollout
                

                    tool_call_query = self.parse_grounding(completion)

                    image_with_bbox = None
                    response = ''
                    if tool_call_query is None:
                        finish_flags[i] = True
                    else:
                        # call the tool and log the output during evaluation
                        if self.args.log_completions and 'eval' in inputs[i] and inputs[i]['eval']:
                            # convert image_inputs[i][-1] to base64
                            buffered = io.BytesIO()
                            image_inputs[i][-1].save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                            try:
                                
                                image_base64, response = self.tools[self.tool_func](img_str = img_str, tool_call_query = tool_call_query, normalized_bboxs = True if 'internvl' in self.model_id else False) #self.tool_processor, self.tool_model,

                                image_with_bbox = base64.b64decode(image_base64)
                                image_with_bbox = Image.open(io.BytesIO(image_with_bbox))
                                if 'crop' in self.args.setting:
                                    bbox = re.search(r'\d+, \d+, \d+, \d+', response)[0]
                                    bbox = [int(x) for x in bbox.split(', ')]
                                    image_with_bbox = image_with_bbox.crop(bbox)

                            except Exception as error:
                                if response == '':
                                    response = f"Failure: {str(error)}"


                    temp_image_list = None
                    if image_with_bbox: # save image to local_temp folder
                        # Turn off debug image saving
                        # time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                        # temp_image_name = time_stamp + str(random.randint(0, 1000)) + '.png'
                        # temp_image_path = f'./local_temp_imgs_{self.args.setting}/{temp_image_name}'
                        # if not os.path.exists(f'./local_temp_imgs_{self.args.setting}'):
                        #     os.makedirs(f'./local_temp_imgs_{self.args.setting}', exist_ok=True)
                            
                        # image_with_bbox.save(temp_image_path) 
                        
                        
                        grounded_images.append(image_with_bbox)
                    else:
                        grounded_images.append(None)

                                          
                    print("tool response: ",response)

        # Manually add EOS tokens
        for i in range(len(completion_ids)):
            if total_completion_pointers[i] < self.max_completion_length:
                total_completion_ids[i, total_completion_pointers[i]] = self.eos_token_id
                total_completion_mask[i, total_completion_pointers[i]] = 1
                total_completion_pointers[i] += 1

        completion_ids = total_completion_ids
        completion_mask = total_completion_mask # this is the "loss computing mask" with the tool content zeroed out
        
        # Concatenate prompt_mask with completion_mask for logit computation
        is_eos = completion_ids == self.eos_token_id
        is_pad = completion_ids == self.pad_token_id
        is_eos = is_eos | is_pad
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask_no_tool = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([original_prompt_mask, completion_mask_no_tool], dim=1)  # (B*G, P+C)
        prompt_completion_ids = torch.cat([original_prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens, this is initially the max completion length but will extend as using tools
        
        if is_train:
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, latest_pixel_values, latest_image_grid_thw, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, latest_pixel_values, latest_image_grid_thw, logits_to_keep
                        )
        else:
            ref_per_token_logps = None
                        
        time_taken = time.time() - start_time
        print(f"\n[Timer]: \nTime taken for tool call and logit probability computation: {time_taken:.2f} seconds.\n")

        
        start_time = time.time()
        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(original_prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):  
            reward_start_time = time.time()
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(original_prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(original_prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "message" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["message", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                if "internvl" in self.model_id.lower():
                    reward_kwargs["normalized_bboxs"] = True
                output_reward_func = reward_func(prompts=original_prompts, completions=completions, completion_ids = completion_ids, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            
            reward_time = time.time() - reward_start_time
            print(f"\n[Timer]: \nTime taken for reward function {reward_func.__name__}: {reward_time:.2f} seconds.\n")

        current_batch_rewards = rewards_per_func.clone()
        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes when the number of generations is larger than the batch size per device
        rewards_per_func = gather(rewards_per_func)
        
        
        idx_gpt_reward = None

        # get grounded_region_specific_thinking_format_reward if exists
        idx_grounded_region_specific_thinking_format_reward = None
        idx_grounded_region_bbox_IOU_loss = None
        idx_grounded_region_bbox_repetitive_loss = None
        idx_answer_format_reward = None
        idx_gpt_reward = None
        
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            if "grounded_region_specific_thinking_format_reward" in reward_func_name:
                idx_grounded_region_specific_thinking_format_reward = i
            elif 'gpt' in reward_func_name:
                idx_gpt_reward = i
            elif 'grounded_region_bbox_IOU_loss' in reward_func_name:
                idx_grounded_region_bbox_IOU_loss = i
            elif 'grounded_region_bbox_repetitive_loss' in reward_func_name:
                idx_grounded_region_bbox_repetitive_loss = i
            elif 'answer_format' in reward_func_name:
                idx_answer_format_reward = i
            elif 'gpt' in reward_func_name:
                idx_gpt_reward = i

                
        # make the gpt reward binary
        if idx_gpt_reward is not None:
            rewards_per_func[:, idx_gpt_reward] = torch.where(rewards_per_func[:, idx_gpt_reward] > 0, 1.0, 0.0)

        if idx_grounded_region_bbox_IOU_loss is not None:
            # make the bbox IOU loss excluded from training loss but just a evaluation metrics
            rewards_per_func[:, idx_grounded_region_bbox_IOU_loss] = 0 *  rewards_per_func[:, idx_grounded_region_bbox_IOU_loss]

        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(original_prompts),
            (self.accelerator.process_index + 1) * len(original_prompts),
        )
        advantages = advantages[process_slice]
        

        # Log the metrics
        reward_per_func = current_batch_rewards.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"{inputs[0]['dataset']}/rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[f"{inputs[0]['dataset']}/reward"].append(rewards.mean().item())
        self._metrics[f"{inputs[0]['dataset']}/reward_std"].append(std_grouped_rewards.mean().item())

        time_taken = time.time() - start_time
        print(f"\n[Timer]: \nTime taken for reward computation: {time_taken:.2f} seconds.\n")


        return_content = {
            "step": str(self.state.global_step),
            "input_output_text": inputs_tobe_updated_each_turn,
            "reward_name": [reward_func_name.__name__ for reward_func_name in self.reward_funcs],
            "reward_list": current_batch_rewards.tolist(),
            "raw_inputs": inputs,
            "image_inputs": image_inputs,
            "grounded_images": grounded_images,
            "prompt_ids": original_prompt_ids,
            "prompt_mask": original_prompt_mask,
            'pixel_values': latest_pixel_values, 
            'image_grid_thw': latest_image_grid_thw,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
    
        return return_content


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        start_time = time.time()
        
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        pixel_values, image_grid_thw = inputs['pixel_values'], inputs['image_grid_thw']
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        
        is_eos = completion_ids == self.processing_class.eos_token_id
        is_pad = completion_ids == self.processing_class.pad_token_id
        is_eos = is_eos | is_pad
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.accelerator.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=self.accelerator.device).expand(is_eos.size(0), -1)
        completion_mask_no_tool = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, completion_mask_no_tool], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, pixel_values, image_grid_thw, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)


        def find_sublist_indices(full_list_of_list, sublist, get_end_id = False):
            sub_len = len(sublist)
            matched_indices = []
            for full_list in full_list_of_list:
                for i in range(len(full_list) - sub_len + 1):
                    if full_list[i:i + sub_len] == sublist:
                        if get_end_id:
                            matched_indices.append(i + sub_len - 1)
                        else:
                            matched_indices.append(i)
                        break
                else:
                    matched_indices.append(9999)
            return matched_indices
        
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        time_taken = time.time() - start_time
        print(f"\n[Timer]: \nTime taken for loss computation: {time_taken:.2f} seconds.\n")
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        
        new_inputs = []
        # remove duplicated inputs for faster evaluation
        existed_inputs_image_answer_question = []
        for i in range(len(inputs)):
            inputs_image_answer_question = inputs[i]['image'] + inputs[i]['gt_answer'] + inputs[i]['message'][0]['content'][-1]['text']
            if inputs_image_answer_question not in existed_inputs_image_answer_question:
                existed_inputs_image_answer_question.append(inputs_image_answer_question)
                new_inputs.append(inputs[i])
        inputs = new_inputs
                
        
        for i in range(len(inputs)):
            inputs[i]['eval'] = True
        inputs = self._prepare_inputs(inputs, is_train=False)
        # remove images in the input_output_text, making it pure text
        for j in range(len(inputs['input_output_text'])):
            for i in range(len(inputs['input_output_text'][j]['message'][1]['content'])):
                if 'type' in inputs['input_output_text'][j]['message'][1]['content'][i] and inputs['input_output_text'][j]['message'][1]['content'][i]['type'] == 'image':
                    inputs['input_output_text'][j]['message'][1]['content'][i]['image'] = 'New image with bounding box.'
                
        
        context_images = [inputs['image_inputs'][i] + ([inputs['grounded_images'][i]] if inputs['grounded_images'][i] else []) for i in range(len(inputs['image_inputs']))]
        
        if self.accelerator.is_main_process and random.random() < 0.3: # random sample a few images to log on wandb
                        
            def stitch_images_vertically(images):   
                scale_factor=0.2
                # Resize images to 1/5 of their original size
                resized_images = [img.resize((int(img.width * scale_factor), int(img.height * scale_factor))) for img in images]

                # Get total height and max width after resizing
                max_width = max(img.width for img in resized_images)
                total_height = sum(img.height for img in resized_images)

                # Create new blank image
                stitched_image = Image.new("RGB", (max_width, total_height))

                # Paste images one below the other
                y_offset = 0
                for img in resized_images:
                    stitched_image.paste(img, (0, y_offset))
                    y_offset += img.height


                return stitched_image
            # log val results on wandb
            columns = ["image", "full_cov", "ground_truth", "reward"]
            data = []
            image_list = context_images[0]
            image = stitch_images_vertically(image_list)
            
            wandb_image = wandb.Image(image) 

            wandb_full_cov = json.dumps(inputs['input_output_text'][0], indent=4)
            wandb_gt = inputs['raw_inputs'][0]['gt_answer']
            wandb_reward = inputs['reward_list'][0]

            data.append([wandb_image, wandb_full_cov, wandb_gt, wandb_reward])
            
            table = wandb.Table(columns=columns, data=data)
            wandb.log({f"global_step_{inputs['step']}_eval_{inputs['raw_inputs'][0]['image'].split('/')[-1][:10].split('.')[0]}": table})
        
        local_log_path = self.args.output_dir + f"/local_log_step_{inputs['step']}/evaluation_results_{self.accelerator.process_index}.json"
        local_log_image_path = self.args.output_dir + f"/local_log_step_{inputs['step']}/evaluation_images/"
        os.makedirs(local_log_image_path, exist_ok=True)
            
        existing_logs = []
        if os.path.exists(local_log_path):
            with open(local_log_path, 'r') as f:
                existing_logs = json.load(f)
        
        for i in range(len(inputs['input_output_text'])):
            conversation_images_per_conv = []
            image_name_prefix = inputs['raw_inputs'][i]['image'].split('/')[-1][:10].split('.')[0]  + f"_{str(random.randint(0, 1000))}"
            for j in range(len(context_images[i])):
                output_image_path = os.path.join(local_log_image_path, f"{image_name_prefix}_{j}.png")
                context_images[i][j].save(output_image_path) # temporarily turned off
                conversation_images_per_conv.append(output_image_path)
            
            existing_logs.append(
                                    {
                                        "input_output_conv":inputs['input_output_text'][i],
                                        "conversation_images": conversation_images_per_conv,
                                        "reward_name": inputs['reward_name'],
                                        "reward_list": str(inputs['reward_list'][i]),
                                    }
                                 )
        json.dump(existing_logs, open(local_log_path, 'w'))
        if self.is_train:
            
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs)
                loss = loss.mean().detach()
            return loss, None, None
        else:
            return torch.tensor(0.0, device=self.accelerator.device), None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    # def create_model_card(
    #     self,
    #     model_name: Optional[str] = None,
    #     dataset_name: Optional[str] = None,
    #     tags: Union[str, list[str], None] = None,
    # ):
    #     """
    #     Creates a draft of a model card using the information available to the `Trainer`.

    #     Args:
    #         model_name (`str` or `None`, *optional*, defaults to `None`):
    #             Name of the model.
    #         dataset_name (`str` or `None`, *optional*, defaults to `None`):
    #             Name of the dataset used for training.
    #         tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
    #             Tags to be associated with the model card.
    #     """
    #     if not self.is_world_process_zero():
    #         return

    #     if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
    #         base_model = self.model.config._name_or_path
    #     else:
    #         base_model = None

    #     tags = tags or []
    #     if isinstance(tags, str):
    #         tags = [tags]

    #     if hasattr(self.model.config, "unsloth_version"):
    #         tags.append("unsloth")

    #     citation = textwrap.dedent(
    #         """\
    #         @article{zhihong2024deepseekmath,
    #             title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
    #             author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
    #             year         = 2024,
    #             eprint       = {arXiv:2402.03300},
    #         }
    #         """
    #     )

    #     model_card = generate_model_card(
    #         base_model=base_model,
    #         model_name=model_name,
    #         hub_model_id=self.hub_model_id,
    #         dataset_name=dataset_name,
    #         tags=tags,
    #         wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
    #         comet_url=get_comet_experiment_url(),
    #         trainer_name="GRPO",
    #         trainer_citation=citation,
    #         paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
    #         paper_id="2402.03300",
    #     )

    #     model_card.save(os.path.join(self.args.output_dir, "README.md"))
