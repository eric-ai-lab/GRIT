
# for internvl3
from internvl.model.internvl_chat.configuration_intern_vit import InternVisionConfig
from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
from internvl.model.internvl_chat.modeling_intern_vit import InternVisionModel
from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from internvl.conversation import get_conv_template
from internvl.train.dataset import build_transform, dynamic_preprocess
from typing import Dict, Literal, Optional
from copy import deepcopy
from transformers.trainer_pt_utils import LabelSmoother
import transformers
import torch
import numpy as np
from PIL import Image

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
EOS_TOKEN = '<|im_end|>'
END_OF_TEXT_TOKEN = '<|endoftext|>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

def load_image(image_file, is_train, input_size=448, max_num=2):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(is_train, input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, [image]


def preprocess_internvl2_5(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sources[0]

    for role, input_id in zip(roles, input_ids):
        final_input_ids.append(input_id)

    input_ids = torch.tensor(np.concatenate(final_input_ids))[:tokenizer.model_max_length]

    return dict(
        input_ids=input_ids,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
