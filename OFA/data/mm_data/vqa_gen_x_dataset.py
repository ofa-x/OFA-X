# Copyright 2022 The OFA-Sys Team and OFA-X team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    conf = None
    if samples[0].get("conf", None) is not None:
        conf = torch.cat([s['conf'] for s in samples], dim=0)

    ref_dict = None
    if samples[0].get("ref_dict", None) is not None:
        ref_dict = np.array([s['ref_dict'] for s in samples])

    constraint_masks = None
    if samples[0].get("constraint_mask", None) is not None:
        constraint_masks = merge("constraint_mask")

    decoder_prompts = None
    if samples[0].get("decoder_prompt", None) is not None:
        decoder_prompts = np.array([s['decoder_prompt'].tolist() for s in samples])

    prefix_tokens = None
    if samples[0].get("decoder_prompt", None) is not None:
        prefix_tokens = merge("decoder_prompt")

    explanations = None
    if samples[0].get("explanation", None) is not None:
        explanations = merge("explanation")

    answers = None
    if samples[0].get("answer", None) is not None:
        answers = merge("answer")

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        )
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "conf": conf,
        "ref_dict": ref_dict,
        "constraint_masks": constraint_masks,
        "decoder_prompts": decoder_prompts,
        "target": target,
        "prefix_tokens": prefix_tokens,
        "explanations": explanations,
        "answers": answers,
    }

    return batch


class VqaGenXDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_object_length=30,
        max_tgt_length=30,
        patch_image_size=224,
        add_object=False,
        imagenet_default_mean_and_std=False,
        prompt_type="none"
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_object_length = max_object_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size

        self.add_object = add_object
        self.prompt_type = prompt_type

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        item = self.dataset[index]
        if len(item) == 5:
            uniq_id, image, question, ref, explanation = item
        else:
            uniq_id, image, question, ref, explanation, caption = item

        # Process image
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        # Process question
        question = self.pre_question(question, self.max_src_length)
        question = question + '?' if not question.endswith('?') else question
        src_item = self.encode_text(' {}'.format(question))

        # Process answer
        ref_dict = {item.split('|!+')[1]: float(item.split('|!+')[0]) for item in ref.split('&&')}
        answer = max(ref_dict, key=ref_dict.get)
        conf = torch.tensor([ref_dict[answer]])
        answer = f" {answer}"
        ans_target_item = self.encode_text(answer)

        # Process explanation
        explanation = f" because {explanation}"
        expl_target_item = self.encode_text(explanation)

        # Combine as '{bos} [question] the answer is [answer] because [explanation] {eos}' following NLX-GPT
        if self.prompt_type == 'prev_output':
            # Prev output item is for teacher forcing
            # includes < bos question answer expl>
            prev_output_item = torch.cat([self.bos_item, src_item, ans_target_item, expl_target_item])

            # Pad items of length of src_item (without bos)
            pad_values = torch.full(tuple([src_item.size(0)]), self.tgt_dict.pad())

            # Target item includes < question answer expl eos >
            target_item = torch.cat([pad_values, ans_target_item, expl_target_item, self.eos_item])
            decoder_prompt = torch.cat([self.bos_item, src_item])

            # prev_output_item = bos q1 q2 q3 q4 t1 t2 t3 t4
            # prompt = Frage = bos q1 q2 q3 q4
            # target = pad1 pad2 pad3 pad4 t1 t2 t3 t4 eos
        elif self.prompt_type == 'without_decoder_prompt':
            # Prev output item is for teacher forcing
            # includes < bos answer expl>
            prev_output_item = torch.cat([self.bos_item, ans_target_item, expl_target_item])

            # Target item includes < question answer expl eos >
            target_item = torch.cat([ans_target_item, expl_target_item, self.eos_item])
            decoder_prompt = torch.cat([self.bos_item])
        else:
            raise NotImplementedError

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "decoder_prompt": decoder_prompt,
            "ref_dict": ref_dict,
            "answer": ans_target_item,
            "explanation": expl_target_item,
            "conf": conf,
        }
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
