import mlx.core as mx
import numpy as np
import torch
from transformers import AutoProcessor
from transformers.image_utils import load_image
from transformers.processing_utils import ChatTemplateLoadKwargs
from transformers.video_utils import load_video


class ProcessorWrapper:
    """
    A wrapper that combines an HF processor and custom processing steps.
    For multimodal model path, self._processor is usually inherits from ProcessorMixin.
    For text only model path, self._processor is a `AutoTokenizer`.
    """

    def __init__(self, model_path: str):
        self._processor = AutoProcessor.from_pretrained(model_path)
        self.apply_chat_template = self._processor.apply_chat_template

    @property
    def processor(self):
        return self._processor

    def __call__(self, *args, **kwargs):
        return self._processor(*args, **kwargs)

    def load_multimodalities(self, conversation, **kwargs):
        """
        Based on transformers.processing_utils.apply_chat_template.
        Audio is skipped for now.
        """
        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        mm_load_kwargs = ChatTemplateLoadKwargs(**kwargs)

        batch_images, batch_videos = [], []
        batch_video_metadata = []
        for conversation in conversations:
            images, videos = [], []
            video_metadata = []
            for message in conversation:
                visuals = [content for content in message["content"] if content["type"] in ["image", "video"]]
                image_fnames = [
                    vision_info[key]
                    for vision_info in visuals
                    for key in ["image", "url", "path", "base64"]
                    if key in vision_info and vision_info["type"] == "image"
                ]
                video_fnames = [
                    vision_info[key]
                    for vision_info in visuals
                    for key in ["video", "url", "path"]
                    if key in vision_info and vision_info["type"] == "video"
                ]

                for fname in image_fnames:
                    images.append(load_image(fname))

                for fname in video_fnames:
                    if isinstance(fname, (list, tuple)) and isinstance(fname[0], str):
                        # Case a: Video is provided as a list of image file names
                        video = [np.array(load_image(image_fname)) for image_fname in fname]
                        video = np.stack(video)
                        metadata = None
                    else:
                        # Case b: Video is provided as a single file path or URL or decoded frames in a np.ndarray or torch.tensor
                        video, metadata = load_video(
                            fname,
                            backend=mm_load_kwargs["video_load_backend"],
                        )
                    videos.append(video)
                    video_metadata.append(metadata)

            # Currently all processors can accept nested list of batches, but not flat list of visuals
            # So we'll make a batched list of images and let the processor handle it
            if images:
                batch_images.append(images)
            if videos:
                batch_videos.append(videos)
                batch_video_metadata.append(video_metadata)
        
        if not is_batched:
            if images:
                batch_images = batch_images[0]
            if videos:
                batch_videos = batch_videos[0]
                batch_video_metadata = batch_video_metadata[0]

        return {
            "images": batch_images,
            "videos": batch_videos,
            "video_meta": batch_video_metadata,
        }

    @staticmethod
    def convert_to_mx_array(np_array_dict):
        mx_array_dict = {}
        for k, v in np_array_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.numpy()
            mx_array_dict[k] = mx.array(v)
        return mx_array_dict
        
