'''
conda create -n ml_env python=3.8
conda activate ml_env
conda install pytorch torchvision -c pytorch
pip install transformers==4.37.2
pip install Pillow overrides termcolor opencv-python
CUDA_VISIBLE_DEVICES=0,5,6,7 python internvl.py

wei chow@usc implement it ref to the official repo: https://huggingface.co/OpenGVLab/InternVL2-26B
'''
import tempfile
import math
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import torch
from eval.eval_utils.task_evaluator import PhysionBenchEvaluator, test_frame
from termcolor import cprint
from io import BytesIO
import cv2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_frame_from_vcap(vidcap, num_frames=10, fps=None, frame_count=None):
    if fps is None or frame_count is None:
        # Recompute fps and frame_count if not provided
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or frame_count == 0:
        print("Video file not found or has no frames. Returning empty images.")
        return [Image.new("RGB", (720, 720))] * num_frames

    duration = frame_count / fps
    frame_interval = frame_count // num_frames

    if frame_interval == 0 and frame_count <= 1:
        print("Frame interval is equal to 0. Returning empty image.")
        return [Image.new("RGB", (720, 720))] * num_frames

    images = []
    count = 0
    success = True
    frame_indices = np.linspace(0, min(frame_count, num_frames) - 1, min(frame_count, num_frames), dtype=int)

    while success:
        success, frame = vidcap.read()
        if success and count in frame_indices:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            images.append(im_pil)
            if len(images) >= min(frame_count, num_frames):
                return images
        count += 1

    if len(images) < num_frames:
        width, height = images[0].size if images else (720, 720)
        padding_frames = [Image.new("RGB", (width, height))] * (num_frames - len(images))
        images.extend(padding_frames)
        print(f"Video has fewer frames than requested. Padding with empty frames: {len(padding_frames)}")

    return images

def opencv_extract_frames_o(vpath_or_bytesio, frames=6, fps=None, frame_count=None):
    """
    Extract frames from a video using OpenCV.

    Args:
        vpath_or_bytesio (str or BytesIO): Path to the video file or BytesIO object containing the video.
        frames (int): Number of frames to extract from the video.

    Returns:
        list: List of PIL Images extracted from the video.

    Raises:
        NotImplementedError: If the type of `vpath_or_bytesio` is not supported.
    """
    import cv2

    if isinstance(vpath_or_bytesio, str):
        vidcap = cv2.VideoCapture(vpath_or_bytesio)
        return get_frame_from_vcap(vidcap, frames, fps=fps, frame_count=frame_count)
    elif isinstance(vpath_or_bytesio, (BytesIO,)):
        # assuming mp4
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
            temp_video.write(vpath_or_bytesio.read())
            temp_video_name = temp_video.name
            vidcap = cv2.VideoCapture(temp_video_name)
            return get_frame_from_vcap(vidcap, frames, fps=fps, frame_count=frame_count)
    else:
        raise NotImplementedError(type(vpath_or_bytesio))

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def split_model_25(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class InternVLChat2():
    def __init__(self, ckpt="OpenGVLab/InternVL2-1B"):
        if '2_5' in ckpt:
            device_map = split_model_25(ckpt.split('/')[-1])  # note: this may need to be changed
        else:
            device_map = split_model(ckpt.split('/')[-1])  # note: this may need to be changed

        print(device_map)
        self.model = AutoModel.from_pretrained(
            ckpt,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.num_frames = test_frame

    def qa(self, image, prompt, mode):
        pixel_values_list = []
        num_patches_list = []
        for img in image:
            pv = load_image(img, max_num=1).to(torch.bfloat16).cuda()  # max_num 表示拆分时每张图片最多可以拆成多少个子图
            pixel_values_list.append(pv)
            num_patches_list.append(pv.size(0))
        pixel_values = torch.cat(pixel_values_list, dim=0)
        # prompt = prompt.replace("<video>", "<image>" * video_token_num)  # for seq (for small model, like 1B)
        prompt = prompt.replace("<video>", "<image>")  # for combine (for big model like 26B and the bigger one)
        response = self.model.chat(self.tokenizer, pixel_values, prompt,
                                       dict(max_new_tokens=10, do_sample=False),
                                       num_patches_list=num_patches_list)

        print('response')
        cprint(response, 'cyan')
        return response

if __name__ == "__main__":
    dataset_path="<your path>"               # todo@physbench step1: download dataset
    model_name = "OpenGVLab/InternVL2_5-78B"

    model= InternVLChat2(ckpt=model_name)                        # todo@physbench step2: implement your model class with method <def qa(self, image, prompt, mode)>



    task_evaluator = PhysionBenchEvaluator(
        model=model,
        mode='general',   # todo@physbench step3: choose your mode in ["image-only", "image&video", "general"]
        dataset_path=dataset_path,
        model_name=model_name,
        resume=True,
        sample_ratio=None,
        split='test'
    )

    # todo@physbench step4: add the model_name in test() function, just like OpenGVLab/InternVL2_5-78B
    task_evaluator.test()
