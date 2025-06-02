# refer to https://github.com/JieyuZ2/TaskMeAnything/blob/main/tma/models/qa_model/imageqa_model.py
import base64
import tempfile
import time
from typing import Callable, Union
import openai
import torch
import random
import os
from PIL import Image
from torch.nn.parallel import DataParallel
from .base_qa_model import QAModel, QAModelInstance
from termcolor import cprint
from io import BytesIO
from eval.eval_utils.task_evaluator import test_frame
import cv2
import av
import numpy as np

imageqa_models = {
    "instructblip-flan-t5-xl"              : ("InstructBlip", 	 "Salesforce/instructblip-flan-t5-xl"),
    "instructblip-flan-t5-xxl"             : ("InstructBlip", 	 "Salesforce/instructblip-flan-t5-xxl"),
    "instructblip-vicuna-7b"               : ("InstructBlip", 	 "Salesforce/instructblip-vicuna-7b"),
    "instructblip-vicuna-13b"              : ("InstructBlip", 	 "Salesforce/instructblip-vicuna-13b"),
    "blip2-flant5xxl"                      : ("BLIP2", 		  	 "Salesforce/blip2-flan-t5-xxl"),
    "llava-1.5-7b-hf"                      : ("LLaVA", 		  	 "llava-hf/llava-1.5-7b-hf"),
    "llava-1.5-13b-hf"                     : ("LLaVA", 		  	 "llava-hf/llava-1.5-13b-hf"),
    "llava-v1.6-mistral-7b-hf"             : ("LLaVA", 			 "llava-hf/llava-v1.6-mistral-7b-hf"),
    "llava-v1.6-vicuna-7b-hf"              : ("LLaVA", 			 "llava-hf/llava-v1.6-vicuna-7b-hf"),
	"deepseek1B"						   : ("DeepSeekVL", 	 "deepseek-ai/deepseek-vl-1.3b-chat"),  # pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git
	"deepseek7B"						   : ("DeepSeekVL", 	 "deepseek-ai/deepseek-vl-7b-chat"),
	"Xinyuan-VL-2B"            			   : ("XinyuanVL", 		 "Cylingo/Xinyuan-VL-2B"),  # pip install qwen-vl-utils # pip install git+https://github.com/huggingface/transformers
	"Aquila-VL-2B"			   			   : ("AquilaVL", 		 "BAAI/Aquila-VL-2B-llava-qwen"),  # # pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
    "Phi-3-vision-128k-instruct"           : ("Phi", 			 "microsoft/Phi-3-vision-128k-instruct"),
	"Phi-3.5V"           				   : ("Phi", 			 "microsoft/Phi-3.5-vision-instruct"),
	"mPLUG-Owl3-1B-241014"				   : ("Owl3", 			 "mPLUG/mPLUG-Owl3-1B-241014"), # pip install icecream torch==2.1.0
	"mPLUG-Owl3-2B-241014"				   : ("Owl3", 			 "mPLUG/mPLUG-Owl3-2B-241014"),
	"mPLUG-Owl3-7B-241101"				   : ("Owl3", 			 "mPLUG/mPLUG-Owl3-7B-241101"),
	"MiniCPM-V2"						   : ("MiniCPMV", 		 "openbmb/MiniCPM-V-2"),
	"MiniCPM-V2.5"						   : ("MiniCPMV", 		 "openbmb/MiniCPM-Llama3-V-2_5"),
	"MiniCPM-V2.6"					       : ("MiniCPMV", 		 "openbmb/MiniCPM-V-2_6"),
    "Qwen-VL-Chat"                         : ("QwenVLChat", 	 "Qwen/Qwen-VL-Chat"),
    "InternVL-Chat-V1-5-quantable"         : ("InternVLChat", 	 'failspy/InternVL-Chat-V1-5-quantable'),
    "llava-interleave-qwen-7b-hf"          : ("LLaVAInterleave", "llava-hf/llava-interleave-qwen-7b-hf"),
    "llava-interleave-qwen-7b-dpo-hf"      : ("LLaVAInterleave", "llava-hf/llava-interleave-qwen-7b-dpo-hf"),
    "vila-1.5-3b"                          : ("VILAModel",  	 "Efficient-Large-Model/VILA1.5-3b"),
    "vila-1.5-3b-s2"                       : ("VILAModel", 		 "Efficient-Large-Model/VILA1.5-3b-s2"),
    "vila-1.5-8b"                          : ("VILAModel", 		 "Efficient-Large-Model/Llama-3-VILA1.5-8B"),
    "vila-1.5-13b"                         : ("VILAModel",       "Efficient-Large-Model/VILA1.5-13b"),
    "cambrian-8b"                          : ("Cambrian", 		 "nyu-visionx/cambrian-8b"),
	"paligemma2-3b"						   : ("PaliGemma2", 	 "google/paligemma2-3b-ft-docci-448"),
	"paligemma2-10b"					   : ("PaliGemma2", 	 "google/paligemma2-10b-ft-docci-448"),
    "LLaVA-NeXT-Video-7B-DPO-hf"           : ("LLaVAVideo", 	 "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf"),
    "LLaVA-NeXT-Video-7B-hf"               : ("LLaVAVideo", 	 "llava-hf/LLaVA-NeXT-Video-7B-hf"),
	"MolmoE-1B"							   : ("MolmoE", 		 "allenai/MolmoE-1B-0924"),
	"MolmoE-7B-O"						   : ("MolmoE", 		 "allenai/Molmo-7B-O-0924"),
	"MolmoE-7B-D"						   : ("MolmoE", 		 "allenai/Molmo-7B-D-0924"),
	"InternVL2-1B"						   : ("InternVLChat2",   'OpenGVLab/InternVL2-1B'),
	"InternVL2-2B"						   : ("InternVLChat2",   'OpenGVLab/InternVL2-2B'),
	"InternVL2-4B"						   : ("InternVLChat2",   'OpenGVLab/InternVL2-4B'),
	"InternVL2-8B"						   : ("InternVLChat2",   'OpenGVLab/InternVL2-8B'),
	"InternVL2-26B"						   : ("InternVLChat2",   'OpenGVLab/InternVL2-26B'),
	"InternVL2-40B"						   : ("InternVLChat2",   'OpenGVLab/InternVL2-40B'),
	"InternVL2-76B"					       : ("InternVLChat2",   'OpenGVLab/InternVL2-Llama3-76B'),
	"InternVL2_5-1B"					   : ("InternVLChat2",   'OpenGVLab/InternVL2_5-1B'),
	"InternVL2_5-2B"					   : ("InternVLChat2",   'OpenGVLab/InternVL2_5-2B'),
	"InternVL2_5-4B"					   : ("InternVLChat2",   'OpenGVLab/InternVL2_5-4B'),
	"InternVL2_5-8B"					   : ("InternVLChat2",   'OpenGVLab/InternVL2_5-8B'),
	"InternVL2_5-26B"					   : ("InternVLChat2",   'OpenGVLab/InternVL2_5-26B'),
	"InternVL2_5-38B"				       : ("InternVLChat2",   'OpenGVLab/InternVL2_5-38B'),
	"InternVL2_5-78B"		   			   : ("InternVLChat2",   'OpenGVLab/InternVL2_5-78B'),
    "Mantis-8B-Idefics2"                   : ("Mantis", 		 "TIGER-Lab/Mantis-8B-Idefics2"),  # pip install git+https://github.com/TIGER-AI-Lab/Mantis.git
    "Mantis-llava-7b"                      : ("Mantis", 		 "TIGER-Lab/Mantis-llava-7b"),	   # pip install git+https://github.com/TIGER-AI-Lab/Mantis.git
    "Mantis-8B-siglip-llama3"              : ("Mantis", 		 "TIGER-Lab/Mantis-8B-siglip-llama3"), # pip install git+https://github.com/TIGER-AI-Lab/Mantis.git
    "Mantis-8B-clip-llama3"                : ("Mantis", 		 "TIGER-Lab/Mantis-8B-clip-llama3"),   # pip install git+https://github.com/TIGER-AI-Lab/Mantis.git
    "gpt4v"                                : ("GPT4V", 			 "<Your GPT Key>"),
    "gpt4o"                                : ("GPT4O", 			 "<Your GPT Key>"),
	"o1"								   : ("O1", 			 "<Your GPT Key>"),
    "gpt4o-mini"                           : ("GPT4Omini", 		 "<Your GPT Key>"),
    "gemini-1.5-flash"                     : ("GeminiFlash", 	 "<Your Gemini Key>"),
    "gemini-1.5-pro"                       : ("GeminiPro", 		 "<Your Gemini Key>"),
    "claude-3-5-sonnet"                    : ("Claude_sonnetx",  "<Your Claude Key>"),
    "claude-3-sonnet"                      : ("Claude_sonnet",   "<Your Claude Key>"),
    "claude-3-opus"                        : ("Claude_opus",     "<Your Claude Key>"),
    "claude-3-haiku"                       : ("Claude_haiku",    "<Your Claude Key>"),
}


# Refer to https://github.com/NVlabs/VILA
def get_frame_from_vcap(vidcap, num_frames=10, fps=None, frame_count=None):
    import cv2
    from PIL import Image
    import numpy as np

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

def set_imageqa_model_key(model_name, key):
	imageqa_models[model_name] = (imageqa_models[model_name][0], key)


def list_imageqa_models():
	return list(imageqa_models.keys())


def image_to_base64(pil_image):
	import io
	import base64
	img_byte_arr = io.BytesIO()
	pil_image.save(img_byte_arr, format='PNG')
	img_byte_arr = img_byte_arr.getvalue()
	base64_str = base64.b64encode(img_byte_arr).decode('utf-8')
	return base64_str


class ImageQAModel(QAModel):
	def __init__(
			self,
			model_name: str,
			prompt_name: str = None,
			prompt_func: Callable = None,
			model: QAModelInstance = None,
			torch_device: Union[int, str] = -1,
			precision=torch.bfloat16,
			choice_format='letter',
			enable_choice_search: bool = False,
			cache_path: str = None,

	):
		super().__init__(model_name, prompt_name, prompt_func, choice_format, enable_choice_search, cache_path)

		if isinstance(torch_device, str):
			torch_device = torch.device(torch_device)
		else:
			if torch_device == -1:
				torch_device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
			else:
				torch_device = torch.device(f"cuda:{torch_device}")

		if model is None:
			print(f"Loading {model_name}...")
			class_name, ckpt = imageqa_models[model_name]
			self.model_precision = precision
			self.model = eval(class_name)(ckpt, torch_device, self.model_precision)
			print(f"Finish loading {model_name}")
		else:
			print(f"Using provided model...")
			self.model = model

	def _data_to_str(self, data):
		if isinstance(data, str):
			return data
		else:
			return image_to_base64(data)


class BLIP2(QAModelInstance):
	def __init__(self, ckpt="Salesforce/blip2-flan-t5-xxl", torch_device=torch.device("cuda"), model_precision=torch.float32):
		from transformers import Blip2Processor, Blip2ForConditionalGeneration
		self.processor = Blip2Processor.from_pretrained(ckpt, device_map=torch_device)
		self.model = Blip2ForConditionalGeneration.from_pretrained(
			ckpt,
			device_map=torch_device,
			torch_dtype=model_precision,
			low_cpu_mem_usage=True,
		).eval()

	def qa(self, image, prompt):
		if isinstance(image, str):
			image = Image.open(image).convert('RGB')
		inputs = self.processor(image, prompt, return_tensors="pt").to(self.model.device)
		out = self.model.generate(**inputs, max_new_tokens=200)
		answer = self.processor.decode(out[0], skip_special_tokens=True)
		return answer

class MolmoE(QAModelInstance):
	def __init__(self, ckpt="allenai/MolmoE-1B-0924", torch_device=torch.device("cuda"), model_precision=torch.float32):
		# 🔥 only support float32
		# https://huggingface.co/allenai/MolmoE-1B-0924/discussions/12
		# change torch_dtype will lead to error
		model_precision = torch.float32
		from transformers import AutoModelForCausalLM, AutoProcessor
		self.processor = AutoProcessor.from_pretrained(
			ckpt,
			trust_remote_code=True,
			torch_dtype='auto',
			device_map='auto'
		)
		self.model = AutoModelForCausalLM.from_pretrained(
			ckpt,
			trust_remote_code=True,
			torch_dtype='auto',
			device_map='auto'
		).to(model_precision).eval()
		self.model_precision = model_precision

	def qa(self, image, prompt):
		try:
			from transformers import GenerationConfig
			if isinstance(image, str):
				image = Image.open(image).convert('RGB')
			print(prompt)
			inputs = self.processor.process(
				images=[image],
				text= prompt.replace('<image>\n', '')
			)
			# move inputs to the correct device and make a batch of size 1
			# generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
			inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

			output = self.model.generate_from_batch(
				inputs,
				GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
				tokenizer=self.processor.tokenizer
			)

			# only get generated tokens; decode them to text
			generated_tokens = output[0, inputs['input_ids'].size(1):]
			generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

			generated_text= generated_text.replace(' ', '', 1)  # remove the first ' '
			cprint(generated_text, 'cyan')

			return generated_text
		except:
			cprint("nonono", 'cyan')
			return None

class InstructBlip(QAModelInstance):
	def __init__(self, ckpt="Salesforce/instructblip-flan-t5-xxl", torch_device=torch.device("cuda"), model_precision=torch.float32):
		from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, InstructBlipConfig, AutoModelForVision2Seq
		from accelerate import infer_auto_device_map, init_empty_weights
		if ckpt == "Salesforce/instructblip-vicuna-13b":
			# Load the model configuration.
			config = InstructBlipConfig.from_pretrained(ckpt)
			# Initialize the model with the given configuration.
			with init_empty_weights():
				model = AutoModelForVision2Seq.from_config(config)
				model.tie_weights()
			# Infer device map based on the available resources.
			device_map = infer_auto_device_map(model, max_memory={0: "40GiB", 1: "40GiB"},
											no_split_module_classes=['InstructBlipEncoderLayer', 'InstructBlipQFormerLayer', 'LlamaDecoderLayer'])
			device_map['language_model.lm_head'] = device_map['language_projection'] = device_map[('language_model.model.embed_tokens')]
		else:
			device_map = torch_device
		self.processor = InstructBlipProcessor.from_pretrained(ckpt, device_map="auto")
		self.model = InstructBlipForConditionalGeneration.from_pretrained(
			ckpt,
			device_map=device_map,
			torch_dtype=model_precision,
			low_cpu_mem_usage=True,
		).eval()

	def qa(self, image, prompt):
		if isinstance(image, str):
			image = Image.open(image).convert('RGB')
		inputs = self.processor(image, prompt, return_tensors="pt").to(self.model.device)

		out = self.model.generate(**inputs, max_new_tokens=200)
		answer = self.processor.decode(out[0], skip_special_tokens=True)
		cprint(answer, 'cyan')
		return answer


class LLaVA(QAModelInstance):
	def __init__(self, ckpt="llava-hf/llava-1.5-7b-hf", torch_device=torch.device("cuda"), model_precision=torch.float32):
		if ckpt == "llava-hf/llava-v1.6-34b-hf":  # run model on multi gpus
			from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
			model = LlavaNextForConditionalGeneration.from_pretrained(ckpt,
																	torch_dtype=torch.float16,
																	low_cpu_mem_usage=True,
																	load_in_4bit=True,
																	# use_flash_attention_2=True,
																	).to(torch_device).eval()
			self.model = DataParallel(model)
			self.processor = LlavaNextProcessor.from_pretrained(ckpt)
		elif 'llava-v1.6-mistral-7b-hf' in ckpt or 'llama3-llava-next-8b-hf' in ckpt or \
				'llava-v1.6-vicuna-7b-hf' in ckpt:
			from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
			self.model = LlavaNextForConditionalGeneration.from_pretrained(
				ckpt,
				torch_dtype=model_precision,
				low_cpu_mem_usage=True,
			).to(torch_device).eval()
			self.processor = LlavaNextProcessor.from_pretrained(ckpt)
		else:
			from transformers import AutoProcessor, LlavaForConditionalGeneration
			self.model = LlavaForConditionalGeneration.from_pretrained(
				ckpt,
				torch_dtype=model_precision,
				low_cpu_mem_usage=True,
			).to(torch_device).eval()
			self.processor = AutoProcessor.from_pretrained(ckpt)

	def qa(self, image, prompt):
		if isinstance(image, str):
			image = Image.open(image).convert('RGB')

		prompt = "USER:" + prompt + "\nASSISTANT:"
		print(prompt)
		if isinstance(self.model, torch.nn.DataParallel):
			inputs = self.processor(prompt, image, return_tensors='pt').to(next(self.model.parameters()).device)
			out = self.model.module.generate(**inputs, max_new_tokens=200, do_sample=False)
		else:
			inputs = self.processor(prompt, image, return_tensors='pt').to(self.model.device)
			out = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
		answer = self.processor.decode(out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
		cprint(answer, 'cyan')
		return answer

class PaliGemma2(QAModelInstance):
	def __init__(self, ckpt="dgoogle/paligemma2-3b-ft-docci-448", torch_device=torch.device("cuda"), model_precision=torch.float32):
		from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
		self.model = PaliGemmaForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.bfloat16,
																device_map="auto").eval()
		self.processor = PaliGemmaProcessor.from_pretrained(ckpt)

	def qa(self, image, prompt):
		from transformers.image_utils import load_image
		if isinstance(image, Image.Image):
			# Check if the image is a PIL.Image object and save to a temporary file if so
			with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
				image.save(tmp.name)
				image_path = tmp.name
		else:
			image_path = image
		image = load_image(image_path)
		model_inputs = self.processor(text=prompt.replace('<image>\n', '') + 'your answer is: ', images=image, return_tensors="pt")\
			.to(torch.bfloat16).to(self.model.device)
		input_len = model_inputs["input_ids"].shape[-1]

		with torch.inference_mode():
			generation = self.model.generate(**model_inputs, max_new_tokens=5, do_sample=False)
			generation = generation[0][input_len:]
			decoded = self.processor.decode(generation, skip_special_tokens=True)
			cprint(decoded, 'cyan')
			return decoded

class LLaVAVideo(QAModelInstance):
	def __init__(self, ckpt="llava-hf/LLaVA-NeXT-Video-7B-hf", torch_device=torch.device("cuda"), model_precision=torch.float32):
		from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
		model_precision = torch.float16  # only support this
		self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
			ckpt,
			torch_dtype=model_precision,
			low_cpu_mem_usage=True
		).to(torch_device).eval()
		self.model_precision = model_precision
		self.processor = LlavaNextVideoProcessor.from_pretrained(ckpt, use_fast=False)  # for bug: https://github.com/huggingface/transformers/issues/31713
		self.num_frames = test_frame
	def _read_video_pyav(self, container, indices):
		'''
		Decode the video with PyAV decoder.
		Args:
			container (`av.container.input.InputContainer`): PyAV container.
			indices (`List[int]`): List of frame indices to decode.
		Returns:
			result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
		'''
		frames = []
		container.seek(0)
		start_index = indices[0]
		end_index = indices[-1]
		for i, frame in enumerate(container.decode(video=0)):
			if i > end_index:
				break
			if i >= start_index and i in indices:
				frames.append(frame)
		return np.stack([x.to_ndarray(format="rgb24") for x in frames])
	def qa(self, image, prompt, mode):
		clip = None
		imgs = None
		prompt = "USER:" + prompt + "\nASSISTANT:"
		for it in image:
			if it.endswith(".mp4"):
				container = av.open(it)
				total_frames = container.streams.video[0].frames
				indices = np.arange(0, total_frames, total_frames / self.num_frames).astype(int)
				clip = self._read_video_pyav(container, indices)
			else:
				if imgs==None:
					imgs = [Image.open(it).convert("RGB")]
				else:
					imgs.append(Image.open(it).convert("RGB"))

		inputs = self.processor(text=prompt, videos=clip, images=imgs, padding=True, return_tensors="pt").to(self.model.device, self.model_precision)

		# Generate
		try:
			generate_ids = self.model.generate(**inputs, max_new_tokens=10)
			out = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("ASSISTANT:")[-1].strip()
			cprint(out, 'cyan')
		except:
			return None

		del inputs
		return out

class LLaVAInterleave(QAModelInstance):
	def __init__(self, ckpt="llava-hf/llava-interleave-qwen-7b-hf", torch_device=torch.device("cuda"), model_precision=torch.float32):
		from transformers import AutoProcessor, LlavaForConditionalGeneration
		self.model = LlavaForConditionalGeneration.from_pretrained(
			ckpt,
			torch_dtype=model_precision,
			low_cpu_mem_usage=True,
		).to(torch_device).eval()
		self.processor = AutoProcessor.from_pretrained(ckpt)
		self.model_precision = model_precision
		self.num_frames = test_frame
	def qa(self, image, prompt, mode):
		if mode == 'image-only':
			image_pil = Image.open(image[0]).convert('RGB')
			prompt = "USER:" + prompt + "\nASSISTANT:"
			inputs = self.processor(prompt, image_pil, return_tensors='pt').to(self.model.device, self.model_precision)
		elif mode == 'image&video':
			video_frames = opencv_extract_frames_o(image[0], frames=self.num_frames)
			prompt = "USER:" + prompt.replace("<video>", "<image>" * len(video_frames)) + "\nASSISTANT:"
			inputs = self.processor(prompt, video_frames, return_tensors='pt').to(self.model.device, self.model_precision)
		else:  # general
			image_list = []
			video_token_num = 0
			for it in image:
				if it.endswith(".mp4"):
					ori = len(image_list)
					image_list+=opencv_extract_frames_o(it, frames=self.num_frames)
					video_token_num = len(image_list) - ori
				else:
					image_list.append(Image.open(it).convert('RGB'))
			prompt = "USER:" + prompt.replace("<video>", "<image>" * video_token_num) + "\nASSISTANT:"
			inputs = self.processor(prompt, image_list, return_tensors='pt').to(self.model.device, self.model_precision)
		print(prompt)
		output = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
		response = self.processor.decode(output[0][2:], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
		cprint(response, 'cyan')
		return response


class VILAModel(QAModelInstance):
	def __init__(self, ckpt, torch_device=torch.device("cuda"), model_precision=torch.float16, num_video_frames=8):
		from .model_library.vila.mm_utils import get_model_name_from_path
		from .model_library.vila.model.builder import load_pretrained_model
		from .model_library.vila.utils import disable_torch_init

		# VILA often use model_precision == torch.float16
		disable_torch_init()
		self.tokenizer, self.model, self.image_processor, self.context_len = \
			load_pretrained_model(ckpt, get_model_name_from_path(ckpt), None)
		self.model_precision = torch.float16 # vila only support this
		self.num_video_frames = test_frame

	def qa(self, image, prompt, mode):
		from .model_library.vila.constants import IMAGE_TOKEN_INDEX
		from .model_library.vila.conversation import SeparatorStyle, conv_templates
		from .model_library.vila.mm_utils import (KeywordsStoppingCriteria, process_images, tokenizer_image_token)

		images_pil = []
		for v in image:
			if v.endswith(".mp4"):
				images_pil += opencv_extract_frames_o(v, self.num_video_frames)
			else:
				images_pil.append(load_image_o(v))

		conv = conv_templates['vicuna_v1'].copy()
		# conv.system = ''
		conv.append_message(conv.roles[0], prompt)
		conv.append_message(conv.roles[1], None)
		input = conv.get_prompt()
		print(input)
		images_tensor = process_images(images_pil, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
		input_ids = tokenizer_image_token(input, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

		stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
		keywords = [stop_str]
		stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
		with torch.inference_mode():
			output_ids = self.model.generate(
				input_ids,
				images=[images_tensor],
				do_sample=False,  # t>0, need True
				temperature=0.1,
				top_p=None,
				num_beams=1,
				max_new_tokens=4000,
				use_cache=False,
				# stopping_criteria=[stopping_criteria],
			)

		outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
		outputs = outputs.strip()
		if outputs.endswith(stop_str):
			outputs = outputs[: -len(stop_str)]
		outputs = outputs.strip()
		cprint(outputs,'cyan')
		return outputs

class Mantis(QAModelInstance):
	def __init__(self, ckpt, torch_device=torch.device("cuda"), model_precision=torch.float16, num_video_frames=8):
		self.num_frames = test_frame
		if 'Idefics2' in ckpt:
			from transformers import AutoProcessor, AutoModelForVision2Seq
			self.processor = AutoProcessor.from_pretrained(ckpt)
			self.model = AutoModelForVision2Seq.from_pretrained(ckpt,torch_dtype=torch.bfloat16).to(torch_device)
			self.num_frames = self.num_frames - 2 # for GPU mem
		elif 'llava-7b' in ckpt or 'llama3' in ckpt:
			from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
			self.processor = MLlavaProcessor.from_pretrained(ckpt)
			self.model = LlavaForConditionalGeneration.from_pretrained(ckpt, device_map="auto",torch_dtype=torch.bfloat16,
																	attn_implementation="flash_attention_2")
		elif 'Fuyu' in ckpt:
			from mantis.models.mfuyu import MFuyuForCausalLM, MFuyuProcessor
			self.processor = MFuyuProcessor.from_pretrained(ckpt)
			self.model = MFuyuForCausalLM.from_pretrained(ckpt, device_map="cuda",
													torch_dtype=torch.bfloat16,
													attn_implementation="flash_attention_2")
		else:
			raise ValueError(f'Not support {ckpt}')
		self.ckpt = ckpt

	def qa(self, image, prompt, mode):
		from mantis.models.mllava import chat_mllava
		if mode == 'image-only':
			image_pil = [Image.open(image[0]).convert('RGB')]
			if 'Idefics2' in self.ckpt:
				prompt = "User:" + prompt + "\nAssistant:"
				inputs = self.processor(text=prompt, images=image_pil, return_tensors='pt').to(self.model.device)
			elif 'llava-7b' in self.ckpt or 'Fuyu' in self.ckpt or 'llama3' in self.ckpt:
				prompt = prompt
				images = image_pil
		elif mode == 'image&video':
			video_frames = opencv_extract_frames_o(image[0], frames=self.num_frames)
			if 'Idefics2' in self.ckpt:
				prompt = "User:" + prompt.replace("<video>", "<image>" * len(video_frames)) + "\nAssistant:"
				inputs = self.processor(text=prompt, images=video_frames, return_tensors='pt').to(self.model.device)
			elif 'llava-7b' in self.ckpt or 'Fuyu' in self.ckpt or 'llama3' in self.ckpt:
				prompt = prompt.replace("<video>", "<image>" * len(video_frames))
				images = video_frames
		else:  # general
			image_list = []
			video_token_num = 0
			for it in image:
				if it.endswith(".mp4"):
					ori = len(image_list)
					image_list+=opencv_extract_frames_o(it, frames=self.num_frames)
					video_token_num = len(image_list) - ori
				else:
					image_list.append(Image.open(it).convert('RGB'))
			if 'Idefics2' in self.ckpt:
				prompt = "User:" + prompt.replace("<video>", "<image>" * video_token_num) + "\nAssistant:"
				inputs = self.processor(text=prompt, images=image_list, return_tensors='pt').to(self.model.device)
			elif 'llava-7b' in self.ckpt or 'Fuyu' in self.ckpt or 'llama3' in self.ckpt:
				prompt = prompt.replace("<video>", "<image>" * video_token_num)
				images = image_list

		print(prompt)
		if 'Idefics2' in self.ckpt:
			output = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
			response = self.processor.decode(output[0][2:], skip_special_tokens=True).split("Assistant:")[-1].strip()
		elif 'llava-7b' in self.ckpt or 'Fuyu' in self.ckpt:
			response, history = chat_mllava(prompt, images, self.model, self.processor,
											repetition_penalty=1.0, length_penalty=1.0,
											max_new_tokens=1, num_beams=1, do_sample=False,
											pad_token_id=self.processor.tokenizer.eos_token_id)
		elif 'llama3' in self.ckpt:
			response, history = chat_mllava(prompt, images, self.model, self.processor,
											max_new_tokens=1, num_beams=1, do_sample=False)
		cprint(response, 'cyan')
		return response

class Phi(QAModelInstance):
	def __init__(self, ckpt="microsoft/Phi-3-vision-128k-instruct", torch_device=torch.device("cuda"), model_precision=torch.float32):
		from transformers import AutoProcessor, AutoModelForCausalLM
		self.model = AutoModelForCausalLM.from_pretrained(ckpt,
				device_map="cuda", trust_remote_code=True,
				torch_dtype=model_precision,
				_attn_implementation='flash_attention_2'
		).to(torch_device).eval()  # use _attn_implementation='eager' to disable flash attention
		self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
		self.num_video_frames = test_frame
	def _process_prompt(self, prompt, frame_num):
		while "<video>" in prompt:
			n = next(frame_num)
			prompt = prompt.replace("<video>", "<image>" * n, 1)
		messages = []
		image_counter = 0
		while "<image>" in prompt:
			prompt = prompt.replace("<image>", f"<|image_{image_counter + 1}|>", 1)
			image_counter += 1

		messages.append({"role": "user", "content": prompt})

		return messages

	def qa(self, image, prompt, mode, video_desc_flag=None):
		if image is not None:
			images_pil = []
			frame_num = []
			for v in image:
				if v.endswith(".mp4"):
					frames = opencv_extract_frames_o(v, self.num_video_frames)
					frame_num.append(len(frames))
					images_pil += frames
				else:
					images_pil.append(load_image_o(v))

			messages = self._process_prompt(prompt, iter(frame_num))
		else:
			images_pil = None
			messages = self._process_prompt(prompt, None)
		print(messages)
		input = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		inputs = self.processor(input, images_pil, return_tensors="pt").to("cuda:0")

		generation_args = {
			"max_new_tokens": 500,
			"temperature": 0.0,
			"do_sample": False,
			"repetition_penalty": 1.0,
			"length_penalty": 1.0
		}
		generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)
		# remove input tokens
		generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
		response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
		cprint(response, 'cyan')
		return response

class Cambrian(QAModelInstance):
	def __init__(self, ckpt="nyu-visionx/cambrian-8b", torch_device=torch.device("cuda"), model_precision=torch.float32):
		# init as https://github.com/cambrian-mllm/cambrian/blob/main/inference.py
		from eval.models.qa_model.model_library.cambrian.model.builder import load_pretrained_model
		from eval.models.qa_model.model_library.cambrian.mm_utils import get_model_name_from_path
		seed = 42
		torch.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

		model_path = os.path.expanduser(ckpt)
		model_name = get_model_name_from_path(model_path)
		self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name)
		self.temperature = 0
		self._choose_conv_mode(ckpt)
		# only support torch.float16

	def _choose_conv_mode(self, ckpt):
		if "cambrian-phi3-3b" in ckpt:
			self.conv_mode = "phi3"
		elif "cambrian-8b" in ckpt:
			self.conv_mode = "llama_3"
		elif "cambrian-34b" in ckpt:
			self.conv_mode = "chatml_direct"
		elif "cambrian-13b" in ckpt:
			self.conv_mode = "vicuna_v1"
		else:
			raise NotImplementedError
	def _process(self, image, question, tokenizer, image_processor, model_config):
		from eval.models.qa_model.model_library.cambrian.constants import IMAGE_TOKEN_INDEX
		from eval.models.qa_model.model_library.cambrian.conversation import conv_templates
		from eval.models.qa_model.model_library.cambrian.mm_utils import tokenizer_image_token, process_images
		conv = conv_templates[self.conv_mode].copy()
		conv.append_message(conv.roles[0], question+'Or you can describe what you can see')
		conv.append_message(conv.roles[1], None)
		prompt = conv.get_prompt()
		print(prompt)
		image_size = [image.size]
		image_tensor = process_images([image], image_processor, model_config)
		input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

		return input_ids, image_tensor, image_size, prompt

	def qa(self, image, prompt):
		image = Image.open(image).convert('RGB')
		input_ids, image_tensor, image_sizes, prompt = self._process(image, prompt, self.tokenizer, self.image_processor,
															self.model.config)
		input_ids = input_ids.to(device='cuda', non_blocking=True)
		with torch.inference_mode():
			output_ids = self.model.generate(
				input_ids,
				images=image_tensor,
				image_sizes=image_sizes,
				do_sample=True if self.temperature > 0 else False,
				temperature=self.temperature,
				num_beams=1,
				max_new_tokens=512,
				use_cache=True)
		outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
		cprint(outputs, 'cyan')
		return outputs

class QwenVL(QAModelInstance):
	def __init__(self, ckpt="Qwen/Qwen-VL", torch_device=torch.device("cuda"), model_precision=torch.float32):
		from transformers import AutoModelForCausalLM, AutoTokenizer
		self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
		if model_precision == torch.float32:
			self.model = AutoModelForCausalLM.from_pretrained(
				ckpt,
				device_map=torch_device,
				trust_remote_code=True,
				fp32=True,
				low_cpu_mem_usage=True,
			).eval()
		else:
			self.model = AutoModelForCausalLM.from_pretrained(
				ckpt,
				device_map=torch_device,
				trust_remote_code=True,
				bf16=True,
				low_cpu_mem_usage=True,
			).eval()

	def qa(self, image, prompt):
		if isinstance(image, Image.Image):
			# Check if the image is a PIL.Image object and save to a temporary file if so
			with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
				image.save(tmp.name)
				image_path = tmp.name

				# Use the temporary image path for the tokenizer
				query = self.tokenizer.from_list_format([
					{'image': image_path},
					{'text': prompt},
				])

				inputs = self.tokenizer(query, return_tensors='pt').to(self.model.device)
				out = self.model.generate(**inputs)

		else:
			# If `image` is not a PIL.Image object, use it directly
			query = self.tokenizer.from_list_format([
				{'image': image},
				{'text': prompt},
			])

			inputs = self.tokenizer(query, return_tensors='pt').to(self.model.device)
			out = self.model.generate(**inputs)

		answer = self.tokenizer.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True).strip()

		return answer


class QwenVLChat(QAModelInstance):
	def __init__(self, ckpt="Qwen/Qwen-VL-Chat", torch_device=torch.device("cuda"), model_precision=torch.float32):
		from transformers import AutoModelForCausalLM, AutoTokenizer
		from transformers.generation import GenerationConfig

		self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
		if model_precision == torch.float32:
			self.model = AutoModelForCausalLM.from_pretrained(
				ckpt,
				device_map=torch_device,
				trust_remote_code=True,
				fp32=True,
				low_cpu_mem_usage=True,
			).eval()
		else:
			self.model = AutoModelForCausalLM.from_pretrained(
				ckpt,
				device_map=torch_device,
				trust_remote_code=True,
				bf16=True,
				low_cpu_mem_usage=True,
			).eval()

		# Specify hyperparameters for generation
		self.model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

	def qa(self, image, prompt):
		if isinstance(image, Image.Image):
			# Check if the image is a PIL.Image object and save to a temporary file if so
			with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
				image.save(tmp.name)
				image_path = tmp.name

				# Use the temporary image path for the tokenizer
				query = self.tokenizer.from_list_format([
					{'image': image_path},
					{'text': prompt},
				])

				answer, history = self.model.chat(self.tokenizer, query=query, history=None)
		else:
			# If `image` is not a PIL.Image object, use it directly
			query = self.tokenizer.from_list_format([
				{'image': image},
				{'text': prompt},
			])

			answer, history = self.model.chat(self.tokenizer, query=query, history=None)
		cprint(answer, 'cyan')
		return answer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T


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


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
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



class InternVLChat(QAModelInstance):
	def __init__(self, ckpt="OpenGVLab/InternVL-Chat-V1-5", torch_device=torch.device("cuda"), model_precision=torch.float32):
		from transformers import AutoTokenizer, AutoModel
		# Required a 80GB A100. current not support multi gpus now, internvl's bug. 
		self.model = AutoModel.from_pretrained(
			ckpt,
			torch_dtype=torch.bfloat16,
			low_cpu_mem_usage=True,
			trust_remote_code=True,
			device_map='auto').eval()
		self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

	def _load_image(self, image_file, input_size=448, max_num=6):
		image = Image.open(image_file).convert('RGB')
		transform = build_transform(input_size=input_size)
		images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
		pixel_values = [transform(image) for image in images]
		pixel_values = torch.stack(pixel_values)
		return pixel_values

	def qa(self, image, prompt):
		if isinstance(image, Image.Image):
			# Check if the image is a PIL.Image object and save to a temporary file if so
			with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
				image.save(tmp.name)
				image_path = tmp.name
				pixel_values = self._load_image(image_path, max_num=6).to(torch.bfloat16).cuda()
		else:
			pixel_values = self._load_image(image, max_num=6).to(torch.bfloat16).cuda()

		generation_config = dict(
			num_beams=1,
			max_new_tokens=512,
			do_sample=False,
		)

		response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
		cprint(response, 'cyan')
		return response

class InternVLChat2(QAModelInstance):
	# zw@usc: 2025.06.01 update
	# This is the code of internvl2, which is based on the implementation in huggingface (https://huggingface.co/OpenGVLab/InternVL2-1B).
	# This class encapsulation is relatively bare, 
	# so it is recommended to refer to the implementation of lmdeploy https://github.com/InternLM/lmdeploy
	def __init__(self, ckpt="OpenGVLab/InternVL2-1B", torch_device=torch.device("cuda"), model_precision=torch.float32):
		from transformers import AutoTokenizer, AutoModel
		if '26B' in ckpt or '40B' in ckpt or '76B' in ckpt:  # multi-GPU mode
			if '2_5' not in ckpt:
				if '26' in ckpt:
					name = 'InternVL2-26B'
				elif '40B' in ckpt:
					name = 'InternVL2-40B'
				else:
					name = 'InternVL2-Llama3-76B'
			else:
				if '26' in ckpt:
					name = 'InternVL2_5-26B'
				elif '38B' in ckpt:
					name = 'InternVL2_5-38B'
				else:
					name = 'InternVL2_5-78B'
			device_map = self._split_model(name) # ex: 'InternVL2-1B'
			self.model = AutoModel.from_pretrained(
				ckpt,
				torch_dtype=torch.bfloat16,
				load_in_8bit=True,
				low_cpu_mem_usage=True,
				use_flash_attn=True,
				trust_remote_code=True,
				device_map=device_map
			).eval()
		else:
			self.model = AutoModel.from_pretrained(
				ckpt,
				torch_dtype=torch.bfloat16,
				low_cpu_mem_usage=True,
				use_flash_attn=True,
				trust_remote_code=True).eval().cuda()

		self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, use_fast=False)
		self.num_frames = test_frame

	def _build_transform(self, input_size):
		MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
		transform = T.Compose([
			T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
			T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
			T.ToTensor(),
			T.Normalize(mean=MEAN, std=STD)
		])
		return transform

	def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
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

	def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
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

	def _load_image(self, image_file, input_size=448, max_num=12):
		image = Image.open(image_file).convert('RGB')
		transform = build_transform(input_size=input_size)
		images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
		pixel_values = [transform(image) for image in images]
		pixel_values = torch.stack(pixel_values)
		return pixel_values
	def _split_model(self, model_name):
		device_map = {}
		world_size = torch.cuda.device_count()
		num_layers = {
			'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
			'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80,
			'InternVL2_5-1B': 24, 'InternVL_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
			'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80
		}[model_name]
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
		print(device_map)
		return device_map

	def qa(self, image, prompt, mode):
		image_list = []
		video_token_num = 0
		for it in image:
			if it.endswith(".mp4"):
				ori = len(image_list)
				image_list += opencv_extract_frames_o(it, frames=self.num_frames)
				video_token_num = len(image_list) - ori
			else:
				image_list.append(Image.open(it).convert('RGB'))

		pixel_values_list = []
		num_patches_list = []
		for img in image_list:
			with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
				img.save(tmp.name)
				image_path = tmp.name
				pv = self._load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
				pixel_values_list.append(pv)
				num_patches_list.append(pv.size(0))
		pixel_values = torch.cat(pixel_values_list, dim=0)

		prompt = prompt.replace("<video>", "<image>" * video_token_num)
		response, history = self.model.chat(self.tokenizer, pixel_values, prompt,
									   dict(max_new_tokens=10, do_sample=False),
									   num_patches_list=num_patches_list,
									   history=None, return_history=True)
		print('response')
		cprint(response, 'cyan')
		return response

class GPT4V(QAModelInstance):
	model_stamp = 'gpt-4-turbo'

	def __init__(self, ckpt, *args, **kwargs):
		from openai import OpenAI
		if isinstance(ckpt, str):
			self.client = OpenAI(api_key=ckpt)
		elif isinstance(ckpt, list):
			self.client = [OpenAI(api_key=c) for c in ckpt]
		self.completion_tokens = 0
		self.prompt_tokens = 0
		self.test_frame = test_frame
		self.resolution = 512

	def _video_to_base64_frames(self, video_path, num_frames=6):
		# ref to https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding
		video = cv2.VideoCapture(video_path)
		base64Frames = []
		total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
		frame_interval = max(total_frames // num_frames, 1)

		for i in range(total_frames):
			success, frame = video.read()
			if not success:
				break
			if i % frame_interval == 0:
				_, buffer = cv2.imencode(".jpg", frame)
				base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
				if len(base64Frames) >= num_frames:
					break
		video.release()
		return base64Frames

	def _replace_placeholders(self, prompt: str, images: list, video_len: int):
		img_idx = 0
		result = []

		# Split the prompt by <video> and <image> to handle each part separately
		parts = prompt.split('<video>')
		for i, part in enumerate(parts):
			if i > 0:  # if this is not the first part, it means we had a <video> placeholder
				if img_idx + video_len <= len(images):
					# Replace <video> with video_len images
					video_urls = [
						{"type": "image_url",
						"image_url": {"url": f"data:image/jpeg;base64,{images[img_idx + j]}",
									"size": self.resolution,
									"detail": "low"}}
						for j in range(video_len)
					]
					result.extend(video_urls)
					img_idx += video_len

			image_parts = part.split('<image>')
			for j, text in enumerate(image_parts):
				if j > 0:  # if this is not the first sub-part, it means we had an <image> placeholder
					if img_idx < len(images):
						image_url = {"type": "image_url",
									"image_url": {"url": f"data:image/jpeg;base64,{images[img_idx]}",
												"size": self.resolution,
												"detail": "low"}}
						result.append(image_url)
						img_idx += 1

				if text:  # Add the text part
					result.append({"type": "text", "text": text})

		return result

	def _get_response(self, client, image:list, prompt, video_len):
		# ref to https://platform.openai.com/docs/guides/vision
		while True:
			try:
				processed_prompt = self._replace_placeholders(prompt, image, video_len)
				if self.model_stamp == 'o1-2024-12-17':
					response = client.chat.completions.create(
						model=self.model_stamp,
						messages=[
							{
								"role": "user",
								"content": processed_prompt
							}
						],
						# max_completion_tokens=300,  # 300 for text; 2000 for others Not support
						# temperature=0., Not support
						seed=42,
					)
				else:
					response = client.chat.completions.create(
						model=self.model_stamp,
						messages=[
							{
								"role"   : "user",
								"content": processed_prompt
							}
						],
						max_tokens=2000,  # 300 for text; 2000 for others
						temperature=0.,
						seed=42,
					)
			except openai.BadRequestError as e:
				if e.code == "sanitizer_server_error":
					continue
				elif e.code == "content_policy_violation":
					response = ""
				else:
					raise e
			except openai.InternalServerError as e:
				continue
			break
		return response

	def cost(self):
		# https://openai.com/api/pricing/
		if self.model_stamp == 'gpt-4-turbo':
			return (0.03 * self.completion_tokens + 0.01 * self.prompt_tokens) / 1000  # dollar
		elif self.model_stamp == 'gpt-4o':
			return (0.005 * self.completion_tokens + 0.0015 * self.prompt_tokens) / 1000  # dollar
		elif self.model_stamp == 'gpt-4o-mini':
			return (0.00015 * self.completion_tokens + 0.000075 * self.prompt_tokens) / 1000  # dollar
		elif self.model_stamp == 'o1-2024-12-17':
			return (0.015 * self.completion_tokens + 0.0075 * self.prompt_tokens) / 1000  # dollar
		else:
			raise ValueError(f'not supporft {self.model_stamp}')

	def qa(self, image, prompt, mode, video_desc_flag=True):
		print(self.cost())
		v_frames = None
		if image is not None:
			# to avoid response 'As a text-based AI, I'm unable to view or analyze video content.'
			video_desc = 'The video is split to a series of images sampled at equal intervals from the beginning to the end of it, based on the series of images, answer the question.'
			base64_imgs = []
			for img in image:
				if img.endswith(".mp4"):
					# Because it may not be enough for self.test_frame, you may need another one to record
					v_frames = self._video_to_base64_frames(img, num_frames=self.test_frame)
					base64_imgs+=v_frames
					if video_desc_flag and (video_desc not in prompt):
						prompt = video_desc + prompt
				else:
					with open(img, "rb") as image_file:
						base64_imgs.append(base64.b64encode(image_file.read()).decode('utf-8'))
		else:
			base64_imgs = None
		print(prompt)
		if isinstance(self.client, list):
			pointer = 0
			while True:
				client = self.client[pointer]
				try:
					response = self._get_response(client, base64_imgs, prompt, len(v_frames) if v_frames is not None else None)
				except openai.RateLimitError as e:
					if pointer < len(self.client) - 1:
						pointer += 1
						continue
					else:
						raise e
				break
		else:
			response = self._get_response(self.client, base64_imgs, prompt, len(v_frames) if v_frames is not None else None)
		if isinstance(response, str):
			cprint(response, 'cyan')
			return response
		else:
			self.completion_tokens += response.usage.completion_tokens
			self.prompt_tokens += response.usage.prompt_tokens
			cprint(response.choices[0].message.content, 'cyan')
			return response.choices[0].message.content


class GPT4O(GPT4V):
	model_stamp = 'gpt-4o'
class GPT4Omini(GPT4V):
	model_stamp = 'gpt-4o-mini'
class O1(GPT4V):
	model_stamp = 'o1-2024-12-17'


def upload_image_to_oss(image_path, bucket_name='benverse', endpoint='http://oss-cn-hongkong.aliyuncs.com',
						access_key_id='<your access key>', access_key_secret='<you access key secret>'):
	import oss2
	import secrets

	endpoint = endpoint
	auth = oss2.Auth(access_key_id, access_key_secret)
	bucket = oss2.Bucket(auth, endpoint, bucket_name)

	file_name = f"{secrets.token_hex(9)}.png"
	with open(image_path, 'rb') as file:
		bucket.put_object(file_name, file)

	domain = endpoint[endpoint.find("http://") + 7:]
	return f'https://{bucket_name}.{domain}/{file_name}', file_name


def delete_image_from_oss(file_name, bucket_name='benverse', endpoint='http://oss-cn-hongkong.aliyuncs.com',
						access_key_id='<your access key>', access_key_secret='<you access key secret>'):
	import oss2
	endpoint = endpoint
	auth = oss2.Auth(access_key_id, access_key_secret)
	bucket = oss2.Bucket(auth, endpoint, bucket_name)
	bucket.delete_object(file_name)


class QwenVLAPI(QAModelInstance):
	model_name = None

	def __init__(self, ckpt, *args, **kwargs):
		self.ckpt = ckpt[0]
		self.access_key_id = ckpt[1]
		self.access_key_secret = ckpt[2]

	def _get_response(self, image_path, prompt):
		import dashscope
		dashscope.api_key = self.ckpt
		image_url, image_file_name = upload_image_to_oss(image_path, access_key_id=self.access_key_id, access_key_secret=self.access_key_secret)
		messages = [{
			'role'   : 'system',
			'content': [{
				'text': 'You are a helpful assistant.'
			}]
		}, {
			'role'   :
				'user',
			'content': [
				{
					'image': image_url
				},
				{
					'text': prompt
				},
			]
		}]
		while True:
			try:
				response = dashscope.MultiModalConversation.call(model=self.model_name, messages=messages)
				if response.code == 'DataInspectionFailed':
					response = ""
				elif response.code == 'Throttling.RateQuota':
					time.sleep(60)
					continue
				else:
					response = response["output"]["choices"][0]["message"]["content"][0]["text"]
			except:
				continue
			break
		delete_image_from_oss(image_file_name, access_key_id=self.access_key_id, access_key_secret=self.access_key_secret)
		return response

	def qa(self, image, prompt):
		if isinstance(image, str):
			response = self._get_response(image, prompt)
		else:
			with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
				image.save(tmp.name)
				image_path = tmp.name
				response = self._get_response(image_path, prompt)
		return response


class QwenVLPlus(QwenVLAPI):
	model_name = 'qwen-vl-plus'


class QwenVLMax(QwenVLAPI):
	model_name = 'qwen-vl-max'

import google.generativeai as genai
import re
import google
class GeminiVisionAPI(QAModelInstance):
	model_name = None

	def __init__(self, ckpt, *args, **kwargs):

		GOOGLE_API_KEY = ckpt
		genai.configure(api_key=GOOGLE_API_KEY)
		self.model = genai.GenerativeModel(self.model_name)

	def _process_prompt(self, prompt, image_paths):
		# Split the prompt based on <video> and <image> placeholders
		prompt_parts = re.split(r'(<video>|<image>)', prompt)

		# Initialize a list to hold the processed elements (text and loaded files)
		processed_prompt = []

		# Index to track which image or video path we're processing
		image_path_index = 0

		for part in prompt_parts:
			if part == "<video>":
				# Process video file
				video_file = genai.upload_file(path=image_paths[image_path_index])
				while video_file.state.name == "PROCESSING":
					print('.', end='')
					time.sleep(10)
					video_file = genai.get_file(video_file.name)
				if video_file.state.name == "FAILED":
					raise ValueError(video_file.state.name)
				processed_prompt.append(genai.get_file(name=video_file.name))
				image_path_index += 1
			elif part == "<image>":
				# Process image file
				img = Image.open(image_paths[image_path_index])
				if img.mode != 'RGB':
					img = img.convert('RGB')
				processed_prompt.append(img)
				image_path_index += 1
			else:
				# Just add the text part to the processed prompt
				processed_prompt.append(part)

		return processed_prompt

	def _get_response(self, image_path, prompt):
		prompt_p = self._process_prompt(prompt=prompt, image_paths=image_path)
		while True:
			try:
				print(prompt_p)
				response = self.model.generate_content(prompt_p, stream=True)
				response.resolve()
				response = response.text
			except ValueError:
				response = ""
			except genai.types.generation_types.BlockedPromptException:
				response = ""
			except google.api_core.exceptions.DeadlineExceeded:
				time.sleep(60)
				continue
			except google.api_core.exceptions.InternalServerError:
				continue
			break
		return response

	def cost(self):
		if self.model_name == 'xx':
			return (0.03 * self.completion_tokens + 0.01 * self.prompt_tokens) / 1000  # dollar

	def qa(self, image, prompt, mode):
		try:
			response = self._get_response(image, prompt)
			cprint(response, 'cyan')
		except Exception as e:
			print("An error occurred:", str(e))
			time.sleep(30)
			return None
		return response


class GeminiFlash(GeminiVisionAPI):
	model_name = 'gemini-1.5-flash'

class GeminiPro(GeminiVisionAPI):
	model_name = 'gemini-1.5-pro'

from anthropic import Anthropic
class Claude(QAModelInstance):
	# https://docs.anthropic.com/en/docs/build-with-claude/vision#example-multiple-images-with-a-system-prompt
	model_name = None
	def __init__(self, ckpt, *args, **kwargs):
		# https://docs.anthropic.com/en/api/client-sdks
		self.model = Anthropic(api_key=ckpt)
	def _get_response(self, image, prompt):
		img = Image.open(image)
		if img.mode != 'RGB':
			img = img.convert('RGB')
		img = img.resize((128, 128))
		buffered = BytesIO()
		img.save(buffered, format="JPEG")
		img_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

		message_content = [
			{
				"role": "user",
				"content": [
					{
						"type": "image",
						"source": {
							"type": "base64",
							"media_type": "image/jpeg",
							"data": img_data,
						},
					},
					{
						"type": "text",
						# "text": "what is shown of the image?" # to ensure can be seen
						"text": prompt.replace('<image>\n', '').replace('<video>\n', '').replace('<image>', '').replace(
							'<video>', '')
					}
				],
			}
		]
		print("Message content:", message_content)
		message = self.model.messages.create(
			model=self.model_name,
			max_tokens=1000,
			temperature=0,
			messages=message_content,
		)
		response = message.content[0].text
		return response
	def qa(self, image, prompt):
		try:
			response = self._get_response(image=image, prompt=prompt)
			cprint(response, 'cyan')
		except Exception as e:
			print("An error occurred:", str(e))
			time.sleep(30)
			return None
		return response

class Claude_sonnetx(Claude):
	model_name = 'claude-3-5-sonnet-20240620'
class Claude_sonnet(Claude):
	model_name = 'claude-3-sonnet-20240229'
class Claude_opus(Claude):
	model_name = 'claude-3-opus-20240229'
class Claude_haiku(Claude):
	model_name = 'claude-3-haiku-20240307'


class ReplicateAPI(QAModelInstance):
	model_name = None
	model_list = None

	def __init__(self, ckpt, *args, **kwargs):
		import replicate
		self.replicate_client = replicate.Client(api_token=ckpt)

	def _get_response(self, image_path, prompt):
		image = open(image_path, "rb")
		input = {
			"image" : image,
			"prompt": prompt
		}
		while True:
			try:
				output = self.replicate_client.run(
					self.model_list[self.model_name],
					input=input
				)
				response = "".join(output)
			except:
				time.sleep(60)
				continue
			break
		return response

	def qa(self, image, prompt):
		if isinstance(image, str):
			response = self._get_response(image, prompt)
		else:
			with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
				image.save(tmp.name)
				image_path = tmp.name
				response = self._get_response(image_path, prompt)

		return response


class LLaVA34B(ReplicateAPI):
	model_name = 'llava-v1.6-34b'
	model_list = {
		"llava-v1.6-34b": "yorickvp/llava-v1.6-34b:41ecfbfb261e6c1adf3ad896c9066ca98346996d7c4045c5bc944a79d430f174",
	}

def load_image_o(image_file):
	image = Image.open(image_file).convert("RGB")
	return image

class Owl3(QAModelInstance):
	def __init__(self, ckpt="mPLUG/mPLUG-Owl3-1B-241014", torch_device=torch.device("cuda"), model_precision=torch.float32):
		from transformers import AutoModel, AutoTokenizer
		self.model = AutoModel.from_pretrained(ckpt, device_map=torch_device,
										  attn_implementation='flash_attention_2', torch_dtype=model_precision,
										  trust_remote_code=True)
		self.model.eval().cuda()
		self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
		self.processor = self.model.init_processor(self.tokenizer)
		self.torch_device = torch_device
		self.num_video_frames = test_frame
	def _process_prompt(self, prompt, frame_num):
		while "<video>" in prompt:
			n = next(frame_num)
			prompt = prompt.replace("<video>", "<image>" * n, 1)
		messages = []
		image_counter = 0
		while "<image>" in prompt:
			prompt = prompt.replace("<image>", f"<|image|>", 1)
			image_counter += 1

		messages.append({"role": "user", "content": prompt})
		messages.append({"role": "assistant", "content": ""})
		return messages

	def qa(self, image, prompt, mode, video_desc_flag=None):
		if image is not None:
			images_pil = []
			frame_num = []
			for v in image:
				if isinstance(v, str) and v.endswith(".mp4"):
					frames = opencv_extract_frames_o(v, self.num_video_frames)
					frame_num.append(len(frames))
					images_pil += frames
				else:
					images_pil.append(load_image_o(v))

			messages = self._process_prompt(prompt, iter(frame_num))
		else:
			images_pil = None
			messages = self._process_prompt(prompt, None)
		print(messages)
		inputs = self.processor(messages, images=images_pil, videos=None)

		inputs.to(self.torch_device)
		inputs.update({
			'tokenizer': self.tokenizer,
			'max_new_tokens': 100,
			'decode_text': True,
		})

		response = self.model.generate(**inputs)[0]
		cprint(response, 'cyan')
		return response

class MiniCPMV(QAModelInstance):
	def __init__(self, ckpt="openbmb/MiniCPM-V-2", torch_device=torch.device("cuda"), model_precision=torch.float32):
		model_precision = torch.float32  # torch.bfloat16 and torch.float16 may be wrong
		from transformers import AutoModel, AutoTokenizer
		self.model = AutoModel.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=model_precision)
		# For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
		self.model = self.model.to(device=torch_device, dtype=model_precision).eval()
		self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
		self.ckpt = ckpt
	def qa(self, image, prompt):
		try:
			if isinstance(image, str):
				image = Image.open(image).convert('RGB')
			prompt = prompt.replace('<image>\n', '')
			print(prompt)
			msgs = [{'role': 'user', 'content': prompt}]
			if '2_6' in self.ckpt or '2_5' in self.ckpt:  # version 2.6
				answer = self.model.chat(
					image=image,
					msgs=msgs,
					context=None,
					tokenizer=self.tokenizer,
					sampling=False,
					temperature=0.1,
					max_new_token=10
				).replace("Answer:", '').replace(" ", '', 1)
			else:
				# debug: https://huggingface.co/openbmb/MiniCPM-V-2/discussions/23
				# 🔥 please follow the link to solve the bug !!!
				# change modeling_minicpm.py in your download checkpoint
				answer, context, _ = self.model.chat(
					image=image,
					msgs=msgs,
					context=None,
					tokenizer=self.tokenizer,
					sampling=False,
					temperature=0.1,
					max_new_token=10
				)
			cprint(answer, 'cyan')
			return answer
		except:
			return None


class DeepSeekVL(QAModelInstance):
	def __init__(self, ckpt="deepseek-ai/deepseek-vl-1.3b-chat", torch_device=torch.device("cuda"), model_precision=torch.float32):
		from transformers import AutoModelForCausalLM
		from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
		self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(ckpt)
		self.tokenizer = self.vl_chat_processor.tokenizer
		vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
		self.vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

	def qa(self, image, prompt):
		from deepseek_vl.utils.io import load_pil_images
		if isinstance(image, Image.Image):
			# Check if the image is a PIL.Image object and save to a temporary file if so
			with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
				image.save(tmp.name)
				image_path = tmp.name
		else:
			image_path = image
		conversation = [
			{
				"role": "User",
				"content": prompt.replace("<image>", "<image_placeholder>"),
				"images": [image_path]
			},
			{
				"role": "Assistant",
				"content": ""
			}
		]

		# load images and prepare for inputs
		pil_images = load_pil_images(conversation)
		prepare_inputs = self.vl_chat_processor(
			conversations=conversation,
			images=pil_images,
			force_batchify=True
		).to(self.vl_gpt.device)

		# run image encoder to get the image embeddings
		inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

		# run the model to get the response
		outputs = self.vl_gpt.language_model.generate(
			inputs_embeds=inputs_embeds,
			attention_mask=prepare_inputs.attention_mask,
			pad_token_id=self.tokenizer.eos_token_id,
			bos_token_id=self.tokenizer.bos_token_id,
			eos_token_id=self.tokenizer.eos_token_id,
			max_new_tokens=10,
			do_sample=False,
			use_cache=True
		)
		answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
		cprint(answer, 'cyan')
		return answer

class XinyuanVL(QAModelInstance):
	def __init__(self, ckpt="Cylingo/Xinyuan-VL-2B", torch_device=torch.device("cuda"), model_precision=torch.float32):
		from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
		# default: Load the model on the available device(s)
		self.model = Qwen2VLForConditionalGeneration.from_pretrained(
			ckpt, torch_dtype=model_precision, device_map=torch_device
		).eval()
		# default processer
		self.processor = AutoProcessor.from_pretrained(ckpt)
		self.torch_device = torch_device
	@torch.inference_mode()
	def qa(self, image, prompt):
		from qwen_vl_utils import process_vision_info

		if isinstance(image, Image.Image):
			raw_image = image.convert('RGB').resize((1024, 1024))
		else:
			raw_image = Image.open(image).convert('RGB').resize((1024, 1024)) # avoid out of Mem

		with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
			raw_image.save(tmp)
			image_path = tmp.name
			messages = [
				{
					"role": "user",
					"content": [
						{
							"type": "image",
							"image": image_path,
						},
						{"type": "text", "text": prompt.replace('<image>\n','')},
					],
				}
			]
			mess = self.processor.apply_chat_template(
				messages, tokenize=False, add_generation_prompt=True
			)
			image_inputs, video_inputs = process_vision_info(messages)
			inputs = self.processor(
				text=[mess],
				images=image_inputs,
				videos=video_inputs,
				padding=True,
				return_tensors="pt",
			)
			inputs = inputs.to(self.torch_device)

			# Inference: Generation of the output
			generated_ids = self.model.generate(**inputs, max_new_tokens=128)
			generated_ids_trimmed = [
				out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
			]
			output_text = self.processor.batch_decode(
				generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
			)[0]
			cprint(output_text, 'cyan')
			return output_text

class AquilaVL(QAModelInstance):
	def __init__(self, ckpt="BAAI/Aquila-VL-2B-llava-qwen", torch_device=torch.device("cuda"), model_precision=torch.float32):
		from llava.model.builder import load_pretrained_model
		self.tokenizer, self.model, self.image_processor, self.max_length = \
			load_pretrained_model(ckpt, None, "llava_qwen", device_map=torch_device)
		self.model.eval()
		self.model_precision = model_precision
		self.torch_device = torch_device
	@torch.inference_mode()
	def qa(self, image, prompt):
		import copy
		from llava.mm_utils import process_images, tokenizer_image_token
		from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
		from llava.conversation import conv_templates
		if isinstance(image, str):
			image = Image.open(image).convert('RGB')

		image_tensor = process_images([image], self.image_processor, self.model.config)
		image_tensor = [_image.to(dtype=self.model_precision, device=self.torch_device) for _image in image_tensor]

		# print(DEFAULT_IMAGE_TOKEN)
		conv = copy.deepcopy(conv_templates["qwen_1_5"])
		conv.append_message(conv.roles[0], prompt)
		conv.append_message(conv.roles[1], None)
		prompt_question = conv.get_prompt()

		input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
			0).to(self.torch_device)
		image_sizes = [image.size]

		cont = self.model.generate(
			input_ids,
			images=image_tensor,
			image_sizes=image_sizes,
			do_sample=False,
			temperature=0,
			max_new_tokens=4096,
		)

		text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

		cprint(text_outputs, 'cyan')
		return text_outputs
