import tempfile
from typing import Callable, Union
from termcolor import cprint
import numpy as np
import torch
from PIL import Image
import os
import torchvision
from decord import VideoReader, cpu
from .base_qa_model import QAModel, QAModelInstance
from .imageqa_model import ImageQAModel
from eval.eval_utils.task_evaluator import test_frame

# you can also use vqamodel on video task by concatenating the frames of the video
videoqa_models = {
    "video-llava-7b"   : ("VideoLLaVA", 	"LanguageBind/Video-LLaVA-7B"), 	# conda activate physbench_video
    "chat-univi-7b"    : ("ChatUniVi", 		"Chat-UniVi/Chat-UniVi"),		# conda activate physbench_video
    "chat-univi-13b"   : ("ChatUniVi", 		"Chat-UniVi/Chat-UniVi-13B"),		# conda activate physbench_video
    "pllava-7b"        : ("PLLaVA", 		"ermu2001/pllava-7b"),				# conda activate physbench
    "pllava-13b"       : ("PLLaVA", 		"ermu2001/pllava-13b"),			# conda activate physbench
}

def list_videoqa_models():
	return list(videoqa_models.keys())

class VideoQAModel(QAModel):
	def __init__(
			self,
			model_name,
			prompt_name: str = None,
			prompt_func: Callable = None,
			model: QAModelInstance = None,
			torch_device: Union[int, str] = -1,
			precision=torch.bfloat16,
			choice_format='letter',
			enable_choice_search: bool = False,
	):
		super().__init__(model_name, prompt_name, prompt_func, choice_format, enable_choice_search)

		if isinstance(torch_device, str):
			torch_device = torch.device(torch_device)
		else:
			if torch_device == -1:
				torch_device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
			else:
				torch_device = torch.device(f"cuda:{torch_device}")

		if model is None:
			print(f"Loading {model_name}...")
			class_name, ckpt = videoqa_models[model_name]
			self.model_precision = precision
			self.model = eval(class_name)(ckpt, torch_device, self.model_precision)
			print(f"Finish loading {model_name}")
		else:
			print(f"Using provided self.model...")
			self.model = model

	@torch.no_grad()
	def _qa(self, data, prompt):
		if isinstance(data, str):
			return self.model.qa(data, prompt)
		else:
			with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
				with open(tmp.name, 'wb') as file:
					file.write(data)
				video_path = tmp.name
				answer = self.model.qa(video_path, prompt)
			return answer


class ImageQAModel4Video(VideoQAModel):
	def __init__(
			self,
			model: ImageQAModel,
			prompt_name: str,
			prompt_func: Callable,
			num_rows: int = 2,
			num_columns: int = 2,
			choice_format='letter',
			enable_choice_search: bool = False,
	):
		super(VideoQAModel, self).__init__(model.model_name, prompt_name, prompt_func, choice_format, enable_choice_search)
		self.num_rows = num_rows
		self.num_columns = num_columns
		self.num_frames = self.num_rows * self.num_columns
		self.model = model

	@torch.no_grad()
	def _qa(self, data, prompt):
		pass


class VideoLLaVA(QAModelInstance):
	def __init__(self, ckpt='LanguageBind/Video-LLaVA-7B', torch_device=torch.device("cuda"), model_precision=torch.float32):
		# Environment setup# Disable certain initializations if necessary

		from .model_library.Video_LLaVA.videollava.utils import disable_torch_init
		from .model_library.Video_LLaVA.videollava import constants
		from .model_library.Video_LLaVA.videollava.conversation import conv_templates, SeparatorStyle
		from .model_library.Video_LLaVA.videollava.model.builder import load_pretrained_model
		from .model_library.Video_LLaVA.videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

		self.constants = constants
		self.SeparatorStyle = SeparatorStyle
		self.tokenizer_image_token = tokenizer_image_token
		self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
		self.conv_templates = conv_templates

		disable_torch_init()
		cache_dir = "cache_dir"

		self.device = torch_device
		model_name = 'video-llava-7b'
		self.tokenizer, self.model, processor, _ = load_pretrained_model(ckpt, None, model_name, device=torch_device, cache_dir=cache_dir)
		self.video_processor = processor['video']
		self.image_processor = processor['image']

	def qa(self, video_path, question):
		conv_mode = "llava_v1"
		conv = self.conv_templates[conv_mode].copy()
		if video_path.endswith('.mp4'):
			video_tensor = self.video_processor(video_path, return_tensors='pt')['pixel_values'] # torch
			if isinstance(video_tensor, list):
				tensor = [video.to(self.device, dtype=torch.float16) for video in video_tensor]
			else:
				tensor = video_tensor.to(self.device, dtype=torch.float16)

			conv.append_message(conv.roles[0], question.replace('<video>', ' '.join([self.constants.DEFAULT_IMAGE_TOKEN] * self.model.get_video_tower().config.num_frames)))


		else:
			image_tensor = self.image_processor(video_path, return_tensors='pt')['pixel_values']  # torch
			if isinstance(image_tensor, list):
				tensor = [video.to(self.device, dtype=torch.float16) for video in image_tensor]
			else:
				tensor = image_tensor.to(self.device, dtype=torch.float16)

			conv.append_message(conv.roles[0], question)


		conv.append_message(conv.roles[1], None)
		prompt = conv.get_prompt()
		print(prompt)
		input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.constants.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
		stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
		keywords = [stop_str]
		stopping_criteria = self.KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

		with torch.inference_mode():
			output_ids = self.model.generate(
				input_ids,
				images=tensor,
				do_sample=True,
				temperature=0.1,
				max_new_tokens=1024,
				use_cache=True,
				stopping_criteria=[stopping_criteria])

		outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
		if outputs.endswith(stop_str):
			outputs = outputs[:-len(stop_str)].strip()
		cprint(outputs, 'cyan')
		return outputs


class ChatUniVi(QAModelInstance):
	def __init__(self, ckpt='chat-univi-7b', torch_device=torch.device("cuda"), model_precision=torch.float32):
		from .model_library.Chat_UniVi.ChatUniVi import constants
		from .model_library.Chat_UniVi.ChatUniVi.conversation import conv_templates, SeparatorStyle
		from .model_library.Chat_UniVi.ChatUniVi.model.builder import load_pretrained_model
		from .model_library.Chat_UniVi.ChatUniVi.utils import disable_torch_init
		from .model_library.Chat_UniVi.ChatUniVi.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
		from decord import VideoReader, cpu

		self.constants = constants
		self.cpu = cpu
		self.VideoReader = VideoReader
		self.conv_templates = conv_templates
		self.tokenizer_image_token = tokenizer_image_token
		self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
		self.SeparatorStyle = SeparatorStyle

		disable_torch_init()
		self.tokenizer, self.model, image_processor, context_len = load_pretrained_model(ckpt, None, "ChatUniVi")

		mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
		mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)
		if mm_use_im_patch_token:
			self.tokenizer.add_tokens([self.constants.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
		if mm_use_im_start_end:
			self.tokenizer.add_tokens([self.constants.DEFAULT_IM_START_TOKEN, self.constants.DEFAULT_IM_END_TOKEN], special_tokens=True)
		self.model.resize_token_embeddings(len(self.tokenizer))

		vision_tower = self.model.get_vision_tower()
		if not vision_tower.is_loaded:
			vision_tower.load_model()
		self.image_processor = vision_tower.image_processor

		if self.model.config.config["use_cluster"]:
			for n, m in self.model.named_modules():
				m = m.to(dtype=torch.float16)

	def qa(self, video_path, question):
		# setting parameters
		max_frames = test_frame

		if video_path is not None:
			if video_path.endswith(".mp4"):
				video_frames, slice_len = self._get_rawvideo_dec(video_path,
								self.image_processor, max_frames=max_frames, video_framerate=2)
				qs = question.replace("<video>", self.constants.DEFAULT_IMAGE_TOKEN * slice_len)
			else:
				video_frames, slice_len = self._process_single_image(video_path, self.image_processor)
				qs = question

			conv = self.conv_templates["simpleqa"].copy()
			conv.system = """A chat between a human and an artificial intelligence assistant. The assistant can only answer use one letter for the option (A, B, C or D)."""


			conv.append_message(conv.roles[0], qs+"Best option: ")
			conv.append_message(conv.roles[1], None)
			prompt = conv.get_prompt()
			cprint(prompt)

			input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.constants.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

			stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
			keywords = [stop_str]
			stopping_criteria = self.KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

			with torch.inference_mode():
				output_ids = self.model.generate(
					input_ids,
					images=video_frames.half().cuda(),
					do_sample=False,
					temperature=0,
					top_p=None,
					num_beams=1,
					output_scores=True,
					return_dict_in_generate=True,
					max_new_tokens=10,
					use_cache=True,
					length_penalty=1,
					stopping_criteria=[stopping_criteria])

			output_ids = output_ids.sequences
			input_token_len = input_ids.shape[1]
			n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
			if n_diff_input_output > 0:
				print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
			outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
			outputs = outputs.strip()
			if outputs.endswith(stop_str):
				outputs = outputs[:-len(stop_str)]
			outputs = outputs.strip()
			outputs = outputs.replace('The best option is ', '')
			cprint(outputs, 'cyan')
			return outputs

	def _process_single_image(self, image_path, image_processor):
		if not os.path.exists(image_path):
			raise FileNotFoundError(f"{image_path} does not exist.")

		image = Image.open(image_path).convert("RGB")
		processed_image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
		processed_image = processed_image.unsqueeze(0)  # Add batch dimension
		return processed_image, 1

	def _get_rawvideo_dec(self, video_path, image_processor, max_frames=None, image_resolution=224, video_framerate=1, s=None, e=None):
		# speed up video decode via decord.
		if max_frames is None:
			max_frames = test_frame
		if s is None:
			start_time, end_time = None, None
		else:
			start_time = int(s)
			end_time = int(e)
			start_time = start_time if start_time >= 0. else 0.
			end_time = end_time if end_time >= 0. else 0.
			if start_time > end_time:
				start_time, end_time = end_time, start_time
			elif start_time == end_time:
				end_time = start_time + 1

		if os.path.exists(video_path):
			vreader = self.VideoReader(video_path, ctx=self.cpu(0))
		else:
			print(video_path)
			raise FileNotFoundError

		fps = vreader.get_avg_fps()
		f_start = 0 if start_time is None else int(start_time * fps)
		f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
		num_frames = f_end - f_start + 1
		if num_frames > 0:
			# T x 3 x H x W
			sample_fps = int(video_framerate)
			t_stride = int(round(float(fps) / sample_fps))

			all_pos = list(range(f_start, f_end + 1, t_stride))
			if len(all_pos) > max_frames:
				sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
			else:
				sample_pos = all_pos

			batch = vreader.get_batch(sample_pos)
			if hasattr(batch, 'asnumpy'):
				batch_np = batch.asnumpy()
			elif hasattr(batch, 'numpy'):
				batch_np = batch.numpy()
			else:
				raise TypeError("The object does not have asnumpy or numpy methods.")
			patch_images = [Image.fromarray(f) for f in batch_np]
			patch_images = torch.stack([image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images])
			slice_len = patch_images.shape[0]

			return patch_images, slice_len
		else:
			print("video path: {} error.".format(video_path))

# ref to https://github.com/magic-research/PLLaVA/tree/main/tasks/eval/videoqabench
class PLLaVA():
	def __init__(self, ckpt='', torch_device=torch.device("cuda"), model_precision=torch.float32):
		self.model, self.processor = self._load_model(
			pretrained_model_name_or_path=ckpt,
			num_frames=test_frame,
			use_lora=True,
			lora_alpha=4,
			weight_dir=ckpt
		)
		self.model = self.model.to(torch_device).eval()
		self.conv_mode = 'eval_videoqabench'
		self.num_frames = test_frame
		self.RESOLUTION = 768 # as default

	def _load_model(self, pretrained_model_name_or_path, num_frames, use_lora, lora_alpha, weight_dir):
		from eval.models.qa_model.model_library.PLLaVA.model_utils import load_pllava
		model, processor = load_pllava(pretrained_model_name_or_path, num_frames=num_frames, use_lora=use_lora,
									lora_alpha=lora_alpha, weight_dir=weight_dir)
		model = model.to(torch.device('cuda'))
		model = model.eval()
		return model, processor
	def _get_index(self, num_frames, num_segments):
		seg_size = float(num_frames - 1) / num_segments
		start = int(seg_size / 2)
		offsets = np.array([
			start + int(np.round(seg_size * idx)) for idx in range(num_segments)
		])
		return offsets

	def _load_video(self, video_path, num_segments=8, return_msg=False, num_frames=4, resolution=336):
		transforms = torchvision.transforms.Resize(size=resolution)
		vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
		num_frames = len(vr)
		frame_indices = self._get_index(num_frames, num_segments)
		images_group = list()
		for frame_index in frame_indices:
			img = Image.fromarray(vr[frame_index].asnumpy())
			images_group.append(transforms(img))
		if return_msg:
			fps = float(vr.get_avg_fps())
			sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
			msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
			return images_group, msg
		else:
			return images_group

	def _load_image(self, image_path, num_segments=4, resolution=336):
		transforms = torchvision.transforms.Resize(size=resolution)
		img = Image.open(image_path)
		img = transforms(img)
		return [img] * num_segments  # Repeat the image to simulate multiple frames

	def qa(self, video_path, question): # image: str
		from eval.models.qa_model.model_library.PLLaVA.model_utils import pllava_answer
		from eval.models.qa_model.model_library.PLLaVA.eval_utils import conv_templates
		media_type = 'video'
		if video_path.lower().endswith('.mp4'):
			img_list = self._load_video(video_path, num_segments=self.num_frames, resolution=self.RESOLUTION)
		else:
			img_list = self._load_image(video_path, num_segments=self.num_frames, resolution=self.RESOLUTION)
			media_type = 'image'

		conv = conv_templates[self.conv_mode].copy()
		conv.user_query(question.replace('<image>\n', '').replace('<image>', '').replace('<video>\n', '').replace('<video>', ''),
						conv.pre_query_prompt, is_mm=True)

		img_list = [img.convert("RGB") if img.mode != "RGB" else img for img in img_list] # debug
		print(conv)
		llm_response, conv = pllava_answer(conv=conv, model=self.model, processor=self.processor,img_list=img_list,
										do_sample=False, max_new_tokens=256, media_type=media_type)
		cprint(llm_response, 'cyan')
		return llm_response