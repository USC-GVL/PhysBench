import json
import os
from tqdm import tqdm
import random
import warnings
import tempfile
from PIL import Image, ImageDraw, ImageFont
from termcolor import cprint
import numpy as np
warnings.filterwarnings('ignore')

test_frame = 8  # 1, 2, 4, 8(default), 16, 32

task_split = {
    # ------------------ image-only
    "instructblip-flan-t5-xl"               : "image-only",
    "instructblip-flan-t5-xxl"              : "image-only",
    "instructblip-vicuna-7b"                : "image-only",
    "instructblip-vicuna-13b"               : "image-only",
    "blip2-flant5xxl"                       : "image-only",
    "llava-1.5-7b-hf"                      : "image-only",
    "llava-1.5-13b-hf"                     : "image-only",
    "llava-v1.6-mistral-7b-hf"             : "image-only",
    "llava-v1.6-vicuna-7b-hf"              : "image-only",
    "MiniCPM-V2"                           : "image-only",
    "MiniCPM-V2.5"                         : "image-only",
    "MiniCPM-V2.6"                         : "image-only",
    "Qwen-VL-Chat"                         : "image-only",
    "InternVL-Chat-V1-5-quantable"         : "image-only",
    "cambrian-8b"                          : "image-only",
    "Xinyuan-VL-2B"                        : "image-only",
    "Aquila-VL-2B"                         : "image-only",
    "deepseek1B"                           : "image-only",
    "deepseek7B"                           : "image-only",
    "paligemma2-3b"                        : "image-only",
    "paligemma2-10b"                       : "image-only",
    "MolmoE-1B"                            : "image-only",
    "MolmoE-7B-O"                          : "image-only",
    "MolmoE-7B-D"                          : "image-only",
    "claude-3-5-sonnet"                    : "image-only",
    "claude-3-sonnet"                      : "image-only",
    "claude-3-opus"                        : "image-only",
    "claude-3-haiku"                       : "image-only",
    
    # ------------------ image&video
    # conda activate video
    "video-llava-7b"                       : "image&video",
    "chat-univi-7b"                        : "image&video",
    "chat-univi-13b"                       : "image&video",
    
    # conda activate image
    "pllava-7b"                            : "image&video",
    "pllava-13b"                           : "image&video",
    
    # ------------------ general
    "llava-interleave-qwen-7b-hf"          : "general",
    "llava-interleave-qwen-7b-dpo-hf"      : "general",
    "vila-1.5-3b-s2"                      : "general",
    "vila-1.5-3b"                         : "general",
    "vila-1.5-8b"                         : "general",
    "vila-1.5-13b"                        : "general",
    "Phi-3-vision-128k-instruct"          : "general",
    "Phi-3.5V"                            : "general",
    "mPLUG-Owl3-1B-241014"                : "general",
    "mPLUG-Owl3-2B-241014"                : "general",
    "mPLUG-Owl3-7B-241101"                : "general",
    "LLaVA-NeXT-Video-7B-DPO-hf"           : "general",
    "LLaVA-NeXT-Video-7B-hf"              : "general",
    "Mantis-8B-Idefics2"                   : "general",
    "Mantis-llava-7b"                     : "general",
    "Mantis-8B-siglip-llama3"             : "general",
    "Mantis-8B-clip-llama3"               : "general",
    "InternVL2-1B"                        : "general",
    "InternVL2-2B"                        : "general",
    "InternVL2-4B"                        : "general",
    "InternVL2-8B"                        : "general",
    "InternVL2-26B"                       : "general",
    "InternVL2-40B"                       : "general",
    "InternVL2-76B"                       : "general",
    'InternVL2_5-1B'                      : "general",
    'InternVL2_5-2B'                      : "general",
    'InternVL2_5-4B'                      : "general",
    'InternVL2_5-8B'                      : "general",
    'InternVL2_5-26B'                     : "general",
    'InternVL2_5-38B'                     : "general",
    'InternVL2_5-78B'                     : "general",
    "gpt4v"                               : "general",
    "gpt4o"                               : "general",
    "o1"                             	  : "general",
    "gpt4o-mini"                          : "general",
    "gemini-1.5-flash"                    : "general",
    "gemini-1.5-pro"                      : "general"
}


class PhysionBenchEvaluator():
	def __init__(
			self,
			model,
			mode:str,
			dataset_path:str,
			model_name:str,
			sample_ratio: float = None,
			resume: bool = True,
			split: str= 'test'
	):
		'''
		:param model: Model, need have a method named qa
		:param mode:
		:model_name: 
		:resume: If enabled, each test will be based on the previous test. This is only supported when sample_ratio is not
		'''
		super().__init__()
		self.model = model
		self.mode = mode
		self.model_name = model_name
		self.seed = 2024 # fix
		self.resume = resume

		assert mode in ["image-only", "image&video", "general"], f"not supporting {mode}"

		# ref to llava 1.5
		self.end_prompt = "\nAnswer with the option's letter from the given choices directly. You can only answer one letter from A, B, C, or D."
		# ref to TaskMeAnything
		self.video2image_prompt = 'This is a series of images sampled at equal intervals from the beginning to the end of a video, based on the series of images, output the best option for the question.\n'

		self.sample_ratio = sample_ratio
		self.split = split
		self._load_dataset(dataset_path)

	def _load_dataset(self, dataset_path, result_path='results'):
		self.dataset_path = dataset_path
		os.makedirs(os.path.join(self.dataset_path, result_path), exist_ok=True)
		if self.sample_ratio is None:
			self.result_file = os.path.join(self.dataset_path, result_path, self.model_name + '.json')
		else:
			self.result_file = os.path.join(self.dataset_path, result_path, self.model_name + f'_{self.sample_ratio}' + '.json')

		if test_frame != 8:   # we use 8 frame for video as default
			self.result_file = self.result_file.replace(self.model_name, f"{self.model_name}_{test_frame}")
			self.mode = "video-only"

		with open(self.dataset_path +r'/test.json', 'r', encoding='utf-8') as file:
			dataset = json.load(file)

		if self.split == 'val':
			val_dataset = []
			for item in dataset:
				if item['split'] == 'val':
					val_dataset.append(item)
			dataset = val_dataset
			self.result_file = self.result_file.replace('.json', '_val.json')

		if self.mode == "image-only":
			# self.dataset = [item for item in dataset if item['mode'] == "image-only"]
			self.dataset = [item for item in dataset if item['mode'] != "general"]
		elif self.mode == "image&video":
			self.dataset = [item for item in dataset if item['mode'] != "general"]
		elif self.mode == "video-only":
			self.dataset = [item for item in dataset if item['mode'] == "image&video"]
		else:
			self.dataset = dataset # all support

		self.model_answers = []  # for save the answer

		if self.sample_ratio is not None:
			assert self.resume == False
			random.seed(self.seed)
			dataset_size = len(self.dataset)
			sample_size = int(self.sample_ratio * dataset_size)
			self.dataset = random.sample(self.dataset, sample_size)
		else:
			if self.resume and os.path.exists(self.result_file):
				with open(self.result_file, 'r', encoding='utf-8') as f:
					model_answers = json.load(f)
				existing_items = {(item['idx']) for item in model_answers if item["answer"] is not None}
				self.model_answers = []
				not_match = 0
				for answer in model_answers:
					if answer["answer"] is None or answer["answer"] =='':
						continue
					matching_item = next((item for item in self.dataset if item['idx'] == answer['idx']), None)
					if matching_item is not None:
						self.model_answers.append(answer)
					else:
						not_match += 1
				if not_match != 0:
					print('[Info] Unmatched: ', not_match)
				self.dataset = [item for item in self.dataset if (item['idx']) not in existing_items]
				print(f'[Info] Still have {len(self.dataset)} to test')
	def _process_visual_path(self, file_name):
		if file_name.endswith(".mp4"):
			file_path = self.dataset_path + '/video/' + file_name
		elif file_name.endswith(".jpg") or file_name.endswith(".JPG") or file_name.endswith(".png"):
			file_path = self.dataset_path + '/image/' + file_name
		else:
			raise NotImplementedError
		return file_path

	def _concat_video(self, video_path:str):
		# ref to TaskMeAnything
		num_rows = 2
		num_columns = 2
		combined_image = video_to_concat_image(video_path, num_rows, num_columns)

		return combined_image

	def test(self):
		for item in tqdm(self.dataset[:5]):
			prompt = item["question"] + self.end_prompt
			visuals = [self._process_visual_path(f) for f in item["file_name"]]
			if self.model_name in ['llava-1.5-7b-hf', 'llava-1.5-13b-hf', 'cambrian-8b',
								'llava-v1.6-mistral-7b-hf', 'llama3-llava-next-8b-hf', 'llava-v1.6-vicuna-7b-hf',
								"claude-3-5-sonnet", "claude-3-sonnet", "claude-3-opus", "claude-3-haiku",
								'MiniCPM-V2', 'MiniCPM-V2.5', 'MiniCPM-V2.6',
								'Xinyuan-VL-2B', 'Aquila-VL-2B', 'deepseek1B', 'deepseek7B', 'paligemma2-3b',
								'paligemma2-10b', 'MolmoE-1B', 'MolmoE-7B-O', 'MolmoE-7B-D',
								'allenai/Molmo-72B-0924']:
				if visuals[0].endswith('.mp4'):
					combined_image = self._concat_video(visuals[0])
					with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp:
						combined_image.save(tmp, format='JPEG')
						tmp.flush()
						video_path = tmp.name
						answer = self.model.qa(image=video_path, prompt=self.video2image_prompt+prompt.replace('<video>', '<image>'))  # image only
				else:
					answer = self.model.qa(image=visuals[0], prompt=prompt)  # image only
			elif self.model_name in ['blip2-flant5xxl', 'instructblip-vicuna-7b', 'instructblip-vicuna-13b',
				'instructblip-flan-t5-xl', 'instructblip-flan-t5-xxl', 'Qwen-VL-Chat',
				'InternVL-Chat-V1-5-quantable']:  # image only
				if visuals[0].endswith('.mp4'):
					combined_image = self._concat_video(visuals[0])
					with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp:
						combined_image.save(tmp, format='JPEG')
						tmp.flush()
						video_path = tmp.name
						answer = self.model.qa(image=video_path, prompt=self.video2image_prompt+prompt.replace('<video>\n', '').replace("A.", "Select from the following choices.\nA."))  # image only
				else:
					answer = self.model.qa(image=visuals[0], prompt=prompt.replace('<image>\n', '').replace("A.", "Select from the following choices.\nA."))  # image only
			elif self.model_name in ['video-llama2-7b', 'video-llama2-13b', 'chat-univi-7b', 'chat-univi-13b',
									'video-chatgpt-7b', 'video-chat2-7b', 'pllava-7b', 'pllava-13b']:
				if visuals[0].endswith(".mp4"):
					prompt = "This is a series of images sampled at equal intervals from the beginning to the end of a video, based on the series of images, answer the question. Based on the video, output the best option for the question. You must only output the option." + prompt
				else:
					prompt = "Based on the image, output the best option for the question. You must only output the option." + prompt
					if self.model_name != 'video-chatgpt-7b':
						prompt = prompt + 'best choice option is: '
				answer = self.model.qa(video_path=visuals[0], question=prompt) # video only
			elif self.model_name in ['video-llava-7b']:
				answer = self.model.qa(video_path=visuals[0], question=prompt)  # video only
			elif self.model_name in ["llava-interleave-qwen-7b-hf", "llava-interleave-qwen-7b-dpo-hf", 'vila-1.5-3b',
									'vila-1.5-8b', 'vila-1.5-13b', 'LLaVA-NeXT-Video-7B-hf', 'LLaVA-NeXT-Video-7B-DPO-hf',
									'gpt4v', "gpt4o-mini", "gpt4o", "o1", 'Phi-3-vision-128k-instruct', 'Phi-3.5V',
									'gemini-1.5-flash', 'gemini-1.5-pro', 'Mantis-8B-Idefics2', 'Mantis-8B-Fuyu',
									'Mantis-8B-clip-llama3', 'Mantis-8B-siglip-llama3', 'mPLUG-Owl3-1B-241014',
									'mPLUG-Owl3-2B-241014', 'mPLUG-Owl3-7B-241101', 'InternVL2-1B', 'InternVL2-2B',
									'InternVL2-4B', 'InternVL2-8B', 'Mantis-llava-7b', 'vila-1.5-3b-s2',
									'InternVL2_5-1B', 'InternVL2_5-2B', 'InternVL2_5-4B', 'InternVL2_5-8B',
									 'Efficient-Large-Model/NVILA-8B', 'Efficient-Large-Model/NVILA-13B',
									 'Efficient-Large-Model/NVILA-Lite-8B', 'Efficient-Large-Model/NVILA-Lite-13B'
									]:  # general
				answer = self.model.qa(image=visuals, prompt=prompt, mode=item["mode"])
			elif self.model_name in ['OpenGVLab/InternVL2-26B', 'OpenGVLab/InternVL2-40B',
									 'OpenGVLab/InternVL2-Llama3-76B',
									 'OpenGVLab/InternVL2_5-26B', 'OpenGVLab/InternVL2_5-38B',
									 'OpenGVLab/InternVL2_5-78B']:
				gt_ind = None
				combined_image = None
				for index in range(len(visuals)):
					if visuals[index].endswith('.mp4') or visuals[index].endswith(
							'.MP4'):  # as there is only one image in PhysBench
						combined_image = self._concat_video(visuals[0])
						gt_ind = index
				if combined_image is not None:
					with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
						combined_image.resize((224, 224))
						print('resize')
						combined_image.save(tmp.name)
						visuals[gt_ind] = tmp.name
						answer = self.model.qa(image=visuals, prompt=prompt, mode=item["mode"])
				else:
					answer = self.model.qa(image=visuals, prompt=prompt, mode=item["mode"])
			else:
				raise NotImplementedError
				answer = self.model.qa(image=visuals, prompt=prompt)

			self.model_answers.append({
				"idx": item["idx"],
				"answer": answer,
			})
			
		os.makedirs(os.path.dirname(self.result_file), exist_ok=True)
		with open(self.result_file, 'w', encoding='utf-8') as f:
			json.dump(self.model_answers, f, ensure_ascii=False, indent=4)


def concatenate_image(images, rows, columns, separator_width=10):
	# Ensure we have the exact number of images needed
	if len(images) != rows * columns:
		raise ValueError(f"Expected {rows * columns} images, but got {len(images)}.")

	# Calculate the max width and height of images to standardize sizes
	max_width = max(img.width for img in images)
	max_height = max(img.height for img in images)

	# Resize images to the max width and height
	resized_images = [img.resize((max_width, max_height), Image.Resampling.LANCZOS) for img in images]

	# Calculate the total width and height for the combined image
	total_width = max_width * columns + separator_width * (columns - 1)
	total_height = max_height * rows + separator_width * (rows - 1)
	combined_image = Image.new('RGB', (total_width, total_height), color='white')

	# Place images in the specified grid
	x_offset = 0
	y_offset = 0
	for i, img in enumerate(resized_images):
		combined_image.paste(img, (x_offset, y_offset))
		if (i + 1) % columns == 0:  # Move to the next row after the last column
			x_offset = 0
			y_offset += img.height + separator_width
		else:  # Move to the next column
			x_offset += img.width + separator_width

	# Add numbers to each image for identification
	draw = ImageDraw.Draw(combined_image)
	try:
		font_size = (max_width + max_height) // 2 // 12
		font = ImageFont.load_default(size=font_size)
	except IOError:
		font = ImageFont.truetype("arial", 20)

	x_offset = 0
	y_offset = 0
	for i, img in enumerate(resized_images):
		text = str(i + 1)
		text_x = x_offset + 10
		text_y = y_offset + 10
		text_width, text_height = font_size, font_size
		font_color = get_contrasting_color(combined_image, text_x, text_y, text_width, text_height)
		draw.text((text_x, text_y), text, fill=font_color, font=font)
		if (i + 1) % columns == 0:
			x_offset = 0
			y_offset += img.height + separator_width
		else:
			x_offset += img.width + separator_width

	return combined_image


def video_to_concat_image(video_path, num_rows, num_columns):
	return concatenate_image(sample_frames(video_path, num_rows * num_columns), num_rows, num_columns)


def sample_frames(video_path, n):
	import cv2
	# Open the video file
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print("Error: Could not open video.")
		return []

	# Calculate total number of frames and video FPS
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# Calculate interval in terms of frames
	interval = max(1, total_frames // n)

	# Sample frames
	sampled_frames = []
	for i in range(0, total_frames, interval):
		# Set the current frame position
		cap.set(cv2.CAP_PROP_POS_FRAMES, i)

		# Read the frame
		ret, frame = cap.read()
		if not ret:
			print(f"Error: Could not read frame {i}.")
			break

		# Convert the frame to PIL Image
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		pil_img = Image.fromarray(frame_rgb)
		sampled_frames.append(pil_img)

		# Stop if we have collected n frames
		if len(sampled_frames) >= n:
			break

	# Release the video capture object
	cap.release()

	return sampled_frames


def get_contrasting_color(image, x, y, width, height):
	"""
	Determine a contrasting color (black or white) based on the average color of a specified area in the image.
	"""
	# Crop the relevant part of the image
	cropped_image = image.crop((x, y, x + width, y + height))
	# Convert to numpy array for analysis
	np_image = np.array(cropped_image)
	# Calculate the average color
	average_color = np.mean(np_image, axis=(0, 1))
	# Brightness calculation based on perceived luminance
	brightness = np.sqrt(0.299 * average_color[0] ** 2 + 0.587 * average_color[1] ** 2 + 0.114 * average_color[2] ** 2)
	# Return white for dark backgrounds and black for light backgrounds
	return 'white' if brightness < 128 else 'black'
