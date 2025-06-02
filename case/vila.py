'''
wei chow@usc implement it ref to the official repo: https://github.com/NVlabs/VILA
CUDA_VISIBLE_DEVICES=1,2,3,4 python vila.py
'''
import llava
from llava import conversation as clib
from llava.media import Image, Video
# please put the submodule `llava` here from https://github.com/NVlabs/VILA
from termcolor import cprint
from eval.eval_utils.task_evaluator import PhysionBenchEvaluator, test_frame

class NVILA():
    def __init__(self, ckpt):
        self.model = llava.load(ckpt)

    def qa(self, image, prompt, mode=None):
        # Set conversation mode
        lang = prompt.replace('<image>', '(image)').replace('<video>', '(video)')
        clib.default_conversation = clib.conv_templates['auto'].copy()

        # Prepare multi-modal prompt
        prompt = []
        if image is not None:
            for media in image or []:
                if any(media.endswith(ext) for ext in [".jpg", ".jpeg", ".png", '.JPG']):
                    media = Image(media)
                elif any(media.endswith(ext) for ext in [".mp4", ".mkv", ".webm", '.MP4']):
                    media = Video(media)
                else:
                    raise ValueError(f"Unsupported media type: {media}")
                prompt.append(media)

        prompt.append(lang)

        # Generate response
        response = self.model.generate_content(prompt)

        cprint(response,'cyan')
        return response


if __name__ == "__main__":
    dataset_path="<your path>"               # todo@physbench step1: download dataset
    model_name = "Efficient-Large-Model/NVILA-Lite-15B"

    model = NVILA(ckpt=model_name)               # todo@physbench step2: implement your model class with method <def qa(self, image, prompt, mode)>

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


