'''
wei chow@usc implement it ref to the official repo: https://huggingface.co/allenai/Molmo-72B-0924
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH='./' python molmo.py
'''
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import torch
from eval.eval_utils.task_evaluator import PhysionBenchEvaluator, test_frame

class Molmo():
    def __init__(self, ckpt):
        self.processor = AutoProcessor.from_pretrained(
            ckpt,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        ).eval()

    def qa(self, image, prompt, mode=None):
        # process the image and text
        inputs = self.processor.process(
            images=[Image.open(image)],
            text=prompt.replace('<image>', '')
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        with torch.no_grad():
            inputs["images"] = inputs["images"].to(torch.bfloat16)
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=2, stop_strings="<|endoftext|>", do_sample=False),
                tokenizer=self.processor.tokenizer
            )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_text = generated_text.replace(' ', '', 1)
        # print the generated text
        print(generated_text)
        return generated_text


if __name__ == "__main__":
    model_name = 'allenai/Molmo-72B-0924'
    dataset_path = "<your path>"  # todo@physbench step1: download dataset

    model= Molmo(ckpt=model_name)   # todo@physbench step2: implement your model class with method <def qa(self, image, prompt, mode)>

    task_evaluator = PhysionBenchEvaluator(
        model=model,
        mode='image-only',   # todo@physbench step3: choose your mode in ["image-only", "image&video", "general"]
        dataset_path=dataset_path,
        model_name=model_name,
        resume=True,
        sample_ratio=None,
        split='test'
    )

    # todo@physbench step4: add the model_name in test() function, just like OpenGVLab/InternVL2_5-78B
    task_evaluator.test()
