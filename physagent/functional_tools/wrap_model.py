from eval.models.qa_model.imageqa_model import GPT4V
import base64, openai
from termcolor import cprint

def get_wrap_model(name):
    if name == 'gpt':
        return GPT4o_wrap()
    else:
        raise ValueError(f"Unknown name: {name}")

class GPT4o_wrap(GPT4V):
    model_stamp = 'gpt-4o'
    def __init__(self):
        super().__init__(ckpt='<GPT API>')  # todo: API
        self.model_name = 'gpt'
    def chat(self, image, prompt, seed=42):
        print(self.cost())
        v_frames = None
        if image is not None:
            # to avoid response 'As a text-based AI, I'm unable to view or analyze video content.'
            video_desc = 'The video is split to a series of images sampled at equal intervals from the beginning to the end of it, based on the series of images, answer the question.'
            base64_imgs = []
            for img in image:
                if img.endswith(".mp4"):
                    v_frames = self._video_to_base64_frames(img, num_frames=self.test_frame)
                    base64_imgs += v_frames
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
                try:  # only have one video
                    response = self._get_response(client, base64_imgs, prompt,
                                                  len(v_frames) if v_frames is not None else None,
                                                  seed=seed)
                except openai.RateLimitError as e:
                    if pointer < len(self.client) - 1:
                        pointer += 1
                        continue
                    else:
                        raise e
                break
        else:
            response = self._get_response(self.client, base64_imgs, prompt,
                                          len(v_frames) if v_frames is not None else None,
                                          seed=seed)
        if isinstance(response, str):
            cprint(response, 'cyan')
            return response
        else:
            self.completion_tokens += response.usage.completion_tokens
            self.prompt_tokens += response.usage.prompt_tokens
            cprint(response.choices[0].message.content, 'cyan')
            return response.choices[0].message.content
