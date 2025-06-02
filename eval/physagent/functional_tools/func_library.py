from termcolor import cprint
from transformers import pipeline
from PIL import Image
import cv2
import re
import numpy as np
from .prompts import cot_prompt
from .knowledge_pool import attribute_table
from eval.physagent.case import detect_object, get_labeled_image, load_ground_sam

class FuncAgent:
    dpt = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)
    sam2_predictor, grounding_processor, grounding_model = load_ground_sam()
    dinox = False

    def _easy_chat(self, question, visuals, vlm, seed=None):
        return vlm.chat(image=visuals, prompt=question)

    def inside_knowlegde(self, attr_type, object_names, dim):
        if (attr_type is not None) and (dim is not None):
            attr_know = f"There are some object have {dim} {attr_type}: {attribute_table[attr_type][dim]}\n"
        else:
            attr_know = ''

        if object_names is not None:
            obj_know = f'The object might be A: {object_names[0]}, B: {object_names[1]}, C: {object_names[2]}, D: {object_names[3]}\n'
        else:
            # test by DINO-X
            obj_know = ''

        if (attr_type is not None) and (dim is not None) and (object_names is not None):
            answer = 0
            index = 0
            for idx, obj in enumerate(object_names):
                if obj in attribute_table[attr_type][dim]:
                    answer += 1
                    index = idx
            if answer == 1:
                return ['A', 'B', 'C', 'D'][index]

        return attr_know + obj_know, None

    def outside_knowledge(self, description):
        if description is not None:
            if type(description) == list and all(isinstance(item, str) for item in description):
                knowledge = ''.join(description)
            else:
                return ''
            return f'You can also draw on external knowledge. The principle of this video and the knowledge involved might be {knowledge}'
        else:
            return ''

    def attribute_func(self, item, visuals, vlm, seed=None):
        if item["object"] is not None and len(item["object"]) == 4:
            object_names = [obj.replace(f"{obj.split('_')[-1]}", '').replace('_', ' ') for obj in item["object"]]
            object_names = [(obj + '?????').replace(' ?????', '') for obj in object_names]
        else:
            object_names = None

        pattern = re.compile(r'\b(sharp|brittle|stiff|elastic|malleable|soft|smooth)\b\??', re.IGNORECASE)
        matches = pattern.findall(item['question'])
        if len(matches) == 1:
            attr_type = matches[0].replace('?', '')
        else:
            attr_type = None

        dim = item["question"].replace(f" {attr_type}?", '').split('\nA.')[0].split(' ')[-1]
        if dim not in ['most', 'least']:
            dim = None

        knowledge_prompt = self.inside_knowlegde(attr_type=attr_type, object_names=object_names, dim=dim)

        if self.dinox: # enable segmentation
            try:
                text = object_names.join('. ')
                img_path = visuals[0]
                results = detect_object(sam2_predictor=self.sam2_predictor, grounding_processor=self.grounding_processor,
                                        grounding_model=self.grounding_model, img_path=img_path, text=text)
                box_annotated_pil, mask_annotated_pil = get_labeled_image(results=results,
                                   sam2_predictor=self.sam2_predictor, img_path=img_path, save_image=True)
                visuals = [mask_annotated_pil]
            except:
                print('continue seg and detect')

        if knowledge_prompt in ['A', 'B', 'C', 'D']:
            return knowledge_prompt

        prompt = 'The following questions inquire some problem about attribute of the object. If there are consecutive images, they represent a video. Please carefully review each image, think thoroughly, and then make your selection. '\
                 + 'Here is some knowledge :' + str(knowledge_prompt) \
                 + '\nThe question is:' + str(item["question"])

        answer = vlm.chat(image=visuals, prompt=prompt)
        return answer

    def depth_func(self, item, visuals, vlm, save_depth_figure=False, seed=None):
        if type(item["description"]) is dict and all(key in item["description"].keys() for key in ['A', 'B', 'C', 'D']):
            ord = item["description"]
            image = Image.open(visuals[0])
            depth = self.dpt(image)["depth"]

            depth_array = np.array(depth)
            if depth_array.shape == (1024,1024):
                depth_opt = [depth_array[ord['A'][1]][ord['A'][0]], depth_array[ord['B'][1]][ord['B'][0]], depth_array[ord['C'][1]][ord['C'][0]], depth_array[ord['D'][1]][ord['D'][0]]]

                if 'closest' in item["question"] or 'nearest' in item["question"]:
                    value = max(depth_opt)
                    index = depth_opt.index(value)
                    answer = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[index]
                    cprint(answer, 'cyan')
                    return answer
                if 'farthest' in item["question"]:
                    value = min(depth_opt)
                    index = depth_opt.index(value)
                    answer = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[index]
                    cprint(answer, 'cyan')
                    return answer

                if save_depth_figure:                  # save as black image
                    depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                    depth_uint8 = depth_normalized.astype(np.uint8)
                    cv2.imwrite('depth_output.jpg', depth_uint8)
                    print("Depth image saved successfully.")

        return self._easy_chat(question=item["question"], visuals=visuals, vlm=vlm)

    def property_func(self, item, visuals, vlm, seed=None):
        prompt = 'The following questions inquire some problem about object property, like size, mass, color, number. If there are consecutive images, they represent a video. Please carefully review the video, think thoroughly, and then make your selection.' + self.outside_knowledge(
            item["description"]) + item["question"]
        if vlm.model_name == 'gpt':  # necessary for GPT
            prompt = prompt + '\nIn any case, please provide the option you believe is the most reliable.'
        answer = vlm.chat(image=visuals, prompt=prompt)
        return answer

    def movement_func(self, item, visuals, vlm, seed=None):
        task_desc = "For questions asking which image will appear next, you need to first analyze whether the red line is turning left, right, or going straight. Then, based on the images in the options, determine whether the scene shown corresponds to the left, right, or front of the image in the question, or if it is unrelated, and make your selection accordingly."
        prompt = 'T. If there are consecutive images, they represent a video. Please carefully review each image, think thoroughly, and then make your selection. ' + self.outside_knowledge(
            item["description"]) + task_desc + item["question"]
        answer = vlm.chat(image=visuals, prompt=prompt)
        return answer

    def deep_think_func(self, item, visuals, vlm, seed=None):
        prompt = 'The following questions inquire some problem about environment or physical dynamics. If there are consecutive images, they represent a video. Please carefully review the video, think thoroughly, and then make your selection.' + self.outside_knowledge(
            item["description"]) + item["question"]
        if vlm.model_name == 'gpt':  # necessary for GPT
            prompt = prompt + '\nIn any case, please provide the option you believe is the most reliable.'
        answer = vlm.chat(image=visuals, prompt=prompt)
        return answer

    def camera_func(self, item, visuals, vlm, seed=None):
        task_desc = """
        Carefully compare the differences in each frame, paying particular attention to the changes in the objects within the scene. The perspective hasn't changed, but the objects are either enlarging or shrinking, indicating that the camera is moving away or closer. If the objects are shifting away from the center of the frame in one direction, this suggests that the camera's angle is rotating in the opposite direction of the object's movement. If the question asks which object is closest in the image, you can convert it into a depth estimation task to further verify your conclusion.
        """
        if vlm.model_name == 'gpt':
            prompt = 'The following questions inquire some problem about the camera setting. If there are consecutive images, they represent a video. Please carefully review each image, think thoroughly, and then make your selection. ' + self.outside_knowledge(
                item["description"]) + task_desc + item["question"]
            prompt = prompt + '\nIn any case, please think twice and provide the option you believe is the most reliable. Do not give an uncertain response or choose the option that is uncertain.'
        elif vlm.model_name == 'phi':
            prompt = item["question"]
        else:
            raise ValueError('Unknown VLM')
        answer = vlm.chat(image=visuals, prompt=prompt)
        return answer

    def light_func(self, item, visuals, vlm, seed=None):
        task_desc = """
        Carefully compare each frame, paying particular attention to changes in shadows. Brighter areas often indicate the location closest to the light source. If there is no change in the scene between the first and last frames, this indicates a change in the color of the light source, not the camera angle. If the distribution of light on the object does not change significantly but the overall brightness decreases, it suggests that the light source is moving away. By observing the rotation of the shadows, you can deduce the opposite direction of the light source's movement. Additionally, note that if the shadows lengthen without changing direction, the light source is moving away from the object; if they shorten without changing direction, the light source is moving closer.
        """
        prompt = 'The following questions inquire some problem about light. If there are consecutive images, they represent a video. Please carefully review each image, think thoroughly, and then make your selection. ' + self.outside_knowledge(
            item["description"]) + task_desc + item["question"]
        if vlm.model_name == 'gpt':
            prompt = prompt + '\nIn any case, please think twice and provide the option you believe is the most reliable. Do not give an uncertain response or choose the option that is uncertain.'
        elif vlm.model_name == 'phi':
            prompt = prompt + '\nIn any case, you must confidently provide an answer. Do not give an uncertain response or choose the option that is uncertain.'
        answer = vlm.chat(image=visuals, prompt=prompt)
        return answer

    def dynamics_func(self, item, visuals, vlm, seed=None):
        prompt = 'The following questions inquire some physical dynamics phenomena. If there are consecutive images, they represent a video. Please carefully review the video, think thoroughly, and then make your selection.' + item["question"]
        prompt = prompt+self.outside_knowledge(item["description"])
        prompt = prompt + '\nIn any case, please provide the option you believe is the most reliable.'

        answer = vlm.chat(image=visuals, prompt=prompt)
        return answer

    def self_verify_func(self, item, visuals, answer, vlm, seed=None) -> bool:
        prompt = "You are a discriminator tasked with determining the correctness of the answer provided in relation to the given question. The question is: " + \
                    item["question"] + \
                    "\nThe answer is:" + \
                    answer + \
                    "\nIf the answer appropriately addresses the question, respond with 'T'. Otherwise, respond with 'F'. You should be strict."

        if seed is not None:
            answer = vlm.chat(image=visuals, prompt=prompt, seed=seed)
        else:
            answer = vlm.chat(image=visuals, prompt=prompt)

        if 'T' in answer or 't' in answer:
            return True
        else:
            return False