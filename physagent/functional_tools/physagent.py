import json
import re
import os
from tqdm import tqdm
from overrides import overrides
from termcolor import cprint
from .wrap_model import get_wrap_model
from .func_library import FuncAgent
from .prompts import choose_tool_system, extract_tool_system
from eval.physagent.eval_utils.task_evaluator import PhysionBenchEvaluator
from eval.eval_utils.caculate_core import sub_task_order
import random

class PhysAgent:
    '''
    ref to PerceptionAgent in https://github.com/USC-GVL/Agent-Driver
    '''
    def __init__(self, name, verbose=False):
        self.model = get_wrap_model(name)
        self.funcagent = FuncAgent()
        self.verbose = verbose  # 打印一些细节信息

    def _self_verify_call(self, item, visuals, type):
        f_name = f"{type}_func"
        function_to_call = getattr(self.funcagent, f_name)
        if type == 'dynamics':
            try_num = 0
            verified_response = False
            while try_num < 3 and verified_response == False:
                if try_num > 0:
                    r_seed = random.randint(0, 2 ** 8 - 1)
                    random.seed(r_seed)  # Random seed within range
                else:
                    r_seed = None
                function_response = function_to_call(
                    item, visuals, self.model, r_seed
                )
                verified_response = self.funcagent.self_verify_func(item=item,
                                             visuals=visuals, answer=function_response, vlm=self.model, seed=r_seed)
                try_num += 1
        else:
            function_response = function_to_call(
                item, visuals, self.model
            )

        return function_response

    def _functional_call(self, item, visuals):
        # Call the function from VLM's response
        type = self.choose_tool(item)
        f_name = f"{type}_func"
        function_to_call = getattr(self.funcagent, f_name)
        if not callable(function_to_call):
            print(f"Function {f_name} is not callable!")
            return None
        else:
            function_response = self._self_verify_call(
                item=item, visuals=visuals, type=type
            )
        return function_response

    def _symbolic_classifer(self, question):
        if re.search(re.compile(
            r'(closest(?: point)?|nearest|is farthest|is closest|the farthest) '
            r'to the (camera|flame)[?:]\n?'
            ), question):
            return 'depth'
        elif any(phrase in question for phrase in (
                "The camera is closer to the objects",
                "The camera is farther away from the objects",
                "The camera moves upward or downward",
                "The camera rotates along the vertical axis (upside or downside)",
                "The camera rotates along the horizontal axis (left or right)",
                "Which object is the camera closest to?",
                "the video shot",
            )):
            return 'camera'
        elif re.search(re.compile(
            r"(?i)("
            r"The color of the light changes from|"
            r"Move parallel to the line between the|"
            r"It's just that the light source is (?:stronger and the light source position remains the same|weaker and the light source position remains the same|closer to the objects|farther away from the objects)|"
            r"The light source moves downward|"
            r"the light source closest to\?|"
            r"the light come from\?|"
            r"How does the light change|"
            r"the light source|"
            r"Where is the light"
            r")"
        ), question):
            return 'light'
        elif any(att in question for att in ("sharp?", "brittle?", "stiff?", "elastic?",
                "malleable?", "soft?", 'smooth?')):
            return 'attribute'
        elif re.compile(
                r"which of the following options are you most likely to (see|encounter|notice|observe|find|come across)\?"
                r"|which of the following options is most likely to be seen\?"
            ).search(question):
            return "movement"
        elif 'biggest in volume?' in question or \
                'size of the ice cube changing in the video?' in question or \
                'size of the ice cube in the video?' in question or \
                "cube's size" in question or \
                'fluid?\n' in question or 'larger density' in question or 'the mass of' in question or \
                'is the largest?' in question or "cube's mass" in question or 'hat is the color of' in question or \
                'What color does' in question or 'What color is' in question or 'What color' in question or \
                'How many' in question:
            return 'property'
        else:
            return None

    def _remove_special_tag(self, text):
        return text.replace('<image>\n', '').replace('<video>\n', '').replace('<image>', '').replace('<video>', '')
    def _get_question(self, text):
        return text.split("A.")[0]
    def _get_pure_question(self, text):
        return self._remove_special_tag(self._get_question(text))
    def choose_tool(self, item):
        # neuro-symbolic auto classify
        # notice the type in PhysAgent is not equal to PhysBench, and the classification accuracy is based on different models itself
        question = self._get_pure_question(item["question"])
        # symbolic
        type = self._symbolic_classifer(question=question)
        # neuro
        if type is None:
            type = self.model.chat(image=None, prompt=choose_tool_system + question)
            try_num = 0
            while type not in self.legal_filed:
                # means not a conclusion choice
                type = self.model.chat(image=None, prompt=extract_tool_system + type)

                try_num = try_num + 1
                if try_num >= 5:
                    type = 'deep_think'
                    break

        if type == 'others':
            type = 'deep_think'

        if self.verbose:
            cprint(f'[Choose tool type] {type}', 'red')

        return type

    def get_perception_results(self, item, visuals):
        return self._functional_call(item=item, visuals=visuals)



class AgentEvaluator(PhysionBenchEvaluator):
    def __init__(
            self,
            name:str,
            dataset_path:str
    ):
        object.__init__(self)
        self.physagent = PhysAgent(name=name)
        self.mode = "general"
        self.model_name = name
        self.seed = 2024  # fix
        self.resume = True
        self.sample_ratio = None
        self._load_dataset(dataset_path, split=f'agent')


        with open(self.result_file, 'r', encoding='utf-8') as f:
            model_answers = json.load(f)
        # filter the done questions
        existing_items = {(str(item['file_name']), str(item['question'])) for item in model_answers if item["answer"] is not None}
        # model_answers = self.model_answers
        self.model_answers = []
        not_match = 0
        for answer in model_answers:
            if answer["answer"] is None or answer["answer"] =='':
                continue
            matching_item = next((item for item in self.dataset if
                                  item['file_name'][0] == answer['file_name'][0] and
                                  item['question'] == answer['question']), None)

            if matching_item is not None:
                answer['task_type'] = matching_item['task_type']
                answer['sub_type'] = matching_item['sub_type']
                answer['ability_type'] = matching_item['ability_type']
                answer['mode'] = matching_item['mode']
                self.model_answers.append(answer)
            else:
                not_match += 1
        if not_match != 0:
            print('Unmatched: ', not_match)
        self.dataset = [item for item in self.dataset if
                        (str(item['file_name']), str(item['question'])) not in existing_items]

        cprint(len(self.dataset), 'cyan')

    @overrides
    def test(self):
        for item in tqdm(self.dataset):
            # thinking
            visuals = [self._process_visual_path(f) for f in item["file_name"]]
            answer = self.physagent.get_perception_results(item=item, visuals=visuals)

            if answer is None:
                continue

            # extract
            extract_prompt = 'The following sentences contain answers (one of A, B, C, D) and corresponding analysis. Your role is to find the answer. Please only return one of the four letters A, B, C, D. The sentences are: \n'
            answer = self.physagent.model.chat(image=None, prompt=extract_prompt + answer)

            self.model_answers.append({
                "file_name": item["file_name"],
                "question": item["question"],
                "answer": answer,
                "gt": item["answer"],
                "task_type": item["task_type"],
                "sub_type": item["sub_type"],
                "ability_type": item["ability_type"],
                "mode": item["mode"]
            })

        with open(self.result_file, 'w', encoding='utf-8') as f:
            json.dump(self.model_answers, f, ensure_ascii=False, indent=4)