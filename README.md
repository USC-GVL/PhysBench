# Evaluation Guidelines

### Clone this scripts

```shell
git clone https://github.com/USC-GVL/PhysBench
```

### Download the dataset

To download the data from the [🤗 Dataset](https://huggingface.co/datasets/USC-GVL/PhysBench), you can prepare the dataset by executing the following command.

It is recommended to set `<your_path_for_dataset>` as `eval/physbench`; however, you may change this to a different path if necessary, in which case you must adjust the `--dataset_path` parameter accordingly.

```shell
cd <your_path_for_dataset>  # such as '/home/usr/dataset'
huggingface-cli download USC-GVL/PhysBench --local-dir . --local-dir-use-symlinks False --repo-type dataset
# Unzip the compressed files of videos and pictures
yes | unzip image.zip -d image
yes | unzip video.zip -d video
```

### Prepare Environment

To avoid conflicts caused by the specific library dependencies of the **video-llava** and **chat-univi** models, which may interfere with other models, we created a separate Conda environment. If you do not intend to test these two models, this step can be disregarded. All other operations are performed within the **physbench** environment.

```shell
# main environment (recommended)
conda create --name physbench python=3.10
conda activate physbench
pip install -r requirements.txt
# Create a special environment for video-llava and chat-univi (selective)
conda create --name physbench_video python=3.10
conda activate physbench_video
pip install -r requirements-video.txt
```

### (Selective) Prepare 74 VLMs

Specifically, we have implemented 74 models within the `models` directory, which can be installed in a single step using the method outlined below.

- **Closed-Source Models**: To integrate closed-source models (such as `GPT`, `Gemini`, and `Claude`), you will need to configure your 🔑 API key in the file `eval/models/qa_model/imageqa_model.py`.
- **Open-Source Models**: All open-source models can be automatically downloaded via Hugging Face. You can refer to `eval/models/qa_model/imageqa_model.py` to identify the relevant keys. For example, the key "Mantis-llava-7b" will invoke the "Mantis" model class, which will automatically download the model from the Hugging Face repository "TIGER-Lab/Mantis-llava-7b".

    ```python
    "Mantis-llava-7b"                      : ("Mantis", 		 "TIGER-Lab/Mantis-llava-7b"),
    ```

### Test for You Model

##### Step 1

To adapt your model, you can refer to our implementation of 39 models within the `eval/models/qa_model/imageqa_model.py` file. You may create your own class, which should primarily include the methods `__init__` and `qa`, ensuring that both methods conform to the specified interfaces.

```shell
class YourModel(QAModelInstance):
	def __init__(self, ckpt, torch_device=torch.device("cuda"), model_precision=torch.float16, num_video_frames=8):
		### load your model
		self.model = ...
		self.tokenizer = ...
	def qa(self, image, prompt, mode):
		### give answer for each item
        output_ids = self.model.generate(
            input_ids,
            images=[images_tensor],
            do_sample=False,
            temperature=0.1,
            top_p=None,
            ... # other your config
        )
		outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
		answer = ...
		return answer # return the answer
```

Once implemented, you can add your model to the `imageqa_models` section at the top of the file, following the format below:

| model name                | class          | checkpoint path                                     |
| ------------------------- | -------------- | --------------------------------------------------- |
| "instructblip-flan-t5-xl" | "InstructBlip" | "./eval/models/checkpoints/instructblip-flan-t5-xl" |

For video-based models, you may refer to `eval/models/qa_model/videoqa_model.py`.

##### Step 2

You will need to take the following steps:

- Add your `model_name` to the `task_split` section in the file located at `eval/eval_utils/task_evaluator.py`.

  `task_split` 中 `"instructblip-flan-t5-xl"        : "image-only"`,`

  For each model value, there are 3 options: `image-only` means only one image is input, `image&video` means only one video is input (actually the test is image-only + image&video), and `general` means interleaved data (actually the test is image-only + image&video + general)

- In the `PhysionBenchEvaluator` class, modify the `test` method to invoke the `prompt` interface of your model.

After completing these changes, you can execute your model using the provided script!

### Run the evaluation script

```shell
CUDA_VISIBLE_DEVICES=9 PYTHONPATH='./' python eval/test_benchmark.py --model_name gpt4o --dataset_path ./eval/physbench
```

After running the script, a file named `[model_name].json` will be generated in the `./eval/physbench/results` directory. Please upload this file to [🔗 EvalAI](https://eval.ai/web/challenges/challenge-page/2461/overview) to automatically evaluate the results.

- [Common Case] We also provide a 📃[tested sample file](https://github.com/USC-GVL/PhysBench/tree/main/eval/physbench/test_case.json), which can be referenced to understand the required JSON submission format. Please select **Test Phase in [🔗 EvalAI](https://eval.ai/web/challenges/challenge-page/2461/overview)**

- [For model only support one image input] If the model you are testing is a model like LLaVA-1.5 that only supports one image input, you need to select `image&video` instead of the `general` above. At this time, you are actually testing the results of removing interleaved items, which is the experiment in the main table in our paper. We also provide 📃[tested sample file w/o interleaved](https://github.com/USC-GVL/PhysBench/tree/main/eval/physbench/test_case.json). Please select **Dev Phase in [🔗 EvalAI](https://eval.ai/web/challenges/challenge-page/2461/overview)**

    > (You need to pay special attention to that for models like LLaVA-1.5 that only support one image input, you need to stitch multiple frames of the item with mode `image&video` (only one video input) into one frame for input. We do not support the evaluation of image-only models due to the problem of the EvalAI platform

### Some  cases

In `case` floder, you can find how to use `eval_utils` get the results.



😀 We sincerely welcome any questions or inquiries and encourage open communication.
