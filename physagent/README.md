# PhysAgent

### 🚀 Prepare Environment

To use PhysAgent, you need to install additional dependencies:

```shell
pip install supervision pycocotools hydra-core iopath
```

Download the pretrained `SAM 2` checkpoints:

```shell
cd eval/physagent/checkpoints
bash download_ckpts.sh
```

Fill in your `<GPT API>` in `./eval/physagent/functional_tools/wrap_model.py`

Other related configurations will be automatically downloaded during runtime.

### 🔬 Test

```shell
CUDA_VISIBLE_DEVICES=2 PYTHONPATH='./' python eval/physagent/eval_physagent.py
```

The results may slightly differ from those in the paper, as the results in the paper were obtained under manual classification. Since the categories in PhysBench-test are not publicly available, this implementation uses a neuro-symbolic classification method, which may lead to differences when using automatic classification. Additionally, we observed during our experiments that the random seed and version of the API can also affect the results.

And because DINO's API was closed, we changed some online function calls to offline result calls.
