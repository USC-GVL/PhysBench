import argparse
from eval.physagent.functional_tools.physagent import AgentEvaluator
from eval.eval_utils.caculate_core import calculate_accuracy, print_accuracies

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='gpt', help="Select the model name")
    parser.add_argument("--dataset_path", type=str, default='./eval/physbench',
                        help="data you put USC-GVL/PhysBench")
    parser.add_argument("--split", type=str, default='val', choices=['val', 'test'],
                        help="Choose between 'val' or 'test' split")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
	args = parse_args()
	task_evaluator = AgentEvaluator(
		name=args.model_name, dataset_path=args.dataset_path
	)
	task_evaluator.test()

	if args.split == 'val':
		accuracies = calculate_accuracy(val_annotation_file='./eval/physbench/val_answer.json',
										user_submission_file=task_evaluator.result_file)
		print_accuracies(accuracies, name=model_name)