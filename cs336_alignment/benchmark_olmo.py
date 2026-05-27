from cs336_alignment.vllm_utils import *
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
import os
import json
from cs336_alignment.modal_utils import DATA_PATH, app, quote_command, submit_commands
def grade_answers(reward_fn, answers, ground_truths, dir, prompts):
    os.makedirs(f"{DATA_PATH}/{dir}", exist_ok=True)
    both = []
    format = []
    neither = []

    for i, answer in enumerate(answers):
        gt = ground_truths[i]
        gt = gt.split("####")[-1].strip()
        ans_txt = answer.text
        res = reward_fn(ans_txt, gt)
        if res["format_reward"] == 1:
            if res["answer_reward"] == 1:
                both.append({"gt": gt, "response": ans_txt, "question": prompts[i]})
            else:
                format.append({"gt": gt, "response": ans_txt, "question": prompts[i]})
        else:
            neither.append({"gt": gt, "response": ans_txt, "question": prompts[i]})
    num_both = len(both)
    num_format = len(format)
    num_neither = len(neither)
    print("Both: ", num_both,"Format only: ", num_format, "Neither: ",num_neither)
   
    with open(f"{DATA_PATH}/{dir}/format.json", "w") as f:
        json.dump(format, f)
    with open(f"{DATA_PATH}/{dir}/both.json", "w") as f:
        json.dump(both, f)
    with open(f"{DATA_PATH}/{dir}/neither.json", "w") as f:
        json.dump(neither, f)


def r1_zero(q_n_a, server):
    print("Starting r1_zero format analysis")
    with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        prompt = f.read()
    prompts = []
    ground_truths = []
    for data_pt in q_n_a:
        new_prompt = prompt.replace("{question}", data_pt["question"])
        prompts.append(new_prompt)
        ground_truths.append(data_pt["answer"])
    sampling_params = {"temperature":1,
            "max_tokens": 512,
            "n": 1,
            "seed": 0,
            "top_p": 1}
    sampling_params['stop'] = ["</answer>"]
    sampling_params['include_stop_str_in_output'] = True
    batch_size = 256
    answers = server.generate_completions(prompts, sampling_params, batch_size)
    dir = "cs336_alignment/benchmark_results_3/r1_zero"
    grade_answers(r1_zero_reward_fn,answers, ground_truths, dir, prompts)
    
    


def r1_three_shot(q_n_a, server):
    print("Starting r1_three shot format benchmarking")
    with open("cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt", "r") as f:
        prompt = f.read()
    prompts = []
    ground_truths = []
    for data_pt in q_n_a:
        new_prompt = prompt.replace("{question}", data_pt["question"])
        prompts.append(new_prompt)
        ground_truths.append(data_pt["answer"])
    sampling_params = {"temperature":1,
            "max_tokens": 512,
            "n": 1,
            "seed": 0,
            "top_p": 1}
    sampling_params['stop'] = ["</answer>"]
    sampling_params['include_stop_str_in_output'] = True
    batch_size = 256
    answers = server.generate_completions(prompts, sampling_params, batch_size)
    dir = "cs336_alignment/benchmark_results_3/r1_three"
    grade_answers(r1_zero_reward_fn,answers, ground_truths, dir, prompts)
    

def question_only(q_n_a, server):
    print("Starting question only format benchmarking")
    with open("cs336_alignment/prompts/question_only.prompt", "r") as f:
        prompt = f.read()
    prompts = []
    ground_truths = []
    for data_pt in q_n_a:
        new_prompt = prompt.replace("{question}", data_pt["question"])
        prompts.append(new_prompt)
        ground_truths.append(data_pt["answer"])
    sampling_params = {"temperature":1,
            "max_tokens": 512,
            "n": 1,
            "seed": 0,
            "top_p": 1}
    batch_size = 256
    answers = server.generate_completions(prompts, sampling_params, batch_size)
    dir = "cs336_alignment/benchmark_results_3/question_only"
    grade_answers(question_only_reward_fn,answers, ground_truths, dir, prompts)


def main():
    model_id = "allenai/OLMo-2-0425-1B"
    q_n_a = []
    with open("data/gsm8k/test.jsonl", "r") as f:
        for line in f:
            q_n_a.append(json.loads(line))
    server = VLLMServer(model_id)
    server.start()
    question_only(q_n_a, server)
    r1_zero(q_n_a, server)
    r1_three_shot(q_n_a, server)
    print("Done")
    server.stop()


if __name__ == "__main__":
    main()