
import json
import itertools
from rag import get_answer
from queue import Queue
import threading
from rich.console import Console
import re  

console = Console()

#params
BM_WEIGHTS = [0.4]
EMBEDDING_WEIGHTS = [0.6]
MAX_STEPS_LIST = [2]
NUM_PATHS_LIST = [2]
with open('dataset.json', 'r', encoding='utf-8') as f:
    test_questions = json.load(f)

param_combinations = list(itertools.product(BM_WEIGHTS, EMBEDDING_WEIGHTS, MAX_STEPS_LIST, NUM_PATHS_LIST))

results = []

def process_question(question, params):
    query = question['question']
    options = question['options']
    correct_answer = question['answer']
    complexity = question.get('complexity', 1) 
    bm_weight, embedding_weight, max_steps, num_paths = params
    formatted_options = "\n".join([f"{idx}. {option}" for idx, option in enumerate(options)])
    full_query = f"Вопрос:{query}\n\nВарианты ответов:\n{formatted_options}\n\n Выбери ТОЛЬКО ОДИН вариант ответа, выбирать несколько нельзя и ответь только номером правильного ответа после слов Финальный ответ:."

    token_queue = Queue()
    result_container = []  #container to store the final answer

    def run_get_answer():
        get_answer(
            full_query, 
            token_queue=token_queue, 
            bm_weight=bm_weight, 
            embedding_weight=embedding_weight, 
            max_steps=max_steps, 
            num_paths=num_paths,
            result_container=result_container  
        )
        token_queue.put(None)

    threading.Thread(target=run_get_answer, daemon=True).start()

    bot_message = ""
    while True:
        token = token_queue.get()
        if token is None:
            break
        bot_message += token
    if result_container: 
        final_answer_str = result_container[-1]
        console.print(f"Final answer: {final_answer_str}", style="bold red")
        pattern = r'финальный ответ:\s*(\d+)'#regex magic
        match = re.search(pattern, final_answer_str, re.IGNORECASE)
        if match:
            final_answer = int(match.group(1))
            console.print(f"Extracted: {final_answer}", style="bold green")
        else:
            final_answer = -1  #fail, answer not found
            console.print(f"Could not find a number after 'финальный ответ:' in: '{final_answer_str}'", style="bold red")
    else:
        final_answer = -1  
        console.print("No final answer received", style="bold red")

    console.print(bot_message, style="bold red")

    is_correct = final_answer == correct_answer

    if complexity in complexity_total:
        complexity_total[complexity] += 1
        if is_correct:
            complexity_correct[complexity] += 1
    else:
        complexity_total[complexity] = 1
        if is_correct:
            complexity_correct[complexity] = 1
        else:
            complexity_correct[complexity] = 0

    if is_correct:
        console.print(f"Selected answer: {final_answer} - Correct", style="bold green")
    else:
        console.print(f"Selected answer: {final_answer} - Incorrect (Expected: {correct_answer})", style="bold red")

    return is_correct, final_answer

for idx, params in enumerate(param_combinations):
    bm_weight, embedding_weight, max_steps, num_paths = params
    print(f"Testing combination {idx+1}/{len(param_combinations)}: BM_WEIGHT={bm_weight}, EMBEDDING_WEIGHT={embedding_weight}, MAX_STEPS={max_steps}, NUM_PATHS={num_paths}")
    correct_count = 0
    total = len(test_questions)
    complexity_correct = {1: 0, 2: 0, 3: 0, 4: 0}
    complexity_total = {1: 0, 2: 0, 3: 0, 4: 0}
    for question in test_questions:
        is_correct, final_answer = process_question(question, params)
        if is_correct:
            correct_count += 1
    per_complexity_accuracy = {}
    for level in sorted(complexity_total.keys()):
        correct = complexity_correct.get(level, 0)
        total_q = complexity_total.get(level, 0)
        accuracy = (correct / total_q) * 100 if total_q > 0 else 0
        per_complexity_accuracy[f"Complexity_{level}"] = {
            "Correct": correct,
            "Total": total_q,
            "Accuracy": round(accuracy, 2)
        }
    accuracy = correct_count / total * 100
    results.append({
        "BM_WEIGHT": bm_weight,
        "EMBEDDING_WEIGHT": embedding_weight,
        "MAX_STEPS": max_steps,
        "NUM_PATHS": num_paths,
        "Correct": correct_count,
        "Total": total,
        "Accuracy": round(accuracy, 2),
        "Per_Complexity_Accuracy": per_complexity_accuracy
    })
    print(f"Accuracy: {accuracy:.2f}%\n")

print("\n!=== Final  ===!\n")


total_correct = sum([res["Correct"] for res in results])
total_questions = sum([res["Total"] for res in results])
overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_questions})\n")
aggregated_complexity_correct = {1: 0, 2: 0, 3: 0, 4: 0}
aggregated_complexity_total = {1: 0, 2: 0, 3: 0, 4: 0}

for res in results:
    per_comp = res.get("Per_Complexity_Accuracy", {})
    for comp_key, stats in per_comp.items():
        level = int(comp_key.split("_")[1])
        aggregated_complexity_correct[level] += stats.get("Correct", 0)
        aggregated_complexity_total[level] += stats.get("Total", 0)

print("compelxity:")
for level in sorted(aggregated_complexity_total.keys()):
    correct = aggregated_complexity_correct.get(level, 0)
    total_q = aggregated_complexity_total.get(level, 0)
    accuracy = (correct / total_q) * 100 if total_q > 0 else 0
    print(f"  complexity {level}: {accuracy:.2f}% ({correct}/{total_q})")

with open('test_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("\nsaved to test_results.json")
