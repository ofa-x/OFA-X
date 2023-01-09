import argparse
import pandas as pd
import json


def expl_score(st):
    """Converts explanation score to a number"""
    if st == "yes":
        return 1
    elif st == "weak_yes":
        return 2 / 3
    elif st == "weak_no":
        return 1 / 3
    else:
        return 0


def to_expl_score(ans_dict):
    for k, v in ans_dict.items():
        if v:
            return expl_score(k)


def to_expl_value(ans_dict):
    for k, v in ans_dict.items():
        if v:
            return k


def main(args):
    # Read data from csv
    df = pd.read_csv(args.input)
    total_submissions = len(df)

    # Filter out rejected submissions
    filter = ["Submitted", "Approved"]
    df = df.query("AssignmentStatus in @filter")
    print(f"A total of {total_submissions} assignments were submitted.")
    print(f"{len(df)} assigments left after rejecting those with < 3 of 5 answers correct.")
    # We reject all submissions with less than 3 correct answers only for the first survey iteration
    # We choose to keep all submissions for the second iteration to avoid artificially biasing the results
    # and iterating endlessly.
    # We reject these submissions only because there were too many, reducing sample size too far.

    time_worked = df.WorkTimeInSeconds.mean() / 60
    print(f"Workers spent on average {time_worked:.2f} minutes on the task.")

    # Create new df to save processed results
    results_df = pd.DataFrame(columns=["assignment_id", "img_id", "answer_correct",
                                       "explanation_ours", "expl_score_ours", "confusing_ours", "incorrect_ours",
                                       "insufficient_ours",
                                       "explanation_gt", "expl_score_gt", "confusing_gt", "incorrect_gt",
                                       "insufficient_gt"])

    # Each row the results contains 5 questions (i.e. tasks)
    for i, row in df.iterrows():
        answer = row["Answer.taskAnswers"]
        ans = json.loads(answer)[0]
        for task in range(1, 6):
            if "esnlive" in args.input:
                selected_answer = ans[f"task{task}_image_statement_relationship"]
            else:
                selected_answer = ans[f"task{task}_answer"]
            correct_answer = selected_answer[row[f"Input.answer_{task - 1}"]]
            task_results = {
                "assignment_id": row["AssignmentId"],
                "img_id": ans[f"task{task}_image_id"],
                "answer_correct": correct_answer,
                "explanation_ours": row[f"Input.output_explanation_{task - 1}"],
                "explanation_gt": row[f"Input.explanation_{task - 1}"],
                "expl_value_our": to_expl_value(ans[f"task{task}_explanation1_explanation_justifies_answer"]),
                "expl_value_gt": to_expl_value(ans[f"task{task}_explanation2_explanation_justifies_answer"]),
                "expl_score_ours": to_expl_score(ans[f"task{task}_explanation1_explanation_justifies_answer"]),
                "expl_score_gt": to_expl_score(ans[f"task{task}_explanation2_explanation_justifies_answer"]),
                "confusing_ours": ans[f"task{task}_explanation1_shortcomings_confusing_sentence"],
                "incorrect_ours": ans[f"task{task}_explanation1_shortcomings_incorrect_description"],
                "insufficient_ours": ans[f"task{task}_explanation1_shortcomings_insufficient_justification"],
                "confusing_gt": ans[f"task{task}_explanation1_shortcomings_confusing_sentence"],
                "incorrect_gt": ans[f"task{task}_explanation1_shortcomings_incorrect_description"],
                "insufficient_gt": ans[f"task{task}_explanation1_shortcomings_insufficient_justification"],
                "prefer_ours": ans[f"task{task}_preference"]["expl_1"],
                "prefer_gt": ans[f"task{task}_preference"]["expl_2"],
                "prefer_none": ans[f"task{task}_preference"]["no_preference"],
            }
            results_df = pd.concat([results_df, pd.DataFrame(task_results)], axis=0)

    expl_values = results_df.expl_value_our.groupby("selected").value_counts()
    print(f"Explanation answers (ours):")
    print(expl_values / expl_values.sum())
    expl_values = results_df.expl_value_gt.groupby("selected").value_counts()
    print(f"Explanation answers (ground truth):")
    print(expl_values / expl_values.sum())

    # Print results
    print(f"Our expl score: {results_df.expl_score_ours.mean():.4f}")
    print(f"GT expl score {results_df.expl_score_gt.mean():.4f}")
    print(f"{results_df.prefer_ours.mean() * 100:.1f}% prefer our explanation")
    print(f"{results_df.prefer_gt.mean() * 100:.1f}% prefer the ground truth explanation")
    print(f"{results_df.answer_correct.mean() * 100:.1f}% got the answer correct")

    # Filter out the ones where the answer was correct
    filtered_results = results_df[results_df.answer_correct == True]
    print("After filtering out incorrect answers:")
    print(f"Our expl score: {filtered_results.expl_score_ours.mean():.4f}")
    print(f"GT expl score {filtered_results.expl_score_gt.mean():.4f}")
    print(f"{filtered_results.prefer_ours.mean() * 100:.1f}% prefer our explanation")
    print(f"{filtered_results.prefer_gt.mean() * 100:.1f}% prefer the ground truth explanation")
    print(f"{filtered_results.answer_correct.mean() * 100:.1f}% got the answer correct")

    # Print explanation shortcomings
    print(f"Confusing sentence: {filtered_results.confusing_ours.mean():.4f}")
    print(f"Insufficient justification: {filtered_results.insufficient_ours.mean():.4f}")
    print(f"Incorrect description of image: {filtered_results.incorrect_ours.mean():.4f}")

    # %%
    print("Saving results to csv")
    results_df.to_csv(args.output)
    filtered_results.to_csv(args.output.replace(".csv", "_filtered.csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='results/vqax/vqax_results_raw.csv')
    parser.add_argument('--output', type=str, default='results/vqax/vqax_results_processed.csv')
    args = parser.parse_args()
    main(args)
