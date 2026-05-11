import re
import ast
import json
import argparse
from tqdm import tqdm
from vqa_metric import (
    vqa_evaluation,
    cn_vqa_evaluation,
    math_expression_evaluation,
    vqa_evaluation_case_sensitive,
    counting_evaluation,
    cn_math_expression_evaluation,
)
from IoUscore_metric import (
    vqa_with_position_evaluation,
    calculate_iou,
    extract_coordinates,
)
from TEDS_metric import (
    TEDS,
    convert_markdown_table_to_html,
    convert_str_to_dict,
    convert_str_to_multi_dict,
    generate_combinations,
    dict_to_html,
    compute_f1_score,
    doc_parsing_evaluation,
    wrap_html_table,
)
from page_ocr_metric import cal_per_metrics
from spotting_metric import spotting_evaluation
from eval_text_spotting_en import extract_bounding_boxes, spotting_evaluation_normalized


def is_nan_value(value):
    if value is None:
        return True
    if isinstance(value, str) and value.lower() == "nan":
        return True
    try:
        import pandas as pd

        if pd.isna(value):
            return True
    except:
        pass
    return False


def get_value_or_zero(value):
    return 0.0 if value is None else value


def process_predictions(input_path, output_path):
    with open(input_path, "r") as f:
        predict_file = json.load(f)

    teds = TEDS(n_jobs=32)

    task_type_list = [
        "APP agent en",
        "ASCII art classification en",
        "key information extraction en",
        "key information mapping en",
        "math QA en",
        "full-page OCR en",
        "reasoning VQA en",
        "fine-grained text recognition en",
        "science QA en",
        "table parsing en",
        "text counting en",
        "text grounding en",
        "text recognition en",
        "text spotting en",
        "document classification en",
        "cognition VQA en",
        "VQA with position en",
        "chart parsing en",
        "document parsing en",
        "formula recognition en",
        "diagram QA en",
        "cognition VQA cn",
        "key information extraction cn",
        "formula recognition cn",
        "full-page OCR cn",
        "reasoning VQA cn",
        "text translation cn",
        "table parsing cn",
        "handwritten answer extraction cn",
        "document parsing cn",
    ]

    res_data_list = []

    for index, data_item in enumerate(tqdm(predict_file)):
        if (
            data_item["type"] == "APP agent en"
            or data_item["type"] == "ASCII art classification en"
            or data_item["type"] == "math QA en"
            or data_item["type"] == "reasoning VQA en"
            or data_item["type"] == "science QA en"
            or data_item["type"] == "text recognition en"
            or data_item["type"] == "document classification en"
            or data_item["type"] == "cognition VQA en"
            or data_item["type"] == "diagram QA en"
        ):
            # Infer eval method when missing
            eval_type = data_item.get("eval")
            if not eval_type:
                if (len(data_item["answers"]) == 1
                    and len(data_item["answers"][0]) <= 2
                    and data_item["answers"][0].isalpha()):
                    eval_type = "multiple choice"
            if eval_type == "multiple choice":
                if not isinstance(data_item["answers"], list):
                    data_item["answers"] = [data_item["answers"]]
                assert len(data_item["answers"]) == 1

                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                else:
                    predict = "".join(
                        c for c in data_item["predict"] if c.isalpha()
                    )

                    if predict == data_item["answers"][0]:
                        data_item["score"] = 1
                    else:
                        data_item["score"] = 0
            elif eval_type == "case sensitive":
                data_item["score"] = vqa_evaluation_case_sensitive(
                    data_item["predict"], data_item["answers"]
                )
            else:
                data_item["score"] = vqa_evaluation(
                    data_item["predict"], data_item["answers"]
                )

        elif (
            data_item["type"] == "cognition VQA cn"
            or data_item["type"] == "reasoning VQA cn"
        ):
            eval_type = data_item.get("eval")
            if eval_type == "multiple choice":
                assert len(data_item["answers"]) == 1
                predict = "".join(c for c in data_item["predict"] if c.isalpha())

                if predict == data_item["answers"][0]:
                    data_item["score"] = 1
                else:
                    data_item["score"] = 0
            elif eval_type == "case sensitive":
                data_item["score"] = vqa_evaluation_case_sensitive(
                    data_item["predict"], data_item["answers"]
                )
            else:
                data_item["score"] = cn_vqa_evaluation(
                    data_item["predict"], data_item["answers"]
                )

        elif data_item["type"] == "handwritten answer extraction cn":
            if "简答" in data_item["question"]:
                ocr_metric = cal_per_metrics(
                    data_item["predict"], data_item["answers"][0]
                )
                data_item["score"] = (
                    get_value_or_zero(ocr_metric["bleu"])
                    + get_value_or_zero(ocr_metric["meteor"])
                    + get_value_or_zero(ocr_metric["f_measure"])
                    + (1 - get_value_or_zero(ocr_metric["edit_dist"]))
                ) / 4
            else:
                assert len(data_item["answers"]) == 1
                answer = data_item["answers"][0]
                chars = list(answer)
                if len(answer) > 1:
                    answer_list = [
                        "".join(chars),
                        ".".join(chars),
                        ". ".join(chars),
                        ",".join(chars),
                        ", ".join(chars),
                        "、".join(chars),
                        ";".join(chars),
                        "; ".join(chars),
                        " ".join(chars),
                        "和".join(chars),
                    ]
                    max_score = 0
                    for answer in answer_list:
                        if answer in data_item["predict"]:
                            temp_score = 1
                        else:
                            temp_score = 0
                        if temp_score > max_score:
                            max_score = temp_score
                    data_item["score"] = max_score

                else:
                    if data_item["answers"][0] in data_item["predict"]:
                        data_item["score"] = 1
                    else:
                        data_item["score"] = 0

        elif data_item["type"] == "formula recognition cn":
            if is_nan_value(data_item["predict"]):
                data_item["score"] = 0
            else:
                data_item["score"] = cn_math_expression_evaluation(
                    data_item["predict"], data_item["answers"]
                )

        elif data_item["type"] == "text counting en":
            # Infer eval method when missing: numeric-only answers → regression, else exact match
            eval_method = data_item.get("eval")
            if not eval_method:
                if all(a.strip().isdigit() for a in data_item["answers"]):
                    eval_method = "regression"
                else:
                    eval_method = "exact match"
            data_item["score"] = counting_evaluation(
                data_item["predict"],
                data_item["answers"],
                eval_method,
            )

        elif data_item["type"] == "formula recognition en":
            data_item["score"] = math_expression_evaluation(
                data_item["predict"], data_item["answers"]
            )

        elif data_item["type"] == "table parsing en":
            if type(data_item["answers"]) == list and len(data_item["answers"]) == 1:
                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                elif not isinstance(data_item["question"], str):
                    data_item["ignore"] = "True"
                    data_item["score"] = 0

                elif "html" in data_item["question"].lower():
                    no_find = False
                    predict_table = data_item["predict"].replace("\n", "")
                    if "<body" in predict_table:
                        predict_table = re.findall("<body.*", predict_table)[0]
                    elif "<table" in predict_table:
                        predict_table = re.findall("<table.*", predict_table)[0]
                    else:
                        no_find = True

                    if no_find:
                        data_item["score"] = 0
                    else:
                        pred_table_html = wrap_html_table(predict_table)
                        gold_table_html = wrap_html_table(data_item["answers"][0])
                        try:
                            data_item["score"] = teds.evaluate(
                                pred_table_html, gold_table_html
                            )
                        except:
                            data_item["score"] = 0

                elif "markdown" in data_item["question"].lower():
                    if not isinstance(data_item["predict"], str):
                        prediction = str(data_item["predict"])
                        pred_table_html = convert_markdown_table_to_html(prediction)
                        gt_table_html = convert_markdown_table_to_html(
                            data_item["answers"][0]
                        )
                        data_item["score"] = teds.evaluate(
                            pred_table_html, gt_table_html
                        )

                    else:
                        pred_table_html = convert_markdown_table_to_html(
                            data_item["predict"]
                        )
                        gt_table_html = convert_markdown_table_to_html(
                            data_item["answers"][0]
                        )
                        data_item["score"] = teds.evaluate(
                            pred_table_html, gt_table_html
                        )
            else:
                raise ValueError

        elif data_item["type"] == "table parsing cn":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                no_find = False
                predict_table = data_item["predict"].replace("\n", "")
                if "<body" in predict_table:
                    predict_table = re.findall("<body.*", predict_table)[0]
                elif "<table" in predict_table:
                    predict_table = re.findall("<table.*", predict_table)[0]
                else:
                    no_find = True

                if no_find:
                    data_item["score"] = 0
                else:
                    pred_table_html = wrap_html_table(predict_table)
                    gold_table_html = wrap_html_table(data_item["answers"][0])
                    try:
                        data_item["score"] = teds.evaluate(
                            pred_table_html, gold_table_html
                        )
                    except:
                        data_item["score"] = 0
                        print("error")

        elif data_item["type"] == "chart parsing en":
            answer = data_item["answers"][0]
            if isinstance(answer, str):
                try:
                    answer = ast.literal_eval(answer)
                except:
                    pass
            if data_item["predict"]:
                pred_chart_dict = convert_str_to_multi_dict(data_item["predict"])
                if len(pred_chart_dict) == 0:
                    data_item["score"] = 0
                else:
                    pred_chart_html = dict_to_html(pred_chart_dict)
                    gt_chart_html = dict_to_html(answer)
                    data_item["score"] = teds.evaluate(pred_chart_html, gt_chart_html)
            else:
                data_item["score"] = 0

        elif data_item["type"] == "document parsing en":
            assert type(data_item["answers"]) == list and len(data_item["answers"]) == 1
            data_item["score"] = doc_parsing_evaluation(
                data_item["predict"], data_item["answers"][0]
            )

        elif data_item["type"] == "document parsing cn":
            assert type(data_item["answers"]) == list and len(data_item["answers"]) == 1
            data_item["score"] = doc_parsing_evaluation(
                data_item["predict"], data_item["answers"][0]
            )

        elif (
            data_item["type"] == "key information extraction en"
            or data_item["type"] == "key information mapping en"
        ):
            assert len(data_item["answers"]) == 1
            answers = generate_combinations(data_item["answers"][0])

            if type(answers) == list and len(answers) == 1:
                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                else:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answers[0])
            else:
                max_score = 0
                for answer in answers:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answer)

                    if data_item["score"] > max_score:
                        max_score = data_item["score"]
                data_item["score"] = max_score

        elif data_item["type"] == "key information extraction cn":
            assert len(data_item["answers"]) == 1
            answers = ast.literal_eval(data_item["answers"][0])
            answers = {k: v if isinstance(v, list) else [v] for k, v in answers.items()}
            answers = generate_combinations(answers)
            if type(answers) == list and len(answers) == 1:
                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                else:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answers[0])
            else:
                max_score = 0
                for answer in answers:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answer)

                    if data_item["score"] > max_score:
                        max_score = data_item["score"]
                data_item["score"] = max_score

        elif data_item["type"] == "VQA with position en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                pred_dict = convert_str_to_dict(data_item["predict"])
                data_item["score"] = vqa_with_position_evaluation(pred_dict, data_item)

        elif data_item["type"] == "text translation cn":
            if len(data_item["predict"]) == 0:
                data_item["score"] = 0
            elif len(data_item["answers"][0]) == 0:
                data_item["score"] = 0
                data_item["ignore"] = "True"
            else:
                ocr_metric = cal_per_metrics(
                    data_item["predict"], data_item["answers"][0]
                )
                data_item["score"] = (
                    ocr_metric["bleu"]
                    + ocr_metric["meteor"]
                    + ocr_metric["f_measure"]
                    + (1 - ocr_metric["edit_dist"])
                ) / 4

        elif data_item["type"] == "fine-grained text recognition en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            elif len(data_item["predict"]) == 0:
                data_item["score"] = 0
            else:
                ocr_metric = cal_per_metrics(
                    data_item["predict"], data_item["answers"][0]
                )
                data_item["score"] = (
                    get_value_or_zero(ocr_metric["bleu"])
                    + get_value_or_zero(ocr_metric["meteor"])
                    + get_value_or_zero(ocr_metric["f_measure"])
                    + (1 - get_value_or_zero(ocr_metric["edit_dist"]))
                ) / 4
        elif data_item["type"] == "full-page OCR en":
            if not data_item["predict"]:
                data_item["score"] = 0
            else:
                ocr_metric = cal_per_metrics(
                    data_item["predict"], data_item["answers"][0]
                )
                data_item["score"] = (
                    get_value_or_zero(ocr_metric["bleu"])
                    + get_value_or_zero(ocr_metric["meteor"])
                    + get_value_or_zero(ocr_metric["f_measure"])
                    + (1 - get_value_or_zero(ocr_metric["edit_dist"]))
                ) / 4

        elif data_item["type"] == "full-page OCR cn":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                if len(data_item["predict"]) == 0:
                    data_item["score"] = 0
                else:
                    ocr_metric = cal_per_metrics(
                        data_item["predict"], data_item["answers"][0]
                    )
                    data_item["score"] = (
                        ocr_metric["bleu"]
                        + ocr_metric["meteor"]
                        + ocr_metric["f_measure"]
                        + (1 - ocr_metric["edit_dist"])
                    ) / 4

        elif data_item["type"] == "text grounding en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                predict_bbox = extract_coordinates(data_item["predict"])
                if not predict_bbox:
                    data_item["score"] = 0
                else:
                    data_item["score"] = calculate_iou(
                        predict_bbox, data_item["answers"]
                    )

        elif data_item["type"] == "text spotting en":
            # Parse bbox/content from answers if not present (HF dataset format)
            # GT format: variable-length polygons (8, 12, 16, 28+ coords) followed by text
            if "bbox" not in data_item and "answers" in data_item and data_item["answers"]:
                bboxes, contents = [], []
                for line in data_item["answers"][0].strip().split("\n"):
                    parts = line.split(",")
                    # Find where coordinates end: scan from start while parts are integers
                    num_coords = 0
                    for p in parts:
                        try:
                            int(p.strip())
                            num_coords += 1
                        except ValueError:
                            break
                    if num_coords >= 8 and num_coords < len(parts):
                        coords = [int(p.strip()) for p in parts[:num_coords]]
                        text = ",".join(parts[num_coords:])
                        x_coords = coords[0::2]
                        y_coords = coords[1::2]
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                        bboxes.append([x1, y1, x2, y1, x2, y2, x1, y2])
                        contents.append(text)
                    elif num_coords >= 8 and num_coords == len(parts) and num_coords % 2 == 1:
                        # All-numeric text (e.g. "1700"): odd coord count means last is text
                        coords = [int(p.strip()) for p in parts[:num_coords - 1]]
                        text = parts[num_coords - 1].strip()
                        x_coords = coords[0::2]
                        y_coords = coords[1::2]
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                        bboxes.append([x1, y1, x2, y1, x2, y2, x1, y2])
                        contents.append(text)
                data_item["bbox"] = bboxes
                data_item["content"] = contents
            if "bbox" not in data_item or "content" not in data_item:
                data_item["score"] = 0
            elif not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                predict_bbox = extract_bounding_boxes(data_item["predict"])
                if not predict_bbox:
                    data_item["score"] = 0
                else:
                    data_item["score"] = spotting_evaluation_normalized(predict_bbox, data_item)

        else:
            raise ValueError("Unknown task type!")

        res_data_list.append(data_item)

    for task_name in task_type_list:
        print("\n" + task_name)
        mean_score, total_len = 0, 0.0
        for item in res_data_list:
            if item["type"] == task_name:
                total_len += 1
                mean_score += item["score"]

        mean_score = mean_score / total_len if total_len > 0 else 0
        print(
            f"Task {task_name}, total instructions: {total_len}, average score: {mean_score:.3f}\n"
        )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(predict_file, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process prediction JSON files and evaluate results."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input prediction JSON file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the results JSON file.",
    )

    args = parser.parse_args()

    process_predictions(args.input_path, args.output_path)

print("End of Code!")
