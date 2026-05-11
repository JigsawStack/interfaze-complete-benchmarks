"""
Evaluation script for OCRBench v2 — Text Spotting EN only.
Scores predictions exclusively for the "text spotting en" task type.
"""

import re
import ast
import json
import argparse
from tqdm import tqdm
from spotting_metric import spotting_evaluation


def strip_markdown_fences(text):
    """Remove markdown code fences like ```json ... ``` or ```python ... ```."""
    return re.sub(r"```\w*\n?", "", text).strip()


def extract_bounding_boxes(predict_str):
    """
    Extract coordinates and text from prediction string.
    Handles markdown fences, normalizes inverted coordinates.

    Returns list of [x1, y1, x2, y2, text] or None.
    """
    predict_str = strip_markdown_fences(predict_str)

    results = []
    seen = set()

    # Try parsing with ast.literal_eval
    try:
        data = ast.literal_eval(predict_str)
    except Exception:
        data = None

    if data is not None:
        if isinstance(data, (list, tuple)):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 5:
                    try:
                        x1 = int(str(item[0]).strip())
                        y1 = int(str(item[1]).strip())
                        x2 = int(str(item[2]).strip())
                        y2 = int(str(item[3]).strip())
                    except (ValueError, TypeError):
                        continue

                    text_content = (
                        str(item[4])
                        .replace("\n", "")
                        .strip()
                        .strip('"')
                        .strip("'")
                    )

                    # Normalize inverted coordinates
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)

                    if not (
                        0 <= x1 <= 1000
                        and 0 <= y1 <= 1000
                        and 0 <= x2 <= 1000
                        and 0 <= y2 <= 1000
                    ):
                        continue

                    if x1 == x2 or y1 == y2:
                        continue

                    key = (x1, y1, x2, y2, text_content)
                    if key in seen:
                        continue

                    seen.add(key)
                    results.append([x1, y1, x2, y2, text_content])
    else:
        # Fallback: regex parsing
        items = re.findall(r"[\[\(]\s*([^\[\]\(\)]*?)\s*[\]\)]", predict_str)
        if not items:
            return None

        for item in items:
            parts = item.split(",", 4)
            if len(parts) < 5:
                continue

            try:
                x1 = int(parts[0].strip())
                y1 = int(parts[1].strip())
                x2 = int(parts[2].strip())
                y2 = int(parts[3].strip())
            except (ValueError, TypeError):
                continue

            text_content = parts[4].replace("\n", "").strip().strip('"').strip("'")

            # Normalize inverted coordinates
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            if not (
                0 <= x1 <= 1000
                and 0 <= y1 <= 1000
                and 0 <= x2 <= 1000
                and 0 <= y2 <= 1000
            ):
                continue

            if x1 == x2 or y1 == y2:
                continue

            key = (x1, y1, x2, y2, text_content)
            if key in seen:
                continue

            seen.add(key)
            results.append([x1, y1, x2, y2, text_content])

    return results if results else None


def spotting_evaluation_normalized(prediction_list, img_metas):
    """Wrapper around spotting_evaluation that normalizes inverted coords
    in the submission list instead of discarding them."""
    # Normalize predictions before passing to spotting_evaluation
    normalized = []
    for item in prediction_list:
        if len(item) != 5:
            continue
        x1, y1, x2, y2, rec = item
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x1 == x2 or y1 == y2:
            continue
        normalized.append([x1, y1, x2, y2, rec])

    if not normalized:
        return 0

    return spotting_evaluation(normalized, img_metas)


def process_predictions(input_path, output_path):
    with open(input_path, "r") as f:
        predict_file = json.load(f)

    res_data_list = []

    for index, data_item in enumerate(tqdm(predict_file, desc="Scoring text spotting")):
        if data_item["type"] != "text spotting en":
            # Skip non-text-spotting items (shouldn't be any, but just in case)
            res_data_list.append(data_item)
            continue

        # Parse bbox/content from answers if not present (HF dataset format)
        # GT format: x1,y1,x2,y2,...,xN,yN,text (variable number of coordinate pairs)
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
                # Need at least 8 coords (4 points) and some text
                if num_coords >= 8 and num_coords < len(parts):
                    coords = [int(p.strip()) for p in parts[:num_coords]]
                    text = ",".join(parts[num_coords:])
                    # Reduce polygon to axis-aligned bbox (same as spotting_metric.py)
                    x_coords = coords[0::2]
                    y_coords = coords[1::2]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    # Store as 8-point format for compatibility with spotting_evaluation
                    bboxes.append([x1, y1, x2, y1, x2, y2, x1, y2])
                    contents.append(text)
                elif num_coords >= 8 and num_coords == len(parts):
                    # Edge case: text is purely numeric (e.g. "1700")
                    # Last value after the polygon coords is the text
                    # Polygons always have even number of coords (pairs of x,y)
                    if num_coords % 2 == 1:
                        # Odd count means last "coord" is actually the text
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
                data_item["score"] = spotting_evaluation_normalized(
                    predict_bbox, data_item
                )

        res_data_list.append(data_item)

    # Print summary
    mean_score, total_len = 0, 0
    for item in res_data_list:
        if item["type"] == "text spotting en" and "ignore" not in item:
            total_len += 1
            mean_score += item["score"]

    mean_score = mean_score / total_len if total_len > 0 else 0
    print(
        f"\nTask text spotting en, total instructions: {total_len}, average score: {mean_score:.3f}\n"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(predict_file, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate text spotting EN predictions from OCRBench v2."
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
