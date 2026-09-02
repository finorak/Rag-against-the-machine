"""Evaluatation module."""


import sys
from typing import Any

from .utils import secure_open, source_inside


class EvaluateEngine:
    """Class containing evaluation metrics."""

    def evaluate(
            self, results_path: str, dataset_path: str
    ) -> dict[str, float]:
        """Evaluate search results based on ground truth.

        Doing a simple recal@k calculation on the search results
        and the ground truth from dataset_path.
        For our evaluation metric, we'll use recall@1, 3, 5 and 10
        because that's the get go for a production ready RAG
        metrics.

        Parameters:
        ----------
        student_search_results_path: str
            the results of the student.
        dataset_path: str
            file containing our ground truth.
        """
        student_results = secure_open(results_path)
        ground_truth_raw_data = secure_open(dataset_path)
        ground_truth = ground_truth_raw_data.get('rag_questions')
        recall_metrics: dict[str, float] = {}
        if not ground_truth:
            print("Data not provided", file=sys.stderr)
            sys.exit(1)
        results = student_results.get('search_results')
        if not results:
            print("Data not provided", file=sys.stderr)
            sys.exit(1)
        if len(ground_truth) != len(results):
            print("Ground truth and results mismatch", file=sys.stderr)
            sys.exit(1)
        for metric in [1, 3, 5, 10]:
            score: float = 0
            for index, search_result in enumerate(results):
                current_ground_truth = ground_truth[index].get('sources')
                current_student_source = search_result.get("retrieved_sources")
                if not current_ground_truth or not current_student_source:
                    print(
                            "Data provided doesn't provide sourecs",
                            file=sys.stderr
                            )
                    sys.exit(0)
                score += self._get_scores_on_recall_k(
                        current_ground_truth, current_student_source[:metric]
                        )
            recall_metrics[f"recall@{metric}"] = round(score, 1)
        return recall_metrics

    def _intersection_over_untion(
            self,
            truth: dict[str, Any],
            source: dict[str, Any]
    ) -> float | Any:
        """Calculate intersection over Union of two source.

        By using a simple calculaction on two lines
        where line is supposed to be a point x, to y
        x is the start index of a chunk, and y the
        end index of a chunk. We calculate how much
        is the intersection between those two.


        Parameters:
        ----------
        truth: dict[str, Any]
            the ground truth to use.
        source: dict[str, ANy]
            the dataset the student retrieved.
        """
        x1 = truth['first_character_index']
        y1 = truth['last_character_index']
        x2 = source['first_character_index']
        y2 = source['last_character_index']
        pos1 = (x1, y1)
        pos2 = (x2, y2)
        x1, y1 = min(pos1), max(pos1)
        x2, y2 = min(pos2), max(pos2)
        line1 = y1 - x1
        line2 = y2 - x2
        intersection = max(0, min(y1, y2) - max(x1, x2))
        untion = line1 + line2 - intersection
        return intersection / untion

    def _get_scores_on_recall_k(
            self,
            ground_truth: list[dict[str, int | str]],
            student_sources: list[dict[str, int | str]],
    ) -> float:
        """Calculate the score of recall@k.

        For each metrics we are, we calculate the
        score by using recall@K metrics.

        Formula:
        --------
            recall_k = (true_positives@K) / \
(true_positives@K + false_negative@K)
        Returns:
        -------
            recall_metrics: dict[str, float] a metric
                we use to determine our recall.
        """
        iou_score: float = 0
        for truth in ground_truth:
            if not source_inside(truth['file_path'], student_sources):
                continue
            filtered_data = list(filter(
                lambda item: source_inside(
                    truth['file_path'], student_sources
                    ), student_sources)
            )
            for source in filtered_data:
                iou_score += self._intersection_over_untion(truth, source)
        return iou_score
