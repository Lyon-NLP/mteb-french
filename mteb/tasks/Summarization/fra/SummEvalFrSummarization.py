from __future__ import annotations

from mteb.abstasks import AbsTaskSummarization
from mteb.abstasks.TaskMetadata import TaskMetadata


class SummEvalFrSummarization(AbsTaskSummarization):
    metadata = TaskMetadata(
        name="SummEvalFr",
        description="News Article Summary Semantic Similarity Estimation translated from english to french with DeepL.",
        reference="https://github.com/Yale-LILY/SummEval",
        dataset={
            "path": "lyon-nlp/summarization-summeval-fr-p2p",
            "revision": "08e37d5b2920587685f5289d21303d72519fde3a",
        },
        type="Summarization",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="cosine_spearman",
        date=None,
        form=["written"],
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation="machine-translated",
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5

        return metadata_dict
