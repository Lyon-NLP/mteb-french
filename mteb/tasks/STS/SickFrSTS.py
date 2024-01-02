from ...abstasks.AbsTaskSTS import AbsTaskSTS


class SickFrSTS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "SICKFr",
            "hf_hub_name": "lyon-nlp/sick-fr",
            "description": "SICK dataset french version",
            "reference": "https://huggingface.co/datasets/Lajavaness/SICK-fr",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["fr"],
            "main_score": "cosine_spearman",
            "min_score": 1,
            "max_score": 5,
            "revision": "ad1bddc5f1407c30089cdeaf4f5d88792db50f24",
        }
