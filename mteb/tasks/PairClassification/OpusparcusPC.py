from ...abstasks import AbsTaskPairClassification
import datasets

_LANGUAGES = ["de", "en", "fi", "fr", "ru", "sv"]


class OpusparcusPC(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "OpusparcusPC",
            "hf_hub_name": "GEM/opusparcus",
            "description": "Opusparcus is a paraphrase corpus for six European language: German, English, Finnish, French, Russian, and Swedish. The paraphrases consist of subtitles from movies and TV shows.",
            "reference": "https://gem-benchmark.com/data_cards/opusparcus",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["test", "validation"],
            "eval_langs": _LANGUAGES,
            "main_score": "ap",
            "revision": "9e9b1f8ef51616073f47f306f7f47dd91663f86a",
        }

    def __init__(self, langs=None, **kwargs):
        super().__init__(**kwargs)
        if type(langs) is list:
            langs = [lang for lang in langs if lang in self.description["eval_langs"]]
        if langs is not None and len(langs) > 0:
            self.langs = langs  # TODO: case where user provides langs not in the dataset
        else:
            self.langs = self.description["eval_langs"]
        self.is_multilingual = True

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(
                self.description["hf_hub_name"],
                lang=lang,
                quality=100,
                revision=self.description.get("revision", None),
            )
            self.dataset_transform(lang)
        self.data_loaded = True

    def dataset_transform(self, lang):
        for split in self.dataset[lang]:
            labels = self.dataset[lang][split]["annot_score"]
            sent1 = self.dataset[lang][split]["input"]
            sent2 = self.dataset[lang][split]["target"]
            new_dict = {}
            labels = [0 if label < 3 else 1 if label > 3 else 3 for label in labels]
            neutral = [i for i, val in enumerate(labels) if val == 3]
            for i in sorted(neutral, reverse=True):
                del labels[i]
                del sent1[i]
                del sent2[i]
            new_dict["labels"] = [labels]
            new_dict["sent1"] = [sent1]
            new_dict["sent2"] = [sent2]
            self.dataset[lang][split] = datasets.Dataset.from_dict(new_dict)
