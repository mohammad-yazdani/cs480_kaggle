from lib.loader import Loader


class Score:

    def __init__(self, csv_file, default=1):
        self.scores = Score.load_scores(csv_file)
        self.default = default
        self.nan = default
        for sc in self.scores:
            if sc["keys"][0] == "nan":
                self.nan = sc["score"]

    def score_tag(self, tag: str):
        if type(tag) is float:
            return self.nan
        tags = tag.split(";")
        sum_score = 0
        for sc in self.scores:
            if set(sc["keys"]).issubset(tags):
                sum_score += sc["score"]

        if sum_score == 0:
            sum_score = 1
        return sum_score

    @staticmethod
    def load_scores(csv_file):
        scores = Loader(csv_file, [], None)

        tags = scores.raw["tag"].values.tolist()
        scores = scores.raw["score"].values.tolist()

        score_meta = list()

        for idx in range(len(tags)):
            if tags[idx] != "empty":
                res = str(tags[idx]).split()
            else:
                res = [""]
            score_meta.append({"keys": res, "score": scores[idx]})

        return score_meta
