import pandas.io.parsers as parser


class Loader:

    def __init__(self, csv_file, split_pairs, filter_headers):
        self.split_p = split_pairs
        self.data = dict()
        self.splits = dict()

        self.raw = parser.read_csv(csv_file, usecols=filter_headers)
        self._split()
        self.headers = self.raw.columns.tolist()
        self._objectify()

    def _split(self):
        for key in self.split_p:
            self.splits[key] = dict()
            # TODO : This is a hack
            if key == "label":
                val = self.split_p[key]
                self.splits[key]["gt" + str(val)] = self.raw[self.raw[key] > val]
            else:
                for val in self.split_p[key]:
                    self.splits[key][val] = self.raw[self.raw[key] == val]
                    self.splits[key][val] = self.splits[key][val].drop(key, 1)
            self.raw = self.raw.drop(key, 1)

    def _objectify(self):
        for k in self.splits.keys():
            self.data[k] = dict()
            for kv in self.splits[k].keys():
                self.data[k][kv] = self.splits[k][kv].values.tolist()
        for k in self.data.keys():
            for kv in self.data[k].keys():
                for obj_idx in range(len(self.data[k])):
                    val = self.data[k][kv][obj_idx]
                    obj = dict()
                    for key_idx in range(len(self.headers)):
                        obj[self.headers[key_idx]] = val[key_idx]
                    self.data[k][kv][obj_idx] = obj

    def apply_category_encoding(self):
        self.raw["category"] = self.raw["category"].map({"compo": 1, "jam": 0})

    def map_column(self, column, mapping):
        self.raw[column] = self.raw[column].map(lambda x: mapping(x))

    def map_column_to(self, src, dest, mapping):
        self.raw[dest] = self.raw[src].map(lambda x: mapping(x))

    def map_column_to_new(self, src, dest, mapping, index):
        new_col = self.raw[src].map(lambda x: mapping(x))
        self.raw.insert(index, dest, new_col)
        self.headers = self.raw.columns.tolist()

    def remove_column(self, col):
        del self.raw[col]
        self.headers = self.raw.columns.tolist()
