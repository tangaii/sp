"""Strict grammar for the pinned result_Tadpole.txt format."""
import ast
import re
from pathlib import Path

NUMBER = r"-?\d+\.\d{4}"
EPOCH = re.compile(
    rf"(?P<epoch> {{0,2}}\d+)/(?P<limit>\d+) trn-loss-MF: ({NUMBER})  "
    rf"trn-loss-MF: ({NUMBER}) trn-loss-MF: ({NUMBER}) trn-auc: ({NUMBER})  "
    rf"acc: ({NUMBER}) val-auc: ({NUMBER})  acc: ({NUMBER})"
)
TIME = re.compile(rf"time: ({NUMBER}) sec\.")
TEST = re.compile(rf"tst-auc: ({NUMBER})  acc: ({NUMBER})")
BOUNDARY = re.compile(r"\((\d+),\)")
COUNTER = re.compile(r"Counter\(\{(?:[0-2]: \d+, ){2}[0-2]: \d+\}\)")
MEANS = re.compile(r"val=(\d+\.\d{2})% tst_acc=(\d+\.\d{2})% tst_auc=(\d+\.\d{2})%")
SD = re.compile(rf"tst_acc_std=({NUMBER}) tst_auc_std=({NUMBER})")


def parse(path: Path, limit: int):
    lines = iter(path.read_text().splitlines())
    folds = []
    for fold in range(1, 11):
        head = next(lines)
        assert COUNTER.fullmatch(head), (path, fold, head)
        counter = ast.literal_eval(head.removeprefix("Counter(").removesuffix(")"))
        assert set(counter) == {0, 1, 2} and all(value > 0 for value in counter.values())
        record = {"counter": head, "epochs": []}
        line = next(lines)
        while (match := EPOCH.fullmatch(line)) is not None:
            epoch_string, max_epoch, *fields = match.groups()
            index = len(record["epochs"])
            assert epoch_string == f"{index:3d}" and int(max_epoch) == limit and index < limit
            record["epochs"].append(fields)
            line = next(lines)
        assert record["epochs"] and TIME.fullmatch(line), (path, fold, line)
        record["time"] = TIME.fullmatch(line)[1]
        line = next(lines)
        assert TEST.fullmatch(line), (path, fold, line)
        record["test"] = list(TEST.fullmatch(line).groups())
        for prefix in ("sens", "spes"):
            line = next(lines)
            assert line.startswith(prefix + ": ")
            value = ast.literal_eval(line[len(prefix) + 2:])
            assert isinstance(value, list) and len(value) == fold and all(
                isinstance(row, list) and len(row) == 3 and all(
                    isinstance(v, (float, int)) and 0 <= v <= 1 for v in row)
                for row in value)
            record[prefix] = value
        if fold < 10:
            shapes = [BOUNDARY.fullmatch(next(lines)) for _ in range(2)]
            assert all(shapes), (path, fold, "train/test shape")
            record["shape"] = [int(item[1]) for item in shapes]
        folds.append(record)
    footer = [next(lines) for _ in range(5)]
    vector = r"\[ *-?\d+\.\d*(?: +\d+\.\d*){2} *\]"
    for line, name in zip(footer, ("SEN", "SPE")):
        assert re.fullmatch(rf"{name}_mean: {vector} {name}_std: {vector}", line), (path, line)
    assert MEANS.fullmatch(footer[2]) and SD.fullmatch(footer[3]), (path, footer)
    final = ast.literal_eval(footer[4])
    assert isinstance(final, tuple) and len(final) == 2 and all(isinstance(j, float) for j in final)
    assert next(lines, None) is None
    return folds, footer
