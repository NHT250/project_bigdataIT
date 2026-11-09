import json

import prepare

ROW = {
    "paper_id": "abc",
    "source": ["First sentence.", "Second sentence.", "Third sentence."],
    "target": ["Author written tldr.", "Reviewer written tldr."],
    "title": "A title",
}


def test_input_is_the_whole_abstract():
    assert prepare.to_record(ROW)["input"] == "First sentence. Second sentence. Third sentence."


def test_output_is_the_author_tldr_not_text_from_the_abstract():
    assert prepare.to_record(ROW)["output"] == "Author written tldr."


def test_instruction_asks_for_what_the_labels_contain():
    assert prepare.to_record(ROW)["instruction"] == prepare.INSTRUCTION
    assert "one sentence" in prepare.INSTRUCTION


def test_every_reference_is_kept_for_scoring():
    assert prepare.to_record(ROW)["references"] == ROW["target"]


def test_first_sentences_are_kept_for_the_baseline():
    assert prepare.to_record(ROW)["sentences"] == ROW["source"]


def test_rows_without_abstract_or_tldr_are_dropped():
    assert prepare.to_record({**ROW, "target": []}) is None
    assert prepare.to_record({**ROW, "source": []}) is None


def test_convert_writes_one_json_object_per_kept_row(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text(
        json.dumps(ROW) + "\n" + json.dumps({**ROW, "target": []}) + "\n", encoding="utf-8"
    )
    dst = tmp_path / "out.jsonl"

    assert prepare.convert(src, dst) == 1

    lines = dst.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["output"] == "Author written tldr."
