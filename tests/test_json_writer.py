"""Tests for the .gromit.json writer."""


from gromit.alignment.temporal import AlignedSegment
from gromit.output.json_writer import build_transcript_json
from gromit.transcription.base import Word


def _segments():
    return [
        AlignedSegment(
            start=12.34,
            end=15.10,
            speaker="SPEAKER_01",
            text="привіт",
            avg_logprob=-0.31,
            words=[Word(w="привіт", start=12.34, end=12.71, p=0.87)],
        )
    ]


def test_build_transcript_json_schema():
    payload = build_transcript_json(
        _segments(),
        language="uk",
        model="large-v3",
        hotwords_from=["data/glossary.yaml"],
    )
    assert payload["language"] == "uk"
    assert payload["model"] == "large-v3"
    assert payload["hotwords_from"] == ["data/glossary.yaml"]
    seg = payload["segments"][0]
    assert seg["speaker"] == "SPEAKER_01"
    assert seg["avg_logprob"] == -0.31
    assert seg["words"][0] == {"w": "привіт", "start": 12.34, "end": 12.71, "p": 0.87}


def test_empty_segments_produce_empty_list():
    payload = build_transcript_json(
        [], language="uk", model="large-v3", hotwords_from=[]
    )
    assert payload["segments"] == []
