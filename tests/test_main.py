"""Tests de main.py.

Empezaron como tests de caracterización (paso 1). Los que documentaban
defectos se actualizaron en el paso 3 para describir el comportamiento
corregido.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import deepl
import pytest

import main
from main import Config, SubplexError, SubtitleBlock

TIMING_1 = "00:00:01,000 --> 00:00:02,000"
TIMING_2 = "00:00:03,000 --> 00:00:04,000"


@dataclass
class FakeTextResult:
    text: str


class FakeTranslator:
    """Sustituye a deepl.Translator: no usa red y registra cada llamada."""

    instances: ClassVar[list["FakeTranslator"]] = []

    def __init__(self, auth_key: str) -> None:
        self.auth_key = auth_key
        self.calls: list[tuple[str, str]] = []
        FakeTranslator.instances.append(self)

    def translate_text(self, text: str, target_lang: str) -> FakeTextResult:
        # Reproduce la validación real de deepl 1.17 (translator.py:396).
        if not text:
            raise ValueError("text must not be empty")
        self.calls.append((text, target_lang))
        return FakeTextResult(f"[{target_lang}] {text}")


@pytest.fixture
def subs(tmp_path: Path) -> Path:
    directory = tmp_path / "subs"
    directory.mkdir()
    return directory


@pytest.fixture
def config(subs: Path) -> Config:
    return Config(subtitles_dir=subs, auth_key="fake-key")


@pytest.fixture
def translator() -> FakeTranslator:
    return FakeTranslator("fake-key")


def write_srt(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def write_config(directory: Path, content: object) -> None:
    (directory / "config.json").write_text(json.dumps(content), encoding="utf-8")


# --- translate_file ---------------------------------------------------------


def test_translates_file_and_writes_output_with_language_suffix(
    subs, config, translator
):
    write_srt(
        subs,
        "movie.srt",
        f"1\n{TIMING_1}\nHello\n\n2\n{TIMING_2}\n♪ Line one ♪\nLine two\n",
    )

    target = main.translate_file("movie.srt", "ES", config, translator)

    assert target == subs / "movie_ES.srt"
    assert target.read_text(encoding="utf-8") == (
        f"1\n{TIMING_1}\n[ES] Hello\n\n2\n{TIMING_2}\n[ES] Line one\n[ES] Line two\n\n"
    )


def test_source_file_is_not_modified(subs, config, translator):
    content = f"1\n{TIMING_1}\nHello\n"
    source = write_srt(subs, "movie.srt", content)

    main.translate_file("movie.srt", "ES", config, translator)

    assert source.read_text(encoding="utf-8") == content


def test_one_api_call_per_line(subs, config, translator):
    write_srt(subs, "movie.srt", f"1\n{TIMING_1}\nA\nB\n\n2\n{TIMING_2}\nC\n")

    main.translate_file("movie.srt", "DE", config, translator)

    assert translator.calls == [("A", "DE"), ("B", "DE"), ("C", "DE")]


def test_finds_file_in_nested_subdirectory(subs, config, translator):
    nested = subs / "season1" / "ep1"
    nested.mkdir(parents=True)
    write_srt(nested, "ep.srt", f"1\n{TIMING_1}\nHi\n")

    target = main.translate_file("ep.srt", "ES", config, translator)

    assert target == nested / "ep_ES.srt"


def test_empty_srt_produces_empty_output(subs, config, translator):
    write_srt(subs, "empty.srt", "")

    target = main.translate_file("empty.srt", "ES", config, translator)

    assert target.read_text(encoding="utf-8") == ""
    assert translator.calls == []


def test_extra_blank_lines_between_blocks_are_tolerated(subs, config, translator):
    write_srt(
        subs,
        "movie.srt",
        f"\n1\n{TIMING_1}\nFirst\n\n\n\n2\n{TIMING_2}\nSecond\n\n\n",
    )

    target = main.translate_file("movie.srt", "ES", config, translator)

    assert target.read_text(encoding="utf-8") == (
        f"1\n{TIMING_1}\n[ES] First\n\n2\n{TIMING_2}\n[ES] Second\n\n"
    )


def test_music_only_lines_are_dropped_without_calling_api(subs, config, translator):
    write_srt(
        subs,
        "movie.srt",
        f"1\n{TIMING_1}\nHello\n♪\n\n2\n{TIMING_2}\n♪ ♪\n",
    )

    target = main.translate_file("movie.srt", "ES", config, translator)

    assert target.read_text(encoding="utf-8") == (
        f"1\n{TIMING_1}\n[ES] Hello\n\n2\n{TIMING_2}\n\n"
    )
    assert translator.calls == [("Hello", "ES")]


def test_non_srt_file_is_rejected(subs, config, translator):
    write_srt(subs, "notes.txt", "whatever")

    with pytest.raises(SubplexError, match=r"is not a \.srt file"):
        main.translate_file("notes.txt", "ES", config, translator)
    assert translator.calls == []


def test_uppercase_extension_is_accepted_and_preserved(subs, config, translator):
    write_srt(subs, "MOVIE.SRT", f"1\n{TIMING_1}\nHi\n")

    target = main.translate_file("MOVIE.SRT", "ES", config, translator)

    assert target == subs / "MOVIE_ES.SRT"


def test_srt_in_directory_name_does_not_affect_output_path(subs, config, translator):
    folder = subs / "subs.srt"
    folder.mkdir()
    write_srt(folder, "movie.srt", f"1\n{TIMING_1}\nHi\n")

    target = main.translate_file("movie.srt", "ES", config, translator)

    assert target == folder / "movie_ES.srt"
    assert target.exists()


def test_missing_source_file_raises_clear_error(config, translator):
    with pytest.raises(SubplexError, match=r"'does-not-exist\.srt' not found"):
        main.translate_file("does-not-exist.srt", "ES", config, translator)


def test_non_utf8_source_raises_clear_error(subs, config, translator):
    (subs / "latin1.srt").write_bytes(f"1\n{TIMING_1}\nCañón\n".encode("latin-1"))

    with pytest.raises(SubplexError, match="is not valid UTF-8"):
        main.translate_file("latin1.srt", "ES", config, translator)
    assert not (subs / "latin1_ES.srt").exists()


# --- parse_srt / render_srt -------------------------------------------------


def test_parse_srt_strips_music_symbols_and_whitespace():
    blocks = list(main.parse_srt(["1\n", f"{TIMING_1}\n", "  ♪♪ Sing ♪  \n"]))

    assert blocks == [SubtitleBlock("1", TIMING_1, ["Sing"])]


def test_render_srt_separates_blocks_with_one_blank_line():
    blocks = [
        SubtitleBlock("1", TIMING_1, ["A", "B"]),
        SubtitleBlock("2", TIMING_2, []),
    ]

    assert main.render_srt(blocks) == f"1\n{TIMING_1}\nA\nB\n\n2\n{TIMING_2}\n\n"


# --- load_config ------------------------------------------------------------


def test_load_config_returns_validated_config(tmp_path):
    write_config(tmp_path, [{"path": "/subs", "auth-key": "k"}])

    assert main.load_config(tmp_path / "config.json") == Config(Path("/subs"), "k")


def test_load_config_missing_file(tmp_path):
    with pytest.raises(SubplexError, match=r"Cannot read .*No such file"):
        main.load_config(tmp_path / "config.json")


def test_load_config_empty_file(tmp_path):
    (tmp_path / "config.json").write_text("", encoding="utf-8")

    with pytest.raises(SubplexError, match="is not valid JSON"):
        main.load_config(tmp_path / "config.json")


@pytest.mark.parametrize(
    "content",
    [
        [],
        {"path": "/subs", "auth-key": "k"},
        [{"path": "/subs"}],
        [{"path": "/subs", "auth-key": ""}],
        [{"path": 1, "auth-key": "k"}],
    ],
)
def test_load_config_rejects_wrong_shape(tmp_path, content):
    write_config(tmp_path, content)

    with pytest.raises(SubplexError):
        main.load_config(tmp_path / "config.json")


# --- find_file --------------------------------------------------------------


def test_find_file_returns_none_when_absent(tmp_path):
    assert main.find_file("missing.srt", tmp_path) is None


def test_find_file_returns_path(tmp_path):
    (tmp_path / "a.srt").touch()

    assert main.find_file("a.srt", tmp_path) == tmp_path / "a.srt"


# --- main -------------------------------------------------------------------


@pytest.fixture
def app_dir(tmp_path, subs, monkeypatch):
    """Directorio de trabajo con config.json apuntando a `subs`."""
    write_config(tmp_path, [{"path": str(subs), "auth-key": "fake-key"}])
    monkeypatch.chdir(tmp_path)
    FakeTranslator.instances = []
    monkeypatch.setattr(main.deepl, "Translator", FakeTranslator)
    return tmp_path


def test_main_success_returns_zero(app_dir, subs, capsys):
    write_srt(subs, main.SUBTITLE_NAME, f"1\n{TIMING_1}\nHi\n")

    assert main.main() == 0

    target = subs / "All Of Us Strangers_ES.srt"
    assert target.exists()
    assert [t.auth_key for t in FakeTranslator.instances] == ["fake-key"]
    out = capsys.readouterr()
    assert out.out == f"Translation completed successfully: {target}\n"
    assert out.err == ""


def test_main_reports_error_on_stderr_and_returns_one(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert main.main() == 1

    out = capsys.readouterr()
    assert out.out == ""
    assert out.err == (
        "Error occurred: Cannot read config.json: No such file or directory\n"
    )


def test_main_reports_deepl_errors(app_dir, subs, monkeypatch, capsys):
    write_srt(subs, main.SUBTITLE_NAME, f"1\n{TIMING_1}\nHi\n")

    def reject(self, text, target_lang):
        raise deepl.AuthorizationException("Authorization failure")

    monkeypatch.setattr(FakeTranslator, "translate_text", reject)

    assert main.main() == 1
    assert capsys.readouterr().err == "Error occurred: Authorization failure\n"
