################################
##          SUBPLEX           ##
##                            ##
##  Version: 1.0.             ##
##  Create Name: WillhemM.    ##
##  Date Create: 06-09-22.    ##
##                            ##
################################

"""Traduce archivos de subtítulos SRT con la API de DeepL."""

import json
import os
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path
from typing import cast

import deepl  # cliente oficial de la API de DeepL.

CONFIG_FILE = Path("config.json")
SRT_EXTENSION = ".srt"
MUSIC_SYMBOL = "♪"
SUBTITLE_NAME = "Sense And Sensibility.srt"
TARGET_LANGUAGE = "ES"


class SubplexError(Exception):
    """Error esperado de la aplicación, con un mensaje apto para el usuario."""


@dataclass(frozen=True, slots=True)
class Config:
    """Configuración de la aplicación leída de `config.json`."""

    subtitles_dir: Path
    auth_key: str


@dataclass(frozen=True, slots=True)
class SubtitleBlock:
    """Un bloque SRT: número de secuencia, tiempos y líneas de texto."""

    sequence: str
    timing: str
    lines: list[str]


def load_config(path: Path = CONFIG_FILE) -> Config:
    """Lee y valida la configuración.

    El archivo es una lista JSON cuyo primer objeto tiene `path` (carpeta de
    subtítulos) y `auth-key` (clave de la API de DeepL).

    Raises:
        SubplexError: Si el archivo no se puede leer o no tiene ese formato.
    """
    try:
        with path.open(encoding="utf-8") as file:
            raw = json.load(file)
    except OSError as err:
        raise SubplexError(f"Cannot read {path}: {err.strerror}") from err
    except json.JSONDecodeError as err:
        raise SubplexError(f"{path} is not valid JSON: {err}") from err

    try:
        settings = raw[0]
        subtitles_dir, auth_key = settings["path"], settings["auth-key"]
    except (IndexError, KeyError, TypeError) as err:
        raise SubplexError(
            f'{path} must be a list whose first object has "path" and "auth-key"'
        ) from err
    if not (isinstance(subtitles_dir, str) and isinstance(auth_key, str) and auth_key):
        raise SubplexError(f'{path}: "path" and "auth-key" must be non-empty strings')
    return Config(Path(subtitles_dir), auth_key)


def find_file(name: str, directory: Path) -> Path | None:
    """Busca `name` recursivamente bajo `directory`.

    Returns:
        La ruta del primer archivo encontrado, o None si no existe.
    """
    for root, _dirs, files in os.walk(directory):
        if name in files:
            return Path(root, name)
    return None


def clean_line(line: str) -> str:
    """Quita los símbolos musicales y los espacios sobrantes."""
    return line.replace(MUSIC_SYMBOL, "").strip()


def parse_srt(lines: Iterable[str]) -> Iterator[SubtitleBlock]:
    """Lee bloques SRT de forma perezosa, ya sin símbolos musicales.

    Se toleran líneas en blanco de más entre bloques, y se descartan las
    líneas de texto que quedan vacías tras la limpieza (p. ej. solo "♪").
    """
    stripped = (line.strip() for line in lines)
    for sequence in stripped:
        if not sequence:
            continue
        timing = next(stripped, "")
        text = (clean_line(line) for line in takewhile(bool, stripped))
        yield SubtitleBlock(sequence, timing, [line for line in text if line])


def translate_blocks(
    blocks: Iterable[SubtitleBlock], translator: deepl.DeepLClient, language: str
) -> Iterator[SubtitleBlock]:
    """Traduce cada línea de texto por separado, conservando secuencia y tiempos."""
    for block in blocks:
        # Con un `str` de entrada, deepl devuelve un único TextResult.
        lines = [
            cast(
                deepl.TextResult, translator.translate_text(line, target_lang=language)
            ).text
            for line in block.lines
        ]
        yield SubtitleBlock(block.sequence, block.timing, lines)


def render_srt(blocks: Iterable[SubtitleBlock]) -> str:
    """Genera el SRT con una línea en blanco tras cada bloque."""
    return "".join(
        "\n".join([block.sequence, block.timing, *block.lines]) + "\n\n"
        for block in blocks
    )


def output_path(source: Path, language: str) -> Path:
    """Devuelve la ruta del subtítulo traducido: `<nombre>_<idioma>.srt`."""
    return source.with_name(f"{source.stem}_{language}{source.suffix}")


def translate_file(
    name: str, language: str, config: Config, translator: deepl.DeepLClient
) -> Path:
    """Traduce el subtítulo `name` a `language` y lo guarda junto al original.

    Returns:
        La ruta del archivo traducido.

    Raises:
        SubplexError: Si el archivo no existe, no es SRT o no está en UTF-8.
    """
    source = find_file(name, config.subtitles_dir)
    if source is None:
        raise SubplexError(f"'{name}' not found in {config.subtitles_dir}")
    if source.suffix.lower() != SRT_EXTENSION:
        raise SubplexError(f"'{source}' is not a {SRT_EXTENSION} file")

    try:
        with source.open(encoding="utf-8") as sub:
            translated = list(translate_blocks(parse_srt(sub), translator, language))
    except UnicodeDecodeError as err:
        raise SubplexError(f"'{source}' is not valid UTF-8") from err

    target = output_path(source, language)
    target.write_text(render_srt(translated), encoding="utf-8")
    return target


def main() -> int:
    """Punto de entrada: ensambla dependencias y devuelve el código de salida."""
    try:
        config = load_config()
        translator = deepl.DeepLClient(config.auth_key)
        target = translate_file(SUBTITLE_NAME, TARGET_LANGUAGE, config, translator)
    except (SubplexError, OSError, deepl.DeepLException) as err:
        print(f"Error occurred: {err}", file=sys.stderr)
        return 1
    print(f"Translation completed successfully: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
