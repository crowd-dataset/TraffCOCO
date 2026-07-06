"""
load_prompt.py

Prompt loading utilities for the KGFLM Traffic Annotation Pipeline.

This module provides helper functions for loading prompt templates used
throughout the annotation pipeline. Prompt templates are stored as plain
text files and optionally support Python format-string substitution,
allowing prompts to be dynamically customized during execution.

Rather than embedding prompts directly inside Python source code, every
pipeline stage loads its prompt from disk. This keeps prompts versioned,
easy to edit, and independent from the implementation.

Current prompt templates include:

    • Scene Understanding
    • SmolVLM Scene Understanding
    • Ontology Reasoning (future)
    • Prompt Builder (future)

Responsibilities

• Locate prompt templates.
• Load prompt text from disk.
• Perform variable substitution.
• Validate template existence.
• List available prompt templates.
"""



# ============================================================================
# Imports
# ============================================================================
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

from custom_logger import CustomLogger

logger = CustomLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Default prompt template directory.
_DEFAULT_PROMPTS_DIR = Path(__file__).parent


# ============================================================================
# Prompt Loading
# ============================================================================


def load_prompt(
    template_name: str,
    prompts_dir: Path | str | None = None,
    **variables: Any,
) -> str:
    """
    Load a prompt template from disk.

    Prompt templates are stored as plain text files and may optionally
    contain Python format-string placeholders. Any supplied keyword
    arguments are substituted into the template before it is returned.

    Parameters
    ----------
    template_name
        Prompt template filename.

    prompts_dir
        Directory containing prompt templates.

        If omitted, the built-in prompt directory is used.

    **variables
        Variables used for Python format-string substitution.

    Returns
    -------
    str
        Fully formatted prompt.

    Raises
    ------
    FileNotFoundError
        If the requested prompt template does not exist.

    KeyError
        If a required template variable is missing.
    """

    logger.info(
        "Loading prompt template '{}'.",
        template_name,
    )

    directory = (
        Path(prompts_dir)
        if prompts_dir is not None
        else _DEFAULT_PROMPTS_DIR
    )

    logger.debug(
        "Searching prompt directory '{}'.",
        directory,
    )

    template_path = Path(
        os.path.join(str(directory), template_name)
    )

    if not template_path.is_file():

        logger.error(
            "Prompt template '{}' not found.",
            template_path,
        )

        raise FileNotFoundError(
            f"Prompt template not found: {template_path}"
        )

    logger.debug(
        "Reading prompt template from '{}'.",
        template_path,
    )

    with open(
        template_path,
        "r",
        encoding="utf-8",
    ) as file:

        template = file.read()

    # ---------------------------------------------------------------------
    # Perform template variable substitution
    # ---------------------------------------------------------------------

    if variables:

        logger.debug(
            "Formatting prompt with {} variable(s).",
            len(variables),
        )

        try:

            prompt = template.format(
                **variables,
            )

        except KeyError as exc:

            logger.error(
                "Missing prompt template variable '{}'.",
                exc,
            )

            raise KeyError(
                f"Missing template variable: {exc}"
            ) from exc

    else:

        logger.debug(
            "Prompt contains no template variables."
        )

        prompt = template

    logger.info(
        "Successfully loaded prompt '{}'.",
        template_name,
    )

    return prompt


# ============================================================================
# Prompt Discovery
# ============================================================================


def list_available_prompts(
    prompts_dir: Path | str | None = None,
) -> list[str]:
    """
    List every available prompt template.

    Parameters
    ----------
    prompts_dir
        Directory containing prompt templates.

        If omitted, the built-in prompt directory is searched.

    Returns
    -------
    list[str]
        Sorted list of available prompt template filenames.
    """

    logger.debug(
        "Listing available prompt templates."
    )

    directory = (
        Path(prompts_dir)
        if prompts_dir is not None
        else _DEFAULT_PROMPTS_DIR
    )

    if not directory.exists():

        logger.warning(
            "Prompt directory '{}' does not exist.",
            directory,
        )

        return []

    prompts = sorted(

        prompt.name

        for prompt in directory.iterdir()

        if prompt.is_file()
        and prompt.suffix == ".txt"

    )

    logger.info(
        "Discovered {} prompt template(s).",
        len(prompts),
    )

    return prompts
