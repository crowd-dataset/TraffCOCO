"""Extract labelled image frames from mapped traffic videos.

This module reads a CSV mapping file that describes where videos belong in the
output folder hierarchy and which time ranges should be sampled. For each CSV
row, the script resolves the listed video locally, downloads it from a configured
file server when needed, and extracts PNG frames at the configured interval.

The main processing flow is intentionally defensive:

* CSV list-like fields are parsed without using ``eval``.
* Missing location, time, and vehicle fields are normalised to stable names.
* Output folders are built from cleaned path components.
* Frame extraction writes into a temporary folder first.
* Existing complete outputs are skipped so reruns are efficient.
* Incomplete outputs are rebuilt from zero to avoid mixed stale results.
* Videos downloaded during the current run can be removed after their final use.

Configuration values are supplied by the project-level ``common`` module. The
script expects the following configured paths and values to exist:

* ``mapping``: path to the CSV mapping file.
* ``videos``: local folder used to cache or find MP4 files.
* ``frames``: root folder where extracted frame folders are created.
* ``interval_seconds``: spacing between extracted frames inside each range.
* ``base_url``: file server base URL used for downloads.
* ``delete_downloaded_videos``: whether to remove downloaded videos afterwards.

CSV column contract:

* ``continent``, ``country``, ``state``, and ``locality`` define the location
  portion of the output path.
* ``videos`` contains one or more video references. A reference may include the
  ``.mp4`` suffix or only the file stem.
* ``start_time`` and ``end_time`` contain aligned extraction ranges in seconds.
* ``time_of_day`` and ``vehicle_type`` provide labels for the extracted ranges.
* Per-video columns are expected to align by position with the ``videos`` field.
* Per-range metadata is expanded or truncated only after a warning is logged.

Output structure:

* Frames are written below ``cfg.frames_root``.
* The folder hierarchy is ``continent/country/state/locality/time/vehicle/video``.
* State values that are blank or placeholder values are stored as ``unknown``.
* Each frame is named ``<video_name>_<timestamp>.png``.
* Timestamps are generated from each requested range using ``interval_seconds``.

Safety and rerun behaviour:

* The downloader writes ``.tmp`` files before promoting completed videos.
* The extractor writes ``<output_folder>.tmp`` before promoting completed frames.
* A complete final folder is skipped on reruns.
* A partial final folder is deleted and rebuilt so outputs stay consistent.
* Downloaded videos are tracked separately from pre-existing local videos.
* Cleanup never deletes videos that were already present before this run.

Operational failure handling:

* Missing videos are logged and skipped so the remaining CSV rows can continue.
* Invalid time ranges are skipped instead of stopping the entire job.
* Metadata length mismatches are logged, then normalised to the range count.
* Failed downloads remove temporary files where possible.
* Failed frame extraction keeps temporary frames for inspection, then rebuilds
  them from zero on the next run.
* OpenCV read failures cause the current group to fail safely without promoting
  partial results.

Terminology used in this file:

* A video reference is the raw value from the CSV ``videos`` column.
* A video key is a lowercase filename stem used for usage counting.
* A range is a ``(start_seconds, end_seconds)`` pair.
* A group is a set of ranges that share the same time-of-day and vehicle labels.
* A target is one expected output frame path for one timestamp.
* Promotion means moving completed temporary output into the final location.

Maintenance notes:

* Keep path-cleaning logic centralised in ``clean_name``.
* Keep category mapping logic centralised in ``map_time_of_day`` and
  ``map_vehicle_type``.
* Keep download logic independent from extraction logic.
* Keep cleanup decisions tied to usage counting so shared videos are not deleted
  before their final CSV reference has been processed.
* Prefer adding validation near parsing helpers rather than inside OpenCV loops.
* Preserve the temporary-folder promotion pattern when changing frame extraction.
* Preserve logger messages for operational decisions because long runs need
  traceable row-level output.
* Avoid changing output naming without also updating the preflight existence
  checks, because those checks depend on deterministic filenames.
* When adding new metadata columns, normalise them before entering the extraction
  loop so the OpenCV work remains simple and focused.
* When changing download aliases, keep direct URL attempts before crawl fallback
  because direct checks are faster and produce clearer logs.
* When changing cleanup policy, remember that only files downloaded during this
  run are eligible for deletion.


Review checklist for future edits:

* Confirm that CSV parsing still handles scalar, list, and nested-list values.
* Confirm that missing metadata still maps to deterministic ``Unknown`` folders.
* Confirm that duplicate timestamps from overlapping ranges are written once.
* Confirm that a complete final folder is skipped without opening the video.
* Confirm that a partial final folder is rebuilt before new frames are promoted.
* Confirm that downloaded videos are deleted only after their last remaining use.
* Confirm that pre-existing local videos are never deleted by cleanup helpers.
* Confirm that direct download links are attempted before crawl fallback.
* Confirm that authenticated requests still use the configured session.
* Confirm that temporary video downloads are removed after failed transfers.
* Confirm that stale temporary frame folders are removed before extraction.
* Confirm that frame filenames remain compatible with existence checks.
* Confirm that logger messages include enough row and video context for debugging.
* Confirm that OpenCV capture objects are released on every exit path.
* Confirm that new metadata fields are normalised before grouping ranges.
* Confirm that folder names are passed through ``clean_name`` before joining.
* Confirm that URL joins continue to work when the base URL lacks a trailing slash.
* Confirm that crawl limits remain low enough to avoid accidental long traversals.
* Confirm that progress bars are only used for streaming downloads.
* Confirm that cleanup at the end remains a safety net, not the primary deletion
  mechanism for videos whose final use is known earlier.

Documentation policy in this version:

* Public helpers use Google style docstrings with ``Args`` and ``Returns``.
* Helpers that write, delete, download, or mutate shared state include side-effect
  notes in their docstrings.
* Inline comments are reserved for non-obvious control-flow decisions.
* Repeated comments are avoided where variable names already explain the intent.
* Operational assumptions are documented near the module top so the main loop can
  stay readable.

"""

import os
import re
import csv
import shutil
import cv2
import common
import requests

from custom_logger import CustomLogger
from logmod import logs

from bs4 import BeautifulSoup
from tqdm import tqdm
from types import SimpleNamespace
from urllib.parse import urljoin, urlparse
from collections import Counter


# ----------------------------
# Config and secrets
# ----------------------------
# Keep runtime configuration in one namespace so the rest of the module can read
# values consistently without repeatedly calling common.get_configs().
# Secrets are separated from normal configuration to make credential usage clear.
cfg = SimpleNamespace(
    csv_file=common.get_configs("mapping"),
    videos_root=common.get_configs("videos"),
    frames_root=common.get_configs("frames"),
    interval_seconds=common.get_configs("interval_seconds"),
    base_url=common.get_configs("base_url"),
    token=None,
    timeout=20,
    max_pages=500,
    debug=True,
    delete_downloaded_videos_after_processing=common.get_configs("delete_downloaded_videos"),
    aliases=["tue1", "tue2", "tue3", "tue4"],
)

secrets = SimpleNamespace(
    username=common.get_secrets("ftp_username"),
    password=common.get_secrets("ftp_password"),
)

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)


# Vehicle type mapping.
# Numeric class ids from the mapping CSV are converted into readable folder names.
# Unknown ids are still preserved by the mapping helpers as Unknown_<id>.
vehicle_map = {
    0: "Car",
    1: "Bus",
    2: "Truck",
    3: "Two wheeler",
    4: "Bicycle",
    5: "Automated car",
    6: "Electric scooter",
    7: "Monowheel unicycle",
    8: "Emergency vehicle",
    9: "Automated bus",
    10: "Automated truck",
    11: "Automated two wheeler",
    12: "Non electric scooter",
    13: "Pedestrian",
}

# Time of day mapping.
# The CSV currently uses compact numeric labels for day and night.
time_map = {
    0: "Day",
    1: "Night"
}


# ----------------------------
# General helpers
# ----------------------------
# These helpers keep CSV parsing and path normalisation predictable across the
# script. Most of the downstream workflow assumes these normalised forms.
def clean_name(value):
    """Return a filesystem-safe folder or file name component.

    The mapping CSV can contain blanks, list markers, numeric values, or values
    with characters that are invalid on common filesystems. This helper converts
    such values into stable strings before they are used in output paths.

    Args:
        value: Raw value from the CSV, config, or derived video metadata.

    Returns:
        A cleaned string. Missing or empty values are returned as ``"unknown"``.

    Side Effects:
        None. The function only normalises the provided value.
    """
    if value is None:
        return "unknown"

    value = str(value).strip()

    if value in ("", "[]", "None", "null", "nan"):
        return "unknown"

    value = re.sub(r'[<>:"/\\|?*]+', "_", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value if value else "unknown"


def is_missing_field(value):
    """Check whether a CSV field should be treated as missing.

    This is slightly stricter than a normal truthiness check because the mapping
    file may contain textual placeholders such as ``None`` or ``nan``. Treating
    those strings as missing keeps the folder hierarchy consistent.

    Args:
        value: Field value read from the CSV.

    Returns:
        ``True`` when the field is blank or represents a missing value,
        otherwise ``False``.
    """
    if value is None:
        return True

    value = str(value).strip()
    return value in ("", "[]", "None", "null", "nan", "unknown")


def ensure_list(value):
    """Wrap a scalar value in a list while preserving existing lists.

    Several CSV columns can represent either one value or many values. Downstream
    code expects list-like data, so this helper gives every caller a predictable
    container without changing list inputs.

    Args:
        value: A scalar value, a list, or ``None``.

    Returns:
        A list. ``None`` becomes an empty list, a list is returned unchanged, and
        all other values become a single-item list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_scalar(value):
    """Return the first non-list value from a nested list structure.

    Some parsed CSV fields are nested, for example ``[[0], [1]]``. When a field
    is expected to behave like a scalar, this helper repeatedly follows the first
    element until a scalar is reached.

    Args:
        value: Scalar value or arbitrarily nested list.

    Returns:
        The first scalar found, or ``None`` when an empty list is encountered.
    """
    current = value
    while isinstance(current, list):
        if not current:
            return None
        current = current[0]
    return current


def parse_bracket_value(text):
    """Parse simple bracketed CSV values into Python lists.

    The mapping file stores list-like values as strings, for example ``[a,b]`` or
    ``[[0],[1,2]]``. This helper implements a small safe reader for that limited
    format instead of evaluating the string as Python code.

    Supported token types are intentionally narrow:

    * Nested square-bracket lists.
    * Integers such as ``10`` or ``-3``.
    * Floating point numbers such as ``1.5``.
    * Quoted or unquoted text tokens.

    Args:
        text: Raw CSV cell text to parse.

    Returns:
        A parsed scalar or list. Empty input returns an empty list.

    Notes:
        The parser is permissive for malformed closing brackets because the CSV
        values are treated as operational metadata rather than user-facing data.
    """
    if text is None:
        return []

    text = str(text).strip()
    if text == "":
        return []

    i = 0
    n = len(text)

    def skip_spaces():
        """Advance the local parsing cursor past whitespace.

        The function closes over ``i`` and ``text`` from ``parse_bracket_value``.
        Keeping it nested avoids exposing parser state outside the small parsing
        routine.

        Returns:
            None. The enclosed cursor ``i`` is updated in place.
        """
        nonlocal i
        while i < n and text[i].isspace():
            i += 1

    def parse_value():
        """Parse one scalar token or nested list from the current cursor.

        The local cursor points either at an opening bracket or at the start of a
        scalar token. The helper returns the parsed value and leaves the cursor at
        the delimiter that ended the token.

        Returns:
            Parsed list, integer, float, string, empty string, or ``None`` when
            the cursor is already past the end of the text.
        """
        nonlocal i
        skip_spaces()

        if i >= n:
            return None

        if text[i] == "[":
            return parse_list()

        start = i
        while i < n and text[i] not in ",]":
            i += 1

        token = text[start:i].strip()

        if token == "":
            return ""

        if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
            token = token[1:-1]

        if re.fullmatch(r"-?\d+", token):
            return int(token)

        if re.fullmatch(r"-?\d+\.\d+", token):
            return float(token)

        return token

    def parse_list():
        """Parse a square-bracket list from the current cursor.

        The cursor is expected to be positioned on ``[``. Items are parsed one by
        one until a matching ``]`` or the end of the input is reached.

        Returns:
            A list containing parsed scalar values or nested lists.
        """
        nonlocal i
        items = []
        i += 1

        while True:
            skip_spaces()

            if i >= n:
                break

            if text[i] == "]":
                i += 1
                break

            items.append(parse_value())
            skip_spaces()

            if i < n and text[i] == ",":
                i += 1
                continue

            if i < n and text[i] == "]":
                i += 1
                break

        return items

    return parse_value()


def normalise_time_values(value):
    """Convert start or end time values into integer seconds.

    Time values may arrive as numbers, numeric strings, or single-level lists.
    This helper removes blank values and converts the rest into integer seconds
    so that frame times can be computed reliably.

    Args:
        value: Scalar or list-like time value from the parsed CSV.

    Returns:
        A list of integer second values.

    Raises:
        ValueError: If a non-empty value cannot be converted to a number.
    """
    items = ensure_list(value)
    result = []

    for item in items:
        if item is None or item == "":
            continue
        result.append(int(float(item)))

    return result


def normalise_scalar_values(value):
    """Flatten nested scalar metadata into a one-dimensional list.

    Metadata columns such as ``vehicle_type`` can be nested when the CSV row
    describes multiple videos or multiple time ranges. This helper preserves the
    scalar values while removing list nesting and blank entries.

    Args:
        value: Scalar, list, nested list, or ``None``.

    Returns:
        A flat list containing only non-empty scalar values.
    """
    result = []

    def visit(item):
        """Recursively collect scalar values for ``normalise_scalar_values``.

        Args:
            item: Current scalar or nested list node being inspected.

        Returns:
            None. Valid scalar values are appended to the outer ``result`` list.
        """
        if isinstance(item, list):
            for nested in item:
                visit(nested)
            return

        if item is None or item == "":
            return

        result.append(item)

    visit(value)
    return result


def align_values_to_count(values, count, field_name, video_ref):
    """Align metadata values to the number of time ranges.

    Each valid ``start_time`` and ``end_time`` pair creates one extraction range.
    Related metadata, such as time of day and vehicle type, must have one value
    per range. This helper expands, truncates, or fills metadata so every range
    can be grouped deterministically.

    Args:
        values: Raw scalar or list-like metadata values.
        count: Number of valid time ranges for the video.
        field_name: Human-readable field name used in warnings.
        video_ref: Video identifier used in warnings.

    Returns:
        A list with exactly ``count`` values. Missing values are represented by
        ``None`` so downstream mapping functions can produce ``Unknown``.
    """
    if count <= 0:
        return []

    items = normalise_scalar_values(values)

    if not items:
        logger.warning(f"Video '{video_ref}': field '{field_name}' is missing. "
                       f"Using Unknown for {count} range(s).")
        return [None] * count

    if len(items) == 1:
        return items * count

    if len(items) == count:
        return items

    if len(items) > count:
        logger.warning(f"Video '{video_ref}': field '{field_name}' has {len(items)} value(s) "
                       f"for {count} range(s). Truncating extras.")
        return items[:count]

    logger.warning(f"Video '{video_ref}': field '{field_name}' has {len(items)} value(s) "
                   f"for {count} range(s). Extending with the last value.")
    return items + [items[-1]] * (count - len(items))


def build_segments_for_video(start_entry, end_entry, time_entry, vehicle_entry, video_ref):
    """Build grouped extraction ranges for one video reference.

    The CSV can describe multiple time ranges for the same video. This helper
    pairs starts and ends, validates each range, maps categorical metadata into
    folder names, and groups ranges that share the same time-of-day and vehicle
    labels.

    Args:
        start_entry: Parsed start-time values for one video.
        end_entry: Parsed end-time values for one video.
        time_entry: Parsed time-of-day metadata for one video.
        vehicle_entry: Parsed vehicle-type metadata for one video.
        video_ref: CSV video reference used in logs.

    Returns:
        A dictionary where each key is ``(time_folder, vehicle_folder)`` and each
        value is a list of ``(start_seconds, end_seconds)`` ranges.

    Notes:
        Invalid ranges where the end is before the start are skipped rather than
        failing the whole row.
    """
    starts = normalise_time_values(start_entry)
    ends = normalise_time_values(end_entry)

    count = min(len(starts), len(ends))
    if count == 0:
        if len(starts) != len(ends):
            logger.warning(f"Video '{video_ref}': mismatch between starts and ends: {starts} vs {ends}")
        return {}

    if len(starts) != len(ends):
        logger.warning(f"Video '{video_ref}': mismatch between starts and ends: {starts} vs {ends}. "
                       f"Using the first {count} aligned pair(s).")

    time_values = align_values_to_count(time_entry, count, "time_of_day", video_ref)
    vehicle_values = align_values_to_count(vehicle_entry, count, "vehicle_type", video_ref)

    grouped_ranges = {}

    for i in range(count):
        s = starts[i]
        e = ends[i]

        if e < s:
            logger.warning(f"Skipping invalid range for '{video_ref}': start={s}, end={e}")
            continue

        time_folder = map_time_of_day(time_values[i])
        vehicle_folder = map_vehicle_type(vehicle_values[i])

        key = (time_folder, vehicle_folder)
        if key not in grouped_ranges:
            grouped_ranges[key] = []

        grouped_ranges[key].append((s, e))

    return grouped_ranges


def map_time_of_day(value):
    """Map a raw time-of-day code to a safe folder name.

    Known numeric values are translated through ``time_map``. Unknown numeric
    values are preserved with an ``Unknown_`` prefix, while non-numeric values are
    cleaned and used directly.

    Args:
        value: Raw scalar or nested time-of-day value.

    Returns:
        A filesystem-safe folder name such as ``Day``, ``Night``, or
        ``Unknown``.
    """
    if value is None or value == "":
        return "Unknown"

    scalar = first_scalar(value)
    if scalar is None or scalar == "":
        return "Unknown"

    try:
        key = int(float(scalar))
        return clean_name(time_map.get(key, f"Unknown_{key}"))
    except Exception:
        return clean_name(scalar)


def map_vehicle_type(value):
    """Map raw vehicle type values to a safe folder name.

    A range can contain one vehicle type or several vehicle types. Single values
    are mapped through ``vehicle_map``. Multiple values are mapped individually
    and joined with ``__`` so the combined class remains readable in the output
    hierarchy.

    Args:
        value: Raw scalar, list, or nested vehicle type value.

    Returns:
        A filesystem-safe vehicle folder name.
    """
    if value is None or value == "":
        return "Unknown"

    if isinstance(value, list):
        flattened = []
        for item in value:
            scalar = first_scalar(item)
            if scalar is None or scalar == "":
                continue
            flattened.append(scalar)

        if len(flattened) == 1:
            value = flattened[0]
        else:
            names = []
            for item in flattened:
                try:
                    key = int(float(item))
                    names.append(vehicle_map.get(key, f"Unknown_{key}"))
                except Exception:
                    names.append(str(item))
            return clean_name("__".join(names)) if names else "Unknown"

    try:
        key = int(float(value))
        return clean_name(vehicle_map.get(key, f"Unknown_{key}"))
    except Exception:
        return clean_name(value)


def build_output_folder(continent, country, state_raw, locality, time_of_day, vehicle_type, video_name):
    """Create the final folder path for one grouped extraction target.

    The folder structure encodes location, time-of-day, vehicle class, and video
    name. Every path component is cleaned before joining to avoid accidental path
    traversal or invalid filenames.

    Args:
        continent: Continent value from the CSV row.
        country: Country value from the CSV row.
        state_raw: State value from the CSV row. Missing states become
            ``unknown``.
        locality: Locality value from the CSV row.
        time_of_day: Mapped time-of-day label.
        vehicle_type: Mapped vehicle-type label.
        video_name: Video filename stem.

    Returns:
        Absolute or relative folder path under ``cfg.frames_root``.
    """
    continent = clean_name(continent)
    country = clean_name(country)
    state = "unknown" if is_missing_field(state_raw) else clean_name(state_raw)
    locality = clean_name(locality)
    time_of_day = clean_name(time_of_day)
    vehicle_type = clean_name(vehicle_type)
    video_name = clean_name(video_name)

    return os.path.join(
        cfg.frames_root,
        continent,
        country,
        state,
        locality,
        time_of_day,
        vehicle_type,
        video_name
    )


# ----------------------------
# Video index helpers
# ----------------------------
# Video references in the CSV may include either the full file name or only the
# stem. The index stores both forms so lookups are fast and forgiving.
def build_video_index(root_folder):
    """Index locally available MP4 files by filename and stem.

    The index allows the main workflow to resolve both ``example`` and
    ``example.mp4`` references from the CSV. The first discovered path for each
    key is kept to avoid surprising changes when duplicate filenames exist.

    Args:
        root_folder: Root folder to recursively scan for MP4 files.

    Returns:
        A dictionary mapping lowercase filename keys and lowercase stem keys to
        full local video paths.
    """
    index = {}

    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".mp4"):
                full_path = os.path.join(root, file)
                full_key = file.lower()
                stem_key = os.path.splitext(file)[0].lower()

                if full_key not in index:
                    index[full_key] = full_path
                if stem_key not in index:
                    index[stem_key] = full_path

    return index


def add_video_to_index(video_path, video_index):
    """Add a newly downloaded video to the local lookup index.

    Args:
        video_path: Local path to the MP4 file.
        video_index: Mutable index built by ``build_video_index``.

    Returns:
        None. The index is updated in place with filename and stem keys.
    """
    file_name = os.path.basename(video_path)
    stem = os.path.splitext(file_name)[0]

    video_index[file_name.lower()] = video_path
    video_index[stem.lower()] = video_path


def find_video_path(video_ref, video_index):
    """Resolve a CSV video reference to a local video path.

    Args:
        video_ref: Filename or filename stem from the CSV.
        video_index: Lookup table containing known local videos.

    Returns:
        The local video path when found, otherwise ``None``.
    """
    raw = str(video_ref).strip()
    if not raw:
        return None

    raw_lower = raw.lower()

    if raw_lower in video_index:
        return video_index[raw_lower]

    raw_stem = os.path.splitext(raw_lower)[0]
    if raw_stem in video_index:
        return video_index[raw_stem]

    return None


# ----------------------------
# Download helpers
# ----------------------------
# Download helpers always prefer existing local files and write new files through
# temporary paths. This prevents interrupted downloads from corrupting future runs.
def get_video_metadata(video_path):
    """Read basic metadata from a video file using OpenCV.

    The metadata is used for logging downloaded files. Frame extraction performs
    its own FPS lookup later because it needs an open capture object.

    Args:
        video_path: Local path to the video file.

    Returns:
        A tuple ``(resolution, fps)`` where resolution is ``WIDTHxHEIGHT`` when
        available, otherwise ``unknown``.
    """
    resolution = "unknown"
    fps = 0.0

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if width > 0 and height > 0:
            resolution = f"{width}x{height}"
    cap.release()

    return resolution, fps


def save_response_to_tmp_then_final(response, final_path):
    """Stream a download response to a temporary file, then promote it.

    Writing through a ``.tmp`` path prevents partially downloaded videos from
    being mistaken for complete files on the next run. Promotion uses
    ``os.replace`` so the final path appears atomically once the stream finishes.

    Args:
        response: Streaming ``requests`` response object.
        final_path: Destination MP4 path.

    Returns:
        The final path when the download succeeds or already exists, otherwise
        ``None``.

    Side Effects:
        Creates parent folders, writes a temporary file, replaces the final file,
        and removes failed temporary files when possible.
    """
    temp_path = final_path + ".tmp"

    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    if os.path.exists(final_path):
        logger.info(f"Video already present: {final_path}")
        return final_path

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass

    total = int(response.headers.get("content-length", 0)) or None
    written = 0

    try:
        with open(temp_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {os.path.basename(final_path)}"
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        bar.update(len(chunk))

        os.replace(temp_path, final_path)
        logger.info(f"Download complete: {final_path} ({written} bytes)")
        return final_path

    except Exception as e:
        logger.error(f"Download failed for {final_path}: {e}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        return None


def download_video_from_fileserver(filename):
    """Download one MP4 from the configured file server.

    The downloader first tries direct ``/files`` URLs for each configured alias.
    If direct links fail, it crawls ``/browse`` pages to locate the requested
    filename. Successful downloads are saved under ``cfg.videos_root``.

    Args:
        filename: CSV video reference. The ``.mp4`` suffix is added when absent.

    Returns:
        A tuple ``(local_path, filename, resolution, fps)`` on success. Returns
        ``None`` when the file cannot be found or downloaded.

    Side Effects:
        Performs authenticated HTTP requests when credentials are configured,
        writes files to disk, and logs each attempted path.
    """
    if not cfg.base_url:
        logger.error("Base URL is missing.")
        return None

    base = cfg.base_url if cfg.base_url.endswith("/") else cfg.base_url + "/"

    username = secrets.username if secrets.username != "" else None
    password = secrets.password if secrets.password != "" else None

    filename_with_ext = filename if str(filename).lower().endswith(".mp4") else f"{filename}.mp4"
    filename_lower = filename_with_ext.lower()
    req_params = {"token": cfg.token} if cfg.token else None
    final_path = os.path.join(cfg.videos_root, filename_with_ext)

    if os.path.exists(final_path):
        resolution, fps = get_video_metadata(final_path)
        logger.info(f"Already downloaded: {final_path}")
        return final_path, filename, resolution, fps

    logger.info(f"Starting download for '{filename_with_ext}'")

    with requests.Session() as session:
        if username and password:
            session.auth = (username, password)

        session.headers.update({"User-Agent": "multi-fileserver-downloader/1.0"})

        def fetch(url, stream=False):
            """Fetch a URL using the configured session and timeout.

            Args:
                url: Absolute URL to request.
                stream: Whether the response body should be streamed.

            Returns:
                A ``requests.Response`` object on success, otherwise ``None``.
            """
            try:
                r = session.get(url, timeout=cfg.timeout, params=req_params, stream=stream)
                logger.info(f"GET {url} -> {r.status_code}")
                if r.status_code == 401:
                    logger.error(f"Authentication failed for {url}")
                r.raise_for_status()
                return r
            except requests.RequestException as e:
                logger.warning(f"Request failed [{url}]: {e}")
                return None

        # Direct /files paths
        for alias in cfg.aliases:
            direct_url = urljoin(base, f"v/{alias}/files/{filename_with_ext}")
            logger.info(f"Trying direct URL: {direct_url}")

            r = fetch(direct_url, stream=True)
            if r is None:
                continue

            local_path = save_response_to_tmp_then_final(r, final_path)
            if local_path is None:
                return None

            resolution, fps = get_video_metadata(local_path)
            logger.info(f"Saved '{filename_with_ext}' (res={resolution}, fps={fps})")
            return local_path, filename, resolution, fps

        # Crawl /browse fallback
        visited = set()

        def is_dir_link(href):
            """Return whether a crawl link points to another browse page.

            Args:
                href: Raw ``href`` value from an anchor tag.

            Returns:
                ``True`` when the link should be pushed onto the crawl stack.
            """
            return href.startswith("/v/") and "/browse" in href

        def is_file_link(href):
            """Return whether a crawl link points to a downloadable file.

            Args:
                href: Raw ``href`` value from an anchor tag.

            Returns:
                ``True`` when the link contains the file-serving path segment.
            """
            return "/files/" in href

        def crawl(start_url):
            """Search browse pages for the requested MP4 file URL.

            The crawl is iterative rather than recursive so deep directory trees do
            not risk hitting Python recursion limits. A page limit protects the run
            from cycles or unexpectedly large listings.

            Args:
                start_url: Alias browse URL where the search should begin.

            Returns:
                The matching file URL when found, otherwise ``None``.
            """
            stack = [start_url]
            pages_seen = 0

            while stack:
                url = stack.pop()

                if url in visited:
                    continue

                visited.add(url)
                pages_seen += 1

                if pages_seen > cfg.max_pages:
                    logger.warning(f"Crawl stopped after {cfg.max_pages} pages.")
                    return None

                resp = fetch(url, stream=False)
                if resp is None:
                    continue

                try:
                    soup = BeautifulSoup(resp.text, "html.parser")
                except Exception as e:
                    logger.warning(f"HTML parse failed at {url}: {e}")
                    continue

                for a in soup.find_all("a"):
                    href = (a.get("href") or "").strip()  # type: ignore
                    if not href:
                        continue

                    full = urljoin(url, href)

                    if is_file_link(href):
                        anchor_text = (a.get_text() or "").strip().lower()
                        tail = os.path.basename(urlparse(full).path).lower()

                        if anchor_text == filename_lower or tail == filename_lower:
                            logger.info(f"File located via crawl: {full}")
                            return full

                    if is_dir_link(href):
                        stack.append(full)

            return None

        for alias in cfg.aliases:
            start_url = urljoin(base, f"v/{alias}/browse")
            logger.info(f"Crawling alias: {alias} -> {start_url}")

            found_url = crawl(start_url)
            if not found_url:
                continue

            r = fetch(found_url, stream=True)
            if r is None:
                continue

            local_path = save_response_to_tmp_then_final(r, final_path)
            if local_path is None:
                return None

            resolution, fps = get_video_metadata(local_path)
            logger.info(f"Saved '{filename_with_ext}' via crawl (res={resolution}, fps={fps})")
            return local_path, filename, resolution, fps

    logger.warning(f"File '{filename_with_ext}' was not found in any alias.")
    return None


def ensure_video_available(video_ref, video_index, downloaded_video_paths=None):
    """Return a local video path, downloading the file when required.

    The main workflow calls this only after it knows at least one frame group is
    missing. That ordering prevents unnecessary downloads for videos whose frames
    already exist.

    Args:
        video_ref: CSV video reference.
        video_index: Mutable local video lookup index.
        downloaded_video_paths: Optional set used to track videos downloaded in
            the current run for later cleanup.

    Returns:
        Local path to the video when available, otherwise ``None``.
    """
    existing = find_video_path(video_ref, video_index)
    if existing:
        logger.info(f"Using existing local video for '{video_ref}': {existing}")
        return existing

    logger.info(f"Local video missing for '{video_ref}'. Download will be attempted.")
    result = download_video_from_fileserver(video_ref)
    if not result:
        logger.warning(f"Download failed for '{video_ref}'")
        return None

    local_path = result[0]
    abs_local_path = os.path.abspath(local_path)
    add_video_to_index(local_path, video_index)
    logger.info(f"Downloaded video available at: {local_path}")
    logger.info(f"Absolute downloaded video path: {abs_local_path}")

    if downloaded_video_paths is not None:
        downloaded_video_paths.add(abs_local_path)
        logger.info(f"Tracked downloaded video for cleanup: {abs_local_path}")
        logger.info(f"Tracked downloaded videos count is now {len(downloaded_video_paths)}")

    return local_path


# ----------------------------
# Frame extraction
# ----------------------------
# Extraction is resumable at the group level. A complete group is skipped, while
# an incomplete group is rebuilt so the final folder only contains trusted output.

def build_requested_frame_times(ranges):
    """Compute unique frame timestamps requested by a set of ranges.

    Overlapping ranges can request the same timestamp more than once. This helper
    keeps the first occurrence and removes duplicates so frame files are not
    written repeatedly.

    Args:
        ranges: Iterable of ``(start_seconds, end_seconds)`` pairs.

    Returns:
        Ordered list of unique timestamps in seconds.
    """
    requested_times = []
    seen = set()

    for start_sec, end_sec in ranges:
        current_time = start_sec
        while current_time <= end_sec:
            if current_time not in seen:
                seen.add(current_time)
                requested_times.append(current_time)
            current_time += cfg.interval_seconds

    return requested_times


def expected_frame_targets(output_folder, video_name, ranges):
    """Build expected frame filenames and output paths for ranges.

    Args:
        output_folder: Final folder for the grouped frames.
        video_name: Video filename stem used in each PNG name.
        ranges: Iterable of ``(start_seconds, end_seconds)`` pairs.

    Returns:
        A list of tuples ``(timestamp, output_filename, output_path)``.
    """
    video_name = clean_name(video_name)
    targets = []

    for current_time in build_requested_frame_times(ranges):
        output_filename = f"{video_name}_{current_time}.png"
        output_path = os.path.join(output_folder, output_filename)
        targets.append((current_time, output_filename, output_path))

    return targets


def count_existing_requested_frames(output_folder, video_name, ranges):
    """Count already-created frames for the requested targets.

    Args:
        output_folder: Final output folder for the group.
        video_name: Video filename stem used in expected frame names.
        ranges: Extraction ranges for the group.

    Returns:
        A tuple ``(existing_count, total_count)``.
    """
    targets = expected_frame_targets(output_folder, video_name, ranges)
    existing = [output_path for _, _, output_path in targets if os.path.exists(output_path)]
    return len(existing), len(targets)


def all_requested_frames_exist(output_folder, video_name, ranges):
    """Return whether every expected frame already exists.

    Args:
        output_folder: Final output folder for the group.
        video_name: Video filename stem used in expected frame names.
        ranges: Extraction ranges for the group.

    Returns:
        ``True`` only when at least one frame is expected and all expected files
        are already present.
    """
    existing_count, total_count = count_existing_requested_frames(output_folder, video_name, ranges)
    return total_count > 0 and existing_count == total_count


def extract_frames_for_ranges(video_path, output_folder, ranges):
    """Extract frames for a group of ranges into one output folder.

    The function is designed to be safe for reruns and interruptions. Frames are
    first written into ``<output_folder>.tmp``. Only after every expected frame is
    present in that temporary folder are files promoted to the final folder.

    Existing final outputs are handled carefully:

    * Complete outputs are skipped.
    * Incomplete outputs are deleted and rebuilt from zero.
    * Stale temporary folders from interrupted runs are removed before retrying.

    Args:
        video_path: Local MP4 file to read.
        output_folder: Final folder where PNG frames should be placed.
        ranges: List of ``(start_seconds, end_seconds)`` extraction ranges.

    Returns:
        Number of frames promoted to the final folder. Returns ``0`` if the
        extraction fails or there is nothing to extract.

    Side Effects:
        Creates folders, writes PNG files, deletes stale partial outputs, and
        removes the temporary folder after successful promotion.
    """
    parent_folder = os.path.dirname(output_folder)
    if parent_folder:
        os.makedirs(parent_folder, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_name = clean_name(video_name)

    final_targets = expected_frame_targets(output_folder, video_name, ranges)

    if not final_targets:
        logger.warning(f"No frame times requested for {video_path}")
        return 0

    existing_final = [output_path for _, _, output_path in final_targets if os.path.exists(output_path)]

    temp_output_folder = output_folder + ".tmp"

    if os.path.isdir(temp_output_folder):
        logger.info(f"Removing stale temp folder: {temp_output_folder}")
        shutil.rmtree(temp_output_folder, ignore_errors=True)
    elif os.path.exists(temp_output_folder):
        try:
            os.remove(temp_output_folder)
        except Exception as e:
            logger.warning(f"Could not remove stale temp path {temp_output_folder}: {e}")

    if len(existing_final) == len(final_targets):
        logger.info(f"All requested frames already exist for {video_name}")
        logger.info(f"Finished {video_name}: saved 0, skipped existing {len(existing_final)}")
        return 0

    if existing_final:
        logger.warning(f"Incomplete final output detected for {video_name}. "
                       f"Deleting {len(existing_final)} existing frame(s) and rebuilding from zero.")

        for existing_path in existing_final:
            try:
                os.remove(existing_path)
            except Exception as e:
                logger.warning(f"Could not delete incomplete frame {existing_path}: {e}")

    os.makedirs(temp_output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.error(f"Could not determine FPS: {video_path}")
        cap.release()
        return 0

    saved_count = 0
    extraction_failed = False

    for current_time, output_filename, _ in final_targets:
        target_frame = int(round(current_time * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Could not read frame at {current_time}s from {video_path}")
            extraction_failed = True
            break

        temp_output_path = os.path.join(temp_output_folder, output_filename)
        ok = cv2.imwrite(temp_output_path, frame)
        if ok:
            logger.info(f"Saved to temp: {temp_output_path}")
            saved_count += 1
        else:
            logger.warning(f"Failed to save to temp: {temp_output_path}")
            extraction_failed = True
            break

    cap.release()

    if extraction_failed or saved_count != len(final_targets):
        logger.warning(f"Extraction for {video_name} did not complete. "
                       f"Temp output kept at {temp_output_folder}. It will be deleted and rebuilt on the next run.")
        return 0

    os.makedirs(output_folder, exist_ok=True)

    promoted_count = 0
    for _, output_filename, output_path in final_targets:
        temp_output_path = os.path.join(temp_output_folder, output_filename)

        if not os.path.exists(temp_output_path):
            logger.error(f"Missing temp frame during promotion: {temp_output_path}")
            logger.warning(
                f"Promotion for {video_name} stopped. "
                f"Remaining temp files kept at {temp_output_folder}."
            )
            return promoted_count

        os.replace(temp_output_path, output_path)
        promoted_count += 1

    try:
        shutil.rmtree(temp_output_folder)
    except Exception as e:
        logger.warning(f"Could not remove temp folder {temp_output_folder}: {e}")

    logger.info(
        f"Finished {video_name}: extracted {saved_count} frame(s), "
        f"promoted {promoted_count} to final"
    )
    return promoted_count


def delete_downloaded_videos(downloaded_video_paths):
    """Delete all videos tracked as downloaded during this run.

    Args:
        downloaded_video_paths: Set of absolute video paths that were downloaded
            by ``ensure_video_available``.

    Returns:
        None. Cleanup results are reported through logs.

    Side Effects:
        Removes files from disk when they still exist.
    """
    if not downloaded_video_paths:
        logger.info("No downloaded videos to delete.")
        return

    deleted = 0
    failed = 0
    missing = 0

    logger.info("Starting downloaded video cleanup")
    for video_path in sorted(downloaded_video_paths):
        abs_video_path = os.path.abspath(video_path)
        exists = os.path.exists(abs_video_path)
        logger.info(f"Cleanup candidate: {abs_video_path} | exists={exists}")

        if not exists:
            missing += 1
            continue

        try:
            os.remove(abs_video_path)
            deleted += 1
            logger.info(f"Deleted downloaded video: {abs_video_path}")
        except Exception as e:
            failed += 1
            logger.warning(f"Could not delete downloaded video {abs_video_path}: {e}")

    logger.info(f"Downloaded video cleanup finished: deleted={deleted}, failed={failed}, already_missing={missing}")


def canonical_video_ref(video_ref):
    """Convert a video reference into a stable usage-count key.

    Args:
        video_ref: Raw CSV video reference.

    Returns:
        Lowercase filename stem, or an empty string for blank values.
    """
    raw = str(video_ref).strip().lower()
    if not raw:
        return ""
    return os.path.splitext(raw)[0]


def collect_video_usage_counts(csv_file):
    """Count how many times each video is referenced in the mapping CSV.

    The counts allow downloaded files to be removed immediately after their final
    use, while still keeping them available for later rows that reference the
    same video.

    Args:
        csv_file: Path to the mapping CSV.

    Returns:
        ``Counter`` mapping canonical video references to remaining use counts.
    """
    usage_counts = Counter()

    # utf-8-sig handles CSV files exported with a UTF-8 byte order mark.
    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            videos = ensure_list(parse_bracket_value(row.get("videos", "")))
            for video_ref in videos:
                key = canonical_video_ref(video_ref)
                if key:
                    usage_counts[key] += 1

    return usage_counts


def remove_video_from_index(video_path, video_index):
    """Remove all lookup keys that point to a deleted video path.

    Args:
        video_path: Local path that has been deleted or is considered missing.
        video_index: Mutable video lookup index.

    Returns:
        None. Matching index entries are removed in place.
    """
    keys_to_remove = [key for key, value in video_index.items() if os.path.abspath(value) == os.path.abspath(video_path)]  # noqa: E501
    for key in keys_to_remove:
        video_index.pop(key, None)
    logger.info(f"Removed {len(keys_to_remove)} index entrie(s) for deleted video: {video_path}")


def delete_single_downloaded_video(video_path, downloaded_video_paths, video_index):
    """Delete one downloaded video and remove it from tracking structures.

    Only files downloaded during the current run are eligible for immediate
    deletion. Pre-existing local videos are intentionally preserved.

    Args:
        video_path: Local video path to delete.
        downloaded_video_paths: Set tracking downloaded files from this run.
        video_index: Mutable local video lookup index.

    Returns:
        ``True`` when the file was deleted successfully, otherwise ``False``.
    """
    abs_video_path = os.path.abspath(video_path)

    if abs_video_path not in downloaded_video_paths:
        logger.info(f"Not deleting {abs_video_path} because it was not downloaded in this run")
        return False

    exists = os.path.exists(abs_video_path)
    logger.info(f"Immediate cleanup candidate: {abs_video_path} | exists={exists}")
    if not exists:
        downloaded_video_paths.discard(abs_video_path)
        remove_video_from_index(abs_video_path, video_index)
        logger.warning(f"Immediate cleanup skipped because file is already missing: {abs_video_path}")
        return False

    try:
        os.remove(abs_video_path)
        downloaded_video_paths.discard(abs_video_path)
        remove_video_from_index(abs_video_path, video_index)
        logger.info(f"Deleted downloaded video immediately after final use: {abs_video_path}")
        return True
    except Exception as e:
        logger.warning(f"Could not delete downloaded video immediately {abs_video_path}: {e}")
        return False


def finalize_video_usage(video_key, video_path, remaining_video_uses, downloaded_video_paths, video_index, log_prefix):
    """Update remaining-use tracking after a video has been processed.

    When cleanup is enabled, this helper deletes a downloaded video as soon as no
    future CSV row needs it. The final end-of-run cleanup remains as a safety net
    for any tracked files that could not be removed immediately.

    Args:
        video_key: Canonical video reference used in ``remaining_video_uses``.
        video_path: Local path to the resolved video, if known.
        remaining_video_uses: Mutable counter of future uses.
        downloaded_video_paths: Set of videos downloaded during this run.
        video_index: Mutable local lookup index.
        log_prefix: Row-specific prefix used to make logs easier to trace.

    Returns:
        Remaining use count after decrementing the current use.
    """
    if video_key in remaining_video_uses and remaining_video_uses[video_key] > 0:
        remaining_video_uses[video_key] -= 1

    remaining_after = remaining_video_uses.get(video_key, 0)
    logger.info(f"{log_prefix}: remaining uses after processing for '{video_key}' = {remaining_after}")

    if cfg.delete_downloaded_videos_after_processing:
        if remaining_after == 0:
            if video_path:
                deleted_now = delete_single_downloaded_video(video_path, downloaded_video_paths, video_index)
                logger.info(f"{log_prefix}: immediate delete attempted for '{video_key}' -> deleted={deleted_now}")
            else:
                logger.info(f"{log_prefix}: delete skipped for '{video_key}', no local path")
        else:
            logger.info(f"{log_prefix}: immediate delete skipped for '{video_key}' because future uses remain")

    return remaining_after


# ----------------------------
# Main
# ----------------------------
# The main loop is intentionally organised around CSV rows, then video references,
# then grouped ranges. That mirrors the source mapping file and makes logs easier
# to trace back to a specific row.
def main():
    """Run the end-to-end CSV driven frame extraction workflow.

    The main function coordinates configuration setup, CSV validation, video
    resolution, grouped frame extraction, and optional cleanup. Smaller helpers
    contain the detailed parsing, mapping, downloading, and extraction behaviour.

    Processing order for each video reference is:

    * Parse the row metadata.
    * Build valid extraction groups.
    * Check whether requested frames already exist.
    * Download or resolve the video only when work remains.
    * Extract all pending frame groups.
    * Update usage counters and cleanup downloaded videos when safe.

    Returns:
        None. Progress and warnings are written through the configured logger.

    Raises:
        ValueError: If the mapping CSV is missing a required column.
    """
    os.makedirs(cfg.videos_root, exist_ok=True)
    os.makedirs(cfg.frames_root, exist_ok=True)

    logger.info(f"csv_file={cfg.csv_file}")
    logger.info(f"videos_root={cfg.videos_root}")
    logger.info(f"frames_root={cfg.frames_root}")
    logger.info(f"delete_downloaded_videos_after_processing={cfg.delete_downloaded_videos_after_processing}")

    video_index = build_video_index(cfg.videos_root)
    logger.info(f"Initial local video index entries={len(video_index)}")
    downloaded_video_paths = set()
    remaining_video_uses = collect_video_usage_counts(cfg.csv_file)
    logger.info(f"Tracked unique video refs in mapping={len(remaining_video_uses)}")

    # Open the mapping once for the main pass after usage counts have been built.
    with open(cfg.csv_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = [
            "continent",
            "country",
            "state",
            "locality",
            "videos",
            "time_of_day",
            "start_time",
            "end_time",
            "vehicle_type",
        ]

        # Fail early when the mapping schema is not what the extractor expects.
        for col in required_columns:
            if col not in reader.fieldnames:  # type: ignore
                raise ValueError(f"Missing required column in CSV: {col}")

        for row_num, row in enumerate(reader, start=2):
            continent = row.get("continent", "")
            country = row.get("country", "")
            state = row.get("state", "")
            locality = row.get("locality", "")

            # Parse all list-like CSV fields before indexing into per-video values.
            videos = ensure_list(parse_bracket_value(row.get("videos", "")))
            time_of_day_values = ensure_list(parse_bracket_value(row.get("time_of_day", "")))
            starts = ensure_list(parse_bracket_value(row.get("start_time", "")))
            ends = ensure_list(parse_bracket_value(row.get("end_time", "")))
            vehicle_types = ensure_list(parse_bracket_value(row.get("vehicle_type", "")))

            if not videos:
                logger.warning(f"Row {row_num}: no videos listed")
                continue

            for i, video_ref in enumerate(videos):
                video_key = canonical_video_ref(video_ref)
                logger.info(f"Row {row_num}: starting video_ref={video_ref}")
                logger.info(
                    f"Row {row_num}: uses left for '{video_key}' = "
                    f"{remaining_video_uses.get(video_key, 0)}"
                )

                start_entry = starts[i] if i < len(starts) else []
                end_entry = ends[i] if i < len(ends) else []
                time_entry = time_of_day_values[i] if i < len(time_of_day_values) else None
                vehicle_entry = vehicle_types[i] if i < len(vehicle_types) else None

                grouped_ranges = build_segments_for_video(
                    start_entry=start_entry,
                    end_entry=end_entry,
                    time_entry=time_entry,
                    vehicle_entry=vehicle_entry,
                    video_ref=video_ref,
                )

                if not grouped_ranges:
                    logger.warning(f"Row {row_num}: no valid ranges for video -> {video_ref}")

                    finalize_video_usage(
                        video_key=video_key,
                        video_path=find_video_path(video_ref, video_index),
                        remaining_video_uses=remaining_video_uses,
                        downloaded_video_paths=downloaded_video_paths,
                        video_index=video_index,
                        log_prefix=f"Row {row_num}",
                    )
                    continue

                # Use the known local filename for preflight frame checks when possible.
                existing_video_path = find_video_path(video_ref, video_index)
                if existing_video_path:
                    planned_video_name = os.path.splitext(os.path.basename(existing_video_path))[0]
                else:
                    planned_video_name = os.path.splitext(os.path.basename(str(video_ref).strip()))[0]

                # Only groups with missing frames need video resolution or downloading.
                pending_groups = []

                for (time_folder, vehicle_folder), ranges in grouped_ranges.items():
                    output_folder = build_output_folder(
                        continent=continent,
                        country=country,
                        state_raw=state,
                        locality=locality,
                        time_of_day=time_folder,
                        vehicle_type=vehicle_folder,
                        video_name=planned_video_name,
                    )

                    existing_count, total_count = count_existing_requested_frames(
                        output_folder=output_folder,
                        video_name=planned_video_name,
                        ranges=ranges,
                    )

                    if total_count > 0 and existing_count == total_count:
                        logger.info(f"Row {row_num}: all requested frames already exist for "
                                    f"video '{video_ref}' under time_of_day='{time_folder}' "
                                    f"and vehicle_type='{vehicle_folder}'. Skipping this group without downloading.")
                        continue

                    pending_groups.append((time_folder, vehicle_folder, ranges))

                    if total_count == 0:
                        logger.info(f"Row {row_num}: no requested frames resolved for "
                                    f"video '{video_ref}' under time_of_day='{time_folder}' "
                                    f"and vehicle_type='{vehicle_folder}'")
                    else:
                        logger.info(
                            f"Row {row_num}: {existing_count}/{total_count} frames exist for "
                            f"video '{video_ref}', time='{time_folder}', vehicle='{vehicle_folder}'. "
                            "Download/extraction still needed."
                        )

                if not pending_groups:
                    logger.info(f"Row {row_num}: all requested frame groups already exist for video -> {video_ref}. "
                                f"Skipping video resolution and download entirely.")

                    finalize_video_usage(
                        video_key=video_key,
                        video_path=existing_video_path,
                        remaining_video_uses=remaining_video_uses,
                        downloaded_video_paths=downloaded_video_paths,
                        video_index=video_index,
                        log_prefix=f"Row {row_num}",
                    )
                    continue

                # Resolve the actual video path only after confirming useful work remains.
                video_path = ensure_video_available(video_ref, video_index, downloaded_video_paths)

                if not video_path:
                    logger.warning(f"Row {row_num}: video not found and download failed -> {video_ref}")
                    finalize_video_usage(
                        video_key=video_key,
                        video_path=find_video_path(video_ref, video_index),
                        remaining_video_uses=remaining_video_uses,
                        downloaded_video_paths=downloaded_video_paths,
                        video_index=video_index,
                        log_prefix=f"Row {row_num}",
                    )
                    continue

                logger.info(f"Row {row_num}: resolved video path -> {video_path}")

                count = 0
                actual_video_name = os.path.splitext(os.path.basename(video_path))[0]

                for time_folder, vehicle_folder, ranges in pending_groups:
                    logger.info(f"Row {row_num}: processing {len(ranges)} range(s) for "
                                f"video '{video_ref}' under time_of_day='{time_folder}' "
                                f"and vehicle_type='{vehicle_folder}'")

                    output_folder = build_output_folder(
                        continent=continent,
                        country=country,
                        state_raw=state,
                        locality=locality,
                        time_of_day=time_folder,
                        vehicle_type=vehicle_folder,
                        video_name=actual_video_name,
                    )

                    group_count = extract_frames_for_ranges(
                        video_path=video_path,
                        output_folder=output_folder,
                        ranges=ranges,
                    )
                    count += group_count

                logger.info(f"Done: {video_path} -> {count} new frames")

                finalize_video_usage(
                    video_key=video_key,
                    video_path=video_path,
                    remaining_video_uses=remaining_video_uses,
                    downloaded_video_paths=downloaded_video_paths,
                    video_index=video_index,
                    log_prefix=f"Row {row_num}",
                )

    logger.info("Reached end of CSV processing")
    logger.info(f"delete_downloaded_videos_after_processing={cfg.delete_downloaded_videos_after_processing}")
    logger.info(f"downloaded_video_paths_count={len(downloaded_video_paths)}")
    if downloaded_video_paths:
        for tracked_path in sorted(downloaded_video_paths):
            logger.info(f"Tracked for cleanup: {tracked_path}")

    # Final cleanup is a safety net for downloaded files that were not removed earlier.
    if cfg.delete_downloaded_videos_after_processing:
        delete_downloaded_videos(downloaded_video_paths)
    else:
        logger.info("Cleanup skipped because delete_downloaded_videos_after_processing is False")

    logger.info("All done.")


if __name__ == "__main__":
    main()
