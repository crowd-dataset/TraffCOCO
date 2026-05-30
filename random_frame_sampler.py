# by Shadab Alam <md_shadab_alam@outlook.com>
"""Random frame sampler for the extracted traffic frame dataset.

This script samples already extracted frame images from a remote file server and
copies a random subset into a local output folder. It is designed to work with
the same folder structure produced by the frame extraction pipeline:

    continent / country / state / city / day_or_night / vehicle / video_id / image

The script is intentionally conservative when browsing the server. Instead of
walking every image in the full dataset, it samples through folder levels and
caps the number of video folders and images inspected per folder. This keeps the
script responsive when the data store is very large.

The high level workflow is:

    1. Read user configuration from ``common``.
    2. Convert broad filters into concrete city level filters.
    3. Browse the remote file server folder hierarchy.
    4. Collect unique image candidates that match the filters.
    5. Shuffle candidates and download the requested number of images.
    6. Write a CSV manifest describing every downloaded frame.

Important implementation choices:

    * Empty values and the string ``all`` are treated as wildcards.
    * Output cleanup only removes image, temporary image, and CSV files.
    * Downloads are written to ``.tmp`` files first, then atomically promoted.
    * Candidate URLs are deduplicated before selection and download.
    * Location balancing prevents broad searches from over sampling one country.

Configuration expected from ``common.get_configs``:

    * ``base_url``: Root URL of the FTP style file server.
    * ``num_images``: Number of random images to download.
    * ``LOCATION_TREE``: Nested location filter dictionary or wildcard.
    * ``DAY_NIGHTS``: Day/Night filters or wildcard.
    * ``VEHICLES``: Vehicle filters or wildcard.
    * ``VIDEO_IDS``: Video id filters or wildcard.
    * ``local_output_root``: Destination folder for downloaded frames.
    * ``logger_level``: Logging verbosity.

Secrets expected from ``common.get_secrets``:

    * ``ftp_username``: Optional username for basic authentication.
    * ``ftp_password``: Optional password for basic authentication.


Maintainer guide:

    Reading the configuration:
        The sampler expects ``common`` to return values that are already typed
        sensibly for the deployment. The helper functions still defend against
        strings, lists, tuples, sets, wildcards, and empty markers because config
        files often change format across environments.

    Wildcard behaviour:
        An empty filter means all folders at that hierarchy level are allowed.
        This convention is used consistently for locations, day/night values,
        vehicle types, and video identifiers. Keeping this rule consistent makes
        the nested traversal easier to reason about.

    Location traversal:
        The file server is browsed as a tree. A broad location filter is first
        expanded into concrete city filters so later balancing logic can treat
        every city as a separate sampling unit.

    Country balancing:
        When many cities are eligible, the sampler groups cities by country and
        interleaves the groups. This avoids a common failure mode where the first
        alphabetically listed country contributes nearly every sampled image.

    First pass sampling:
        The first pass uses a small per video folder quota. This spreads images
        across more source videos and locations, which usually gives a more
        diverse random sample.

    Fill pass sampling:
        The fill pass runs only when the first pass cannot satisfy the requested
        count. It allows more images per video folder so sparse locations do not
        prevent the sampler from returning useful output.

    Duplicate protection:
        The sampler checks ``seen_urls`` while collecting candidates and also
        deduplicates the final discovered list by URL. The two stage protection
        is intentional because the same file can be discovered through different
        traversal passes.

    Browse cache:
        ``BROWSE_CACHE`` stores HTML responses for browse pages. This is safe in
        a single run because the script assumes the remote folder tree is stable
        while sampling is happening.

    Network failures:
        Browse failures return empty results instead of aborting immediately.
        This lets the sampler continue through other aliases, cities, vehicles,
        or videos when one folder is unavailable.

    Download failures:
        Image downloads are handled independently. If one selected image fails,
        the script logs the failure and continues with the remaining selected
        candidates.

    Temporary files:
        Every download writes to a ``.tmp`` path before promotion. This prevents
        interrupted downloads from leaving files that look like valid images.

    Output cleanup:
        Cleanup intentionally removes only image files, image temporary files,
        and CSV files. It does not delete unrelated files that a user may have
        placed inside the output folder.

    Unsafe cleanup paths:
        The cleanup helper refuses broad paths such as the filesystem root, the
        current directory, and the user home directory. This guard should stay in
        place whenever cleanup behaviour changes.

    Manifest design:
        The CSV manifest is the reproducibility record for the run. It stores
        both the local path and the remote source URL, plus folder metadata that
        downstream analysis can use without parsing URLs again.

    Folder names:
        The sampler always uses cleaned names for comparisons and output values.
        This prevents small differences in whitespace, case, or invalid path
        characters from breaking matching logic.

    Vehicle mapping:
        Numeric vehicle ids are mapped through ``VEHICLE_MAP`` so configs can use
        either the original ids or the readable folder names.

    Day/night mapping:
        Numeric day/night ids are mapped through ``TIME_MAP``. Unknown numeric
        values are kept as cleaned fallback text rather than raising errors.

    URL construction:
        Path components are quoted one segment at a time. This keeps spaces and
        special characters valid without accidentally turning slashes inside a
        name into hierarchy separators.

    HTML parsing:
        Browse pages are parsed by inspecting anchor links. Parent navigation is
        skipped, folder links are separated from file links, and only direct
        children of the current folder are accepted.

    Sampling randomness:
        Shuffling happens at several levels: location order, folder options,
        video options, image files, and final candidate selection. This reduces
        bias from deterministic server listing order.

    Sampling limits:
        ``MAX_VIDEO_FOLDERS_TO_CHECK`` protects very broad searches from walking
        the whole server. Increase it only when broad searches are expected to
        inspect more folders than the current cap.

    Per folder image limits:
        ``MAX_IMAGES_PER_VIDEO_FOLDER`` controls diversity during the first pass.
        ``FILL_IMAGES_PER_VIDEO_FOLDER`` controls how aggressively the sampler
        fills remaining quota after sparse locations have been tried.

    Flat output layout:
        ``local_folder_for_candidate`` currently returns one flat output folder.
        Keeping the decision in a helper makes it easy to switch to nested output
        folders later without touching download and manifest code.

    Filename collisions:
        ``unique_local_path`` preserves the preferred filename when possible and
        adds numeric suffixes only when a file already exists. This avoids silent
        overwrites.

    Authentication:
        The session stores authentication, and individual requests also pass the
        credentials. This is redundant but harmless, and it preserves behaviour
        if one request path bypasses the session default.

    Error visibility:
        User facing failures are raised for invalid config or zero matching
        images. Recoverable network and download issues are logged so the run can
        produce partial useful output when possible.

    Extension handling:
        Only common image extensions in ``IMAGE_EXTENSIONS`` are sampled. Add new
        extensions there if the extractor begins producing another image format.

    Alias scope:
        The entry point currently scans only ``tue5``. Expanding to multiple
        aliases should happen in that alias loop so deduplication and final
        random selection still operate across the combined candidate pool.

    Test strategy:
        For safe testing, point ``LOCAL_OUTPUT_ROOT`` at a temporary folder, set
        ``NUM_IMAGES`` to a small value, and use a narrow ``LOCATION_TREE`` that
        is known to contain images.

    Performance strategy:
        If sampling is slow, check how broad the location filters are, whether
        the browse cache is being reused, and whether the video folder cap is too
        high for the requested number of images.

    Debugging missing data:
        When a requested vehicle or video id is not found, the warnings include
        the location path that was being searched. Use that path to compare the
        config with the remote folder names.

    Maintaining docstrings:
        Public helpers include Google style ``Args``, ``Returns``, and ``Raises``
        sections where relevant. Keep those sections updated whenever function
        parameters or side effects change.


    Common change points:
        Add new vehicle ids only in ``VEHICLE_MAP``.
        Add new time labels only in ``TIME_MAP``.
        Add new image formats only in ``IMAGE_EXTENSIONS``.
        Change output layout inside ``local_folder_for_candidate``.
        Change server aliases inside the alias loop in the entry point.
        Change browse limits through the constants near the top of the file.
        Change balancing behaviour through ``BALANCE_ACROSS_LOCATIONS``.
        Change cleanup behaviour through ``CLEAR_PREVIOUS_RANDOM_OUTPUTS``.
        Change manifest writing through ``SAVE_RANDOM_FRAMES_CSV``.
        Keep temporary file promotion in place for every future downloader.
        Keep URL deduplication when adding aliases or extra discovery passes.
        Keep cleanup safety checks before expanding deletion rules.
        Keep broad filters expanded to city filters before balancing.
        Keep logging at important decision points for long remote scans.
        Keep config validation close to the main entry point.
        Keep recoverable remote folder misses as warnings, not hard failures.
        Keep invalid local output roots as hard failures.
        Keep the manifest schema stable unless downstream users are updated.
        Keep tests narrow and temporary when verifying destructive cleanup.
        Keep this documentation updated when control flow changes.
"""
import csv
import os
import random
import math
import re
from dataclasses import dataclass
from urllib.parse import quote, unquote, urljoin, urlparse

import common
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from custom_logger import CustomLogger
from logmod import logs


# =============================================================================
# Config and secrets
# =============================================================================

# Remote file server connection settings.
# Authentication is optional; empty credentials make public requests.
BASE_URL = common.get_configs("base_url")
FTP_USERNAME = common.get_secrets("ftp_username")
FTP_PASSWORD = common.get_secrets("ftp_password")

# User controlled sampling filters.
# Empty or "all" values are treated as wildcards by the normalisation helpers.
NUM_IMAGES = common.get_configs("num_images")
LOCATION_TREE = common.get_configs("LOCATION_TREE")
DAY_NIGHTS = common.get_configs("DAY_NIGHTS")
VEHICLES = common.get_configs("VEHICLES")
VIDEO_IDS = common.get_configs("VIDEO_IDS")

LOCAL_OUTPUT_ROOT = common.get_configs("local_output_root")
SAVE_RANDOM_FRAMES_CSV = True

# Delete old sampled frames and CSV files before downloading new random frames.
CLEAR_PREVIOUS_RANDOM_OUTPUTS = True

# Fast sampling controls.
# This avoids scanning all videos/images when the dataset is huge.
MAX_VIDEO_FOLDERS_TO_CHECK = 5000
MAX_IMAGES_PER_VIDEO_FOLDER = 1
FILL_IMAGES_PER_VIDEO_FOLDER = 5

# When multiple locations are provided, first try to take frames from each one.
# If some locations do not have enough frames, the remaining quota is filled
# from locations that still have available frames.
BALANCE_ACROSS_LOCATIONS = True

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# =============================================================================
# Constants and mappings from your extractor
# =============================================================================

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

VEHICLE_MAP = {
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

TIME_MAP = {
    0: "Day",
    1: "Night",
}

# Cache browse HTML by URL so repeated traversal of the same folder is cheap.
BROWSE_CACHE = {}


@dataclass
class LocationFilter:
    """Normalized location selector used while browsing the dataset tree.

        Attributes:
            continent: Optional continent folder name. ``None`` means any continent.
            country: Optional country folder name. ``None`` means any country.
            state: Optional state folder name. ``None`` means any state.
            city: Optional city or locality folder name. ``None`` means any city.
        """
    continent: str | None = None
    country: str | None = None
    state: str | None = None
    city: str | None = None


@dataclass
class FrameCandidate:
    """Remote image candidate plus metadata needed for the output manifest.

        Attributes:
            url: Direct file URL for the candidate frame image.
            filename: Clean local filename derived from the remote URL.
            alias: File server alias, such as ``tue5``.
            continent: Continent folder where the frame was found.
            country: Country folder where the frame was found.
            state: State folder where the frame was found.
            city: City or locality folder where the frame was found.
            day_night: Time of day folder, usually ``Day`` or ``Night``.
            vehicle: Vehicle category folder.
            video_id: Source video folder name.
        """
    url: str
    filename: str
    alias: str
    continent: str
    country: str
    state: str
    city: str
    day_night: str
    vehicle: str
    video_id: str


# =============================================================================
# General helpers
# =============================================================================

def is_empty_marker(value):
    """Return whether a value should be treated as an empty filter marker.

        Args:
            value: Raw value from configuration or intermediate normalisation.

        Returns:
            ``True`` when the value is empty, missing, or one of the placeholder
            markers used by the configuration files. Otherwise ``False``.
        """
    if value is None:
        return True

    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0

    text = str(value).strip()
    return text in ("", ".", "/", "[]", "{}", "None", "null", "nan")


def is_all_marker(value):
    """Return whether a value means no filtering should be applied.

        Args:
            value: Raw value to inspect.

        Returns:
            ``True`` for empty values and the case insensitive string ``all``.
            These values are interpreted as wildcards by the sampler.
        """
    if is_empty_marker(value):
        return True

    return str(value).strip().casefold() == "all"


def clean_name(value):
    """Convert a raw folder or file component into a safe display name.

        Args:
            value: Raw value that may contain illegal path characters or blanks.

        Returns:
            A cleaned string safe to use as a folder or file component. Missing
            values are normalised to ``unknown``.
        """
    if value is None:
        return "unknown"

    value = str(value).strip()

    if value in ("", "[]", "{}", "None", "null", "nan"):
        return "unknown"

    value = re.sub(r'[<>:"/\\|?*]+', "_", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value if value else "unknown"


def normalise_for_match(value):
    """Return a canonical key for case insensitive folder comparisons.

        Args:
            value: Raw folder, filter, or mapping value.

        Returns:
            Cleaned and case folded text suitable for dictionary and set matching.
        """
    return clean_name(value).casefold()


def ensure_list(value):
    """Return ``value`` as a list while preserving meaningful scalar values.

        Args:
            value: A scalar, string, list, tuple, set, or ``None``.

        Returns:
            A list representation of ``value``. Comma separated strings are split,
            empty string markers become an empty list, and scalars become one item.
        """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    if isinstance(value, str):
        text = value.strip()

        if text in ("", "[]"):
            return []

        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]

        return [value]

    return [value]


def unique_preserve_order(items):
    """Remove duplicates without changing the first occurrence order.

        Args:
            items: Iterable containing hashable or unhashable values.

        Returns:
            A list where repeated values are removed using their ``repr`` as a
            stable comparison key.
        """
    seen = set()
    result = []

    for item in items:
        key = repr(item)

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def normalise_day_night(value):
    """Normalise a day/night filter into the dataset folder naming scheme.

        Args:
            value: A string or numeric code representing day, night, or all.

        Returns:
            ``Day``, ``Night``, a cleaned fallback value, or ``None`` when the input
            means all day/night folders are allowed.
        """
    if is_all_marker(value):
        return None

    text = str(value).strip()
    lower = text.casefold()

    if lower in ("0", "day", "daytime"):
        return "Day"

    if lower in ("1", "night", "nighttime"):
        return "Night"

    try:
        key = int(float(text))
        return TIME_MAP.get(key, clean_name(text))
    except Exception:
        return clean_name(text)


def normalise_vehicle(value):
    """Normalise a vehicle filter into the dataset vehicle folder name.

        Args:
            value: A vehicle name, numeric vehicle id, or wildcard marker.

        Returns:
            Canonical vehicle name, cleaned fallback text, or ``None`` when all
            vehicle folders are allowed.
        """
    if is_all_marker(value):
        return None

    text = str(value).strip()
    canonical_by_name = {name.casefold(): name for name in VEHICLE_MAP.values()}

    if text.casefold() in canonical_by_name:
        return canonical_by_name[text.casefold()]

    try:
        key = int(float(text))
        return VEHICLE_MAP.get(key, clean_name(text))
    except Exception:
        return clean_name(text)


def clean_video_id(video_id):
    """Clean a video id filter and remove any file extension.

        Args:
            video_id: Raw video id, filename, or wildcard marker.

        Returns:
            A cleaned video id without extension, or ``None`` when all videos are
            allowed.
        """
    if is_all_marker(video_id):
        return None

    value = str(video_id).strip()
    value = os.path.splitext(value)[0]
    return clean_name(value)


def clean_filter_list(values, normalise_func=None):
    """Clean and normalise a possibly wildcard filter list.

        Args:
            values: Raw filter values from configuration.
            normalise_func: Optional function used to map each value to the dataset
                folder naming convention.

        Returns:
            A deduplicated list of cleaned values. An empty list means no filter is
            applied for that dimension.
        """
    values = ensure_list(values)

    if not values:
        return []

    cleaned = []

    for value in values:
        if is_all_marker(value):
            return []

        if normalise_func is None:
            name = clean_name(value)
        else:
            name = normalise_func(value)

        if name is None:
            return []

        if name != "unknown":
            cleaned.append(name)

    return unique_preserve_order(cleaned)


# =============================================================================
# LOCATION_TREE parsing
# =============================================================================

def parse_location_tree(location_tree):
    """Convert the nested ``LOCATION_TREE`` config into location filters.

        Args:
            location_tree: Nested dictionary in the form continent -> country ->
                state -> cities, or a wildcard marker.

        Returns:
            A list of ``LocationFilter`` objects. A single empty filter means that
            every location in the remote tree is eligible.

        Raises:
            ValueError: If the tree shape is invalid for the expected hierarchy.
        """
    # A wildcard location tree means the sampler should discover every
    # concrete location from the server rather than restricting traversal.
    if is_all_marker(location_tree):
        return [LocationFilter()]

    if not isinstance(location_tree, dict):
        raise ValueError("LOCATION_TREE must be a dictionary.")

    filters = []

    # Walk the tree level by level so validation errors can identify the
    # exact hierarchy level that has the wrong shape.
    for continent, countries in location_tree.items():
        continent_name = clean_name(continent)

        if is_all_marker(countries):
            filters.append(LocationFilter(continent=continent_name))
            continue

        if not isinstance(countries, dict):
            raise ValueError(f"Countries under '{continent_name}' must be a dictionary.")

        for country, states in countries.items():
            country_name = clean_name(country)

            if is_all_marker(states):
                filters.append(LocationFilter(continent=continent_name, country=country_name))
                continue

            if not isinstance(states, dict):
                raise ValueError(
                    f"States under '{continent_name} / {country_name}' must be a dictionary. "
                    "If the state is missing in the dataset, use the state key 'unknown'."
                )

            for state, cities in states.items():
                state_name = clean_name(state)

                if is_all_marker(cities):
                    filters.append(
                        LocationFilter(
                            continent=continent_name,
                            country=country_name,
                            state=state_name,
                        )
                    )
                    continue

                city_list = clean_filter_list(cities)

                if not city_list:
                    filters.append(
                        LocationFilter(
                            continent=continent_name,
                            country=country_name,
                            state=state_name,
                        )
                    )
                    continue

                for city in city_list:
                    filters.append(
                        LocationFilter(
                            continent=continent_name,
                            country=country_name,
                            state=state_name,
                            city=city,
                        )
                    )

    if not filters:
        return [LocationFilter()]

    return unique_preserve_order(filters)


# =============================================================================
# FTP style file server helpers
# =============================================================================

def quote_path(parts):
    """Quote path components for safe use inside a URL path.

        Args:
            parts: Folder path components in dataset order.

        Returns:
            A slash joined, URL encoded path string with empty parts removed.
        """
    clean_parts = []

    for part in parts:
        # Remove slashes before quoting so each part remains one path segment.
        part = str(part).strip().strip("/")

        if part:
            clean_parts.append(quote(part, safe=""))

    return "/".join(clean_parts)


def make_browse_url(base_url, alias, folder_parts, trailing_slash=True):
    """Build a browse URL for one alias and folder path.

        Args:
            base_url: Root file server URL.
            alias: Server alias, for example ``tue5``.
            folder_parts: Dataset folder components under the alias browse root.
            trailing_slash: Whether to append a trailing slash to the browse URL.

        Returns:
            Absolute URL pointing at the requested browse folder.
        """
    base = base_url.rstrip("/") + "/"
    suffix = f"v/{quote(str(alias), safe='')}/browse"
    path = quote_path(folder_parts)

    if path:
        suffix += f"/{path}"

    if trailing_slash:
        suffix += "/"

    return urljoin(base, suffix)


def make_session():
    """Create a configured HTTP session for file server requests.

        Returns:
            A ``requests.Session`` with optional basic authentication and a custom
            user agent identifying this sampler.
        """
    session = requests.Session()

    # Only attach basic auth when both credentials are available.
    if FTP_USERNAME and FTP_PASSWORD:
        session.auth = (FTP_USERNAME, FTP_PASSWORD)

    session.headers.update({"User-Agent": "random-frame-sampler/direct-paths"})
    return session


def fetch_text(session, url, quiet=False):
    """Fetch HTML text from a browse URL with in memory caching.

        Args:
            session: Active HTTP session.
            url: URL to fetch.
            quiet: Reserved flag for quieter fallback fetches. The current
                implementation keeps errors silent and returns ``None``.

        Returns:
            Response text when the request succeeds, cached text for repeated URLs,
            or ``None`` when the request fails.
        """
    # Browse pages are frequently revisited during balancing and fill passes.
    if url in BROWSE_CACHE:
        return BROWSE_CACHE[url]

    try:
        response = session.get(
            url,
            auth=(FTP_USERNAME, FTP_PASSWORD),
            timeout=30,
        )
        response.raise_for_status()
        BROWSE_CACHE[url] = response.text
        return response.text

    except requests.RequestException:
        return None


def fetch_folder(session, base_url, alias, folder_parts):
    """Fetch a browse folder, trying slash and no slash URL forms.

        Args:
            session: Active HTTP session.
            base_url: Root file server URL.
            alias: Server alias to browse.
            folder_parts: Folder path components under the alias.

        Returns:
            Tuple of ``(browse_url, html_text)``. Both values are ``None`` if the
            folder cannot be fetched.
        """
    urls = [
        make_browse_url(base_url, alias, folder_parts, trailing_slash=True),
        make_browse_url(base_url, alias, folder_parts, trailing_slash=False),
    ]

    # Some servers canonicalise folders with a trailing slash; others accept
    # the path without one. Try both before treating the folder as missing.
    for index, url in enumerate(urls):
        html_text = fetch_text(session, url, quiet=(index > 0))
        if html_text is not None:
            return url, html_text

    return None, None


def get_anchor_name(anchor, full_url):
    """Extract a clean folder or file name from an HTML anchor.

        Args:
            anchor: BeautifulSoup anchor element.
            full_url: Absolute URL represented by the anchor.

        Returns:
            Clean anchor text when available, otherwise the decoded final URL path
            component.
        """
    text = anchor.get_text(" ", strip=True).strip()

    if text:
        text = text.rstrip("/")
        return clean_name(text)

    path_tail = os.path.basename(unquote(urlparse(full_url).path).rstrip("/"))
    return clean_name(path_tail)


def parse_browse_html(browse_url, html_text):
    """Parse a file server browse page into child folders and files.

        Args:
            browse_url: URL of the page being parsed. It is used to filter out
                parent or root links.
            html_text: Raw HTML returned by the browse endpoint.

        Returns:
            Tuple ``(folders, files)``. Each list contains ``(name, url)`` pairs,
            deduplicated while preserving page order.
        """
    soup = BeautifulSoup(html_text, "html.parser")
    folders = []
    files = []

    # The current path is used to filter out links that point back upward.
    current_path = urlparse(browse_url).path.rstrip("/")

    for anchor in soup.find_all("a"):
        href = (anchor.get("href") or "").strip()
        label = anchor.get_text(" ", strip=True).strip()

        if not href:
            continue

        # Skip navigation links, not real data folders.
        if label in ("..", "⬅ Back") or label.lower() == "back":
            continue

        full_url = urljoin(browse_url, href)
        parsed_path = urlparse(full_url).path.rstrip("/")

        if "/browse" in parsed_path:
            # Keep only children of the current folder, not parent/root links.
            if not parsed_path.startswith(current_path + "/"):
                continue

            name = os.path.basename(unquote(parsed_path))
            name = clean_name(name)
            folders.append((name, full_url))

        elif "/files/" in parsed_path:
            name = os.path.basename(unquote(parsed_path))
            name = clean_name(name)
            files.append((name, full_url))

    return unique_preserve_order(folders), unique_preserve_order(files)


def list_folder(session, alias, folder_parts):
    """List child folders and files for a remote dataset folder.

        Args:
            session: Active HTTP session.
            alias: Server alias to browse.
            folder_parts: Folder components under the alias browse root.

        Returns:
            Tuple ``(folders, files)`` from the browse page. Empty lists are returned
            when the folder cannot be fetched.
        """
    browse_url, html_text = fetch_folder(session, BASE_URL, alias, folder_parts)

    if html_text is None:
        return [], []

    return parse_browse_html(browse_url, html_text)


def list_child_folders(session, alias, folder_parts):
    """Return only child folders from a remote dataset folder.

        Args:
            session: Active HTTP session.
            alias: Server alias to browse.
            folder_parts: Folder components under the alias browse root.

        Returns:
            List of ``(folder_name, folder_url)`` pairs.
        """
    folders, _ = list_folder(session, alias, folder_parts)
    return folders


def is_image_url(url):
    """Return whether a URL points to a supported image extension.

        Args:
            url: Remote file URL.

        Returns:
            ``True`` when the decoded URL path ends with a supported image suffix.
        """
    path = unquote(urlparse(url).path).casefold()
    return path.endswith(IMAGE_EXTENSIONS)


def file_name_from_url(url):
    """Build a clean local filename from a remote URL.

        Args:
            url: Remote file URL.

        Returns:
            Cleaned basename from the decoded URL path.
        """
    return clean_name(os.path.basename(unquote(urlparse(url).path)))


def get_folder_names_from_filter(session, alias, folder_parts, wanted_name):
    """Resolve either a requested folder name or all child folders.

        Args:
            session: Active HTTP session.
            alias: Server alias to browse.
            folder_parts: Current folder path components.
            wanted_name: Specific child folder name, or ``None`` for all children.

        Returns:
            List of ``(name, path_parts)`` pairs for the next traversal level.
        """
    if wanted_name is not None:
        return [(wanted_name, folder_parts + [wanted_name])]

    child_folders = list_child_folders(session, alias, folder_parts)
    return [(name, folder_parts + [name]) for name, _ in child_folders]


def collect_candidates_from_video_folder(
    session,
    alias,
    folder_parts,
    continent_name,
    country_name,
    state_name,
    city_name,
    day_name,
    vehicle_name,
    video_name,
    max_images,
    seen_urls,
):
    """Sample candidate image files from one video folder.

        Args:
            session: Active HTTP session.
            alias: Server alias where the folder exists.
            folder_parts: Full dataset path to a video folder.
            continent_name: Continent metadata for the candidate.
            country_name: Country metadata for the candidate.
            state_name: State metadata for the candidate.
            city_name: City metadata for the candidate.
            day_name: Day/night metadata for the candidate.
            vehicle_name: Vehicle metadata for the candidate.
            video_name: Source video folder name.
            max_images: Maximum number of image candidates to return.
            seen_urls: URLs already selected elsewhere in the run.

        Returns:
            Randomly ordered ``FrameCandidate`` objects from this video folder.
        """
    _, files = list_folder(session, alias, folder_parts)

    # Keep only supported image files and remove URLs already selected in
    # another location or balancing pass.
    image_files = [
        (filename, file_url)
        for filename, file_url in files
        if is_image_url(file_url) and file_url not in seen_urls
    ]

    # Shuffle before truncating so each video folder contributes a random
    # subset rather than the first files listed by the server.
    random.shuffle(image_files)

    candidates = []
    for _, file_url in image_files[:max_images]:
        candidates.append(
            FrameCandidate(
                url=file_url,
                filename=file_name_from_url(file_url),
                alias=str(alias),
                continent=continent_name,
                country=country_name,
                state=state_name,
                city=city_name,
                day_night=day_name,
                vehicle=vehicle_name,
                video_id=video_name,
            )
        )

    return candidates


def collect_candidates_for_single_location(
    session,
    alias,
    location_filter,
    day_nights,
    vehicles,
    video_ids,
    target_count,
    seen_urls,
    max_images_per_video_folder,
):
    """Collect matching frame candidates for one concrete location filter.

        Args:
            session: Active HTTP session.
            alias: Server alias to scan.
            location_filter: Continent/country/state/city selector.
            day_nights: Allowed day/night folder names. Empty means all.
            vehicles: Allowed vehicle folder names. Empty means all.
            video_ids: Allowed video id folder names. Empty means all.
            target_count: Maximum number of candidates to collect.
            seen_urls: Global URL set used to prevent duplicate candidates.
            max_images_per_video_folder: Per video folder sampling cap.

        Returns:
            Tuple ``(candidates, checked_video_folders)`` for this location.
        """
    candidates = []
    missing_vehicle_messages = set()
    checked_video_folders = 0

    # Precompute normalised lookup sets once; the nested traversal can inspect
    # many folders, so repeated normalisation would add unnecessary overhead.
    wanted_days = {normalise_for_match(value) for value in day_nights}
    wanted_vehicles = {normalise_for_match(value) for value in vehicles}
    wanted_videos = {normalise_for_match(value) for value in video_ids}

    # Start at the alias root and progressively descend through the dataset
    # hierarchy: continent -> country -> state -> city -> day -> vehicle -> video.
    continent_options = get_folder_names_from_filter(session, alias, [], location_filter.continent)
    random.shuffle(continent_options)

    for continent_name, continent_parts in continent_options:
        country_options = get_folder_names_from_filter(
            session, alias, continent_parts, location_filter.country
        )
        random.shuffle(country_options)

        for country_name, country_parts in country_options:
            state_options = get_folder_names_from_filter(
                session, alias, country_parts, location_filter.state
            )
            random.shuffle(state_options)

            for state_name, state_parts in state_options:
                city_options = get_folder_names_from_filter(
                    session, alias, state_parts, location_filter.city
                )
                random.shuffle(city_options)

                if location_filter.city is not None and not city_options:
                    logger.warning(
                        "City not found: "
                        f"{continent_name} / {country_name} / "
                        f"{state_name} / {location_filter.city}"
                    )

                for city_name, city_parts in city_options:
                    day_options = get_folder_names_from_filter(session, alias, city_parts, None)

                    if wanted_days:
                        day_options = [
                            (name, parts)
                            for name, parts in day_options
                            if normalise_for_match(name) in wanted_days
                        ]

                    random.shuffle(day_options)

                    if day_nights and not day_options:
                        logger.warning(
                            "Day/Night not found for city: "
                            f"{continent_name} / {country_name} / "
                            f"{state_name} / {city_name} requested={day_nights}"
                        )

                    for day_name, day_parts in day_options:
                        vehicle_options_all = get_folder_names_from_filter(session, alias, day_parts, None)

                        if wanted_vehicles:
                            available_vehicle_keys = {
                                normalise_for_match(name)
                                for name, _ in vehicle_options_all
                            }

                            missing_vehicles = [
                                vehicle
                                for vehicle in vehicles
                                if normalise_for_match(vehicle) not in available_vehicle_keys
                            ]

                            if missing_vehicles:
                                message = (
                                    "Vehicle not found: "
                                    f"{continent_name} / {country_name} / "
                                    f"{state_name} / {city_name} / {day_name} "
                                    f"missing={missing_vehicles}"
                                )
                                missing_vehicle_messages.add(message)

                            vehicle_options = [
                                (name, parts)
                                for name, parts in vehicle_options_all
                                if normalise_for_match(name) in wanted_vehicles
                            ]
                        else:
                            vehicle_options = vehicle_options_all

                        random.shuffle(vehicle_options)

                        for vehicle_name, vehicle_parts in vehicle_options:
                            video_options = get_folder_names_from_filter(
                                session,
                                alias,
                                vehicle_parts,
                                None,
                            )

                            if wanted_videos:
                                video_options = [
                                    (name, parts)
                                    for name, parts in video_options
                                    if normalise_for_match(name) in wanted_videos
                                ]

                                if not video_options:
                                    logger.warning(
                                        "Video ID not found: "
                                        f"{continent_name} / {country_name} / "
                                        f"{state_name} / {city_name} / "
                                        f"{day_name} / {vehicle_name} "
                                        f"requested={video_ids}"
                                    )

                            random.shuffle(video_options)

                            for video_name, video_parts in video_options:
                                if len(candidates) >= target_count:
                                    break

                                if checked_video_folders >= MAX_VIDEO_FOLDERS_TO_CHECK:
                                    logger.warning(
                                        f"Stopped after checking {MAX_VIDEO_FOLDERS_TO_CHECK} "
                                        "video folders. Increase MAX_VIDEO_FOLDERS_TO_CHECK if needed."
                                    )
                                    break

                                checked_video_folders += 1
                                # Avoid reading more images than still needed for this request.
                                remaining = target_count - len(candidates)
                                max_images = min(max_images_per_video_folder, remaining)

                                sampled = collect_candidates_from_video_folder(
                                    session=session,
                                    alias=alias,
                                    folder_parts=video_parts,
                                    continent_name=continent_name,
                                    country_name=country_name,
                                    state_name=state_name,
                                    city_name=city_name,
                                    day_name=day_name,
                                    vehicle_name=vehicle_name,
                                    video_name=video_name,
                                    max_images=max_images,
                                    seen_urls=seen_urls,
                                )

                                for item in sampled:
                                    # The same file can be reached in fill passes, so guard
                                    # against duplicates immediately before appending.
                                    if item.url in seen_urls:
                                        continue

                                    seen_urls.add(item.url)
                                    candidates.append(item)

                                    if len(candidates) >= target_count:
                                        break

                            if len(candidates) >= target_count:
                                break

                        if len(candidates) >= target_count:
                            break

                    if len(candidates) >= target_count:
                        break

                if len(candidates) >= target_count:
                    break

            if len(candidates) >= target_count:
                break

        if len(candidates) >= target_count:
            break

    # Emit missing vehicle messages once per location rather than repeatedly
    # for every day or video folder examined.
    for message in sorted(missing_vehicle_messages):
        logger.info(message)

    return candidates, checked_video_folders


def expand_location_filters_to_city_filters(session, alias, location_filters):
    """Expand broad location filters into concrete city level filters.

        Broad filters are convenient for users, but sampling should happen at the
        city level so that balancing has meaningful units. This function traverses
        only the location part of the tree and intentionally does not inspect image
        files.

        Args:
            session: Active HTTP session.
            alias: Server alias to scan.
            location_filters: User supplied location filters, possibly broad.

        Returns:
            Deduplicated list of city level ``LocationFilter`` objects.
        """
    expanded = []

    # Convert each possibly broad filter into one filter per concrete city.
    for location_filter in location_filters:
        continent_options = get_folder_names_from_filter(
            session,
            alias,
            [],
            location_filter.continent,
        )
        random.shuffle(continent_options)

        for continent_name, continent_parts in continent_options:
            country_options = get_folder_names_from_filter(
                session,
                alias,
                continent_parts,
                location_filter.country,
            )
            random.shuffle(country_options)

            for country_name, country_parts in country_options:
                state_options = get_folder_names_from_filter(
                    session,
                    alias,
                    country_parts,
                    location_filter.state,
                )
                random.shuffle(state_options)

                for state_name, state_parts in state_options:
                    city_options = get_folder_names_from_filter(
                        session,
                        alias,
                        state_parts,
                        location_filter.city,
                    )
                    random.shuffle(city_options)

                    if location_filter.city is not None and not city_options:
                        logger.warning(
                            "City not found: "
                            f"{continent_name} / {country_name} / "
                            f"{state_name} / {location_filter.city}"
                        )

                    for city_name, _ in city_options:
                        expanded.append(
                            LocationFilter(
                                continent=continent_name,
                                country=country_name,
                                state=state_name,
                                city=city_name,
                            )
                        )

    return unique_preserve_order(expanded)


def country_balanced_location_order(location_filters):
    """Order city filters so countries are interleaved during sampling.

        Args:
            location_filters: City level location filters.

        Returns:
            A shuffled but country balanced list of filters. The order draws one
            city from each country group before returning to the next city from the
            same country.
        """
    groups = {}

    # Group by country first, then interleave groups to reduce sampling bias.
    for location_filter in location_filters:
        key = (
            location_filter.continent,
            location_filter.country,
        )
        groups.setdefault(key, []).append(location_filter)

    for filters in groups.values():
        random.shuffle(filters)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    ordered = []

    # Rotate through country groups until every city filter has been emitted.
    while group_keys:
        next_group_keys = []

        for key in group_keys:
            filters = groups[key]

            if filters:
                ordered.append(filters.pop())

            if filters:
                next_group_keys.append(key)

        group_keys = next_group_keys
        random.shuffle(group_keys)

    return ordered


def collect_candidates_for_alias(
    session,
    alias,
    location_filters,
    day_nights,
    vehicles,
    video_ids,
    target_count,
):
    """Collect candidate images for one server alias.

        Args:
            session: Active HTTP session.
            alias: Server alias to scan.
            location_filters: Location filters derived from ``LOCATION_TREE``.
            day_nights: Allowed day/night values. Empty means all.
            vehicles: Allowed vehicle values. Empty means all.
            video_ids: Allowed video ids. Empty means all.
            target_count: Desired number of candidates.

        Returns:
            List of matching ``FrameCandidate`` objects for the alias.
        """
    location_filters = expand_location_filters_to_city_filters(
        session=session,
        alias=alias,
        location_filters=location_filters,
    )

    if not location_filters:
        return []

    location_filters = country_balanced_location_order(location_filters)

    logger.info(
        f"Expanded to {len(location_filters)} city-level location filter(s)."
    )

    # Non-balanced mode is faster and simply fills the target count in the
    # current shuffled location order.
    if not BALANCE_ACROSS_LOCATIONS:
        candidates = []
        seen_urls = set()
        total_checked_video_folders = 0

        for location_filter in location_filters:
            if len(candidates) >= target_count:
                break

            needed = target_count - len(candidates)
            sampled, checked = collect_candidates_for_single_location(
                session=session,
                alias=alias,
                location_filter=location_filter,
                day_nights=day_nights,
                vehicles=vehicles,
                video_ids=video_ids,
                target_count=needed,
                seen_urls=seen_urls,
                max_images_per_video_folder=MAX_IMAGES_PER_VIDEO_FOLDER,
            )
            candidates.extend(sampled)
            total_checked_video_folders += checked

        logger.info(f"Checked {total_checked_video_folders} video folder(s).")
        return candidates

    candidates = []
    seen_urls = set()
    total_checked_video_folders = 0

    # If there are more cities than requested images, take at most one image
    # from each visited city in the first pass.
    # The first pass spreads quota across cities. If there are more cities
    # than requested frames, the quota becomes one frame per visited city.
    first_pass_location_count = min(len(location_filters), target_count)
    first_pass_quota = max(1, math.ceil(target_count / first_pass_location_count))

    location_counts = {}

    for location_filter in location_filters:
        if len(candidates) >= target_count:
            break

        sampled, checked = collect_candidates_for_single_location(
            session=session,
            alias=alias,
            location_filter=location_filter,
            day_nights=day_nights,
            vehicles=vehicles,
            video_ids=video_ids,
            target_count=first_pass_quota,
            seen_urls=seen_urls,
            max_images_per_video_folder=MAX_IMAGES_PER_VIDEO_FOLDER,
        )

        candidates.extend(sampled)
        total_checked_video_folders += checked

        location_key = (
            location_filter.continent,
            location_filter.country,
            location_filter.state,
            location_filter.city,
        )
        location_counts[location_key] = len(sampled)

    # Second pass: if some locations had fewer frames, fill the remaining
    # request from any location that still has data.
    if len(candidates) < target_count:
        # The fill pass is intentionally less strict, allowing more images per
        # video folder to satisfy the requested total when some cities are sparse.
        random.shuffle(location_filters)

        for location_filter in location_filters:
            if len(candidates) >= target_count:
                break

            remaining = target_count - len(candidates)

            sampled, checked = collect_candidates_for_single_location(
                session=session,
                alias=alias,
                location_filter=location_filter,
                day_nights=day_nights,
                vehicles=vehicles,
                video_ids=video_ids,
                target_count=remaining,
                seen_urls=seen_urls,
                max_images_per_video_folder=FILL_IMAGES_PER_VIDEO_FOLDER,
            )

            candidates.extend(sampled)
            total_checked_video_folders += checked

    logger.info(f"Checked {total_checked_video_folders} video folder(s).")

    country_counts = {}
    for item in candidates:
        key = (item.continent, item.country)
        country_counts[key] = country_counts.get(key, 0) + 1

    return candidates


# =============================================================================
# Output cleanup helpers
# =============================================================================

def clear_previous_random_outputs(output_root):
    """Delete old random sampler outputs from the local output folder.

        Args:
            output_root: Folder configured as the random frame output destination.

        Raises:
            ValueError: If ``output_root`` resolves to an unsafe broad path such as
            the filesystem root, the current directory, or the user home directory.
        """
    if not CLEAR_PREVIOUS_RANDOM_OUTPUTS:
        return

    output_root = os.path.abspath(output_root)

    # Safety checks so a wrong config does not delete important folders.
    # Refuse destructive cleanup when configuration accidentally points at a
    # broad system folder. This protects users from deleting unrelated files.
    unsafe_paths = {
        os.path.abspath(os.sep),
        os.path.abspath(os.path.expanduser("~")),
        os.path.abspath("."),
    }

    if output_root in unsafe_paths:
        raise ValueError(
            f"Refusing to clear unsafe output folder: {output_root}"
        )

    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
        logger.info(f"Output folder created: {output_root}")
        return

    deleted_files = 0
    deleted_dirs = 0

    for current_root, dir_names, file_names in os.walk(output_root, topdown=False):
        for file_name in file_names:
            file_path = os.path.join(current_root, file_name)
            lower_name = file_name.casefold()

            # Only remove files produced by this sampler; unrelated files stay.
            is_frame_file = lower_name.endswith(IMAGE_EXTENSIONS)
            is_temp_frame_file = any(
                lower_name.endswith(ext + ".tmp")
                for ext in IMAGE_EXTENSIONS
            )
            is_csv_file = lower_name.endswith(".csv")

            if not (is_frame_file or is_temp_frame_file or is_csv_file):
                continue

            try:
                os.remove(file_path)
                deleted_files += 1
            except OSError as error:
                logger.warning(f"Could not delete old output file {file_path}: {error}")

        # Remove empty sub-folders left from older nested output versions.
        if current_root == output_root:
            continue

        try:
            os.rmdir(current_root)
            deleted_dirs += 1
        except OSError:
            pass

    logger.info(
        f"Cleared old random outputs from {output_root}: "
        f"deleted {deleted_files} file(s) and {deleted_dirs} empty folder(s)."
    )


# =============================================================================
# Download helpers
# =============================================================================


def unique_local_path(folder, filename):
    """Return a file path that does not overwrite an existing local file.

        Args:
            folder: Destination folder.
            filename: Preferred filename.

        Returns:
            A path using ``filename`` when available, otherwise a suffixed variant
            such as ``name_1.ext``.
        """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    # Prefer the original filename when possible so manifest rows remain easy
    # to compare with source URLs.
    if not os.path.exists(path):
        return path

    stem, ext = os.path.splitext(path)
    counter = 1

    # Add a numeric suffix until a free path is found.
    while True:
        candidate = f"{stem}_{counter}{ext}"

        if not os.path.exists(candidate):
            return candidate

        counter += 1


def local_folder_for_candidate(candidate):
    """Return the local destination folder for a candidate frame.

        Args:
            candidate: Frame candidate selected for download.

        Returns:
            Current flat output folder. The function exists so nested output layouts
            can be introduced later without changing download code.
        """
    return LOCAL_OUTPUT_ROOT


def download_image(session, candidate):
    """Download one selected frame image to local storage.

        Args:
            session: Active HTTP session.
            candidate: Frame candidate containing URL and filename metadata.

        Returns:
            Final local path when the download succeeds, otherwise ``None``. The
            function writes to a temporary path first and promotes it atomically.
        """
    output_folder = local_folder_for_candidate(candidate)
    final_path = unique_local_path(output_folder, candidate.filename)
    # Temporary downloads prevent partially written files from looking valid
    # if the process is interrupted.
    temp_path = final_path + ".tmp"

    try:
        with session.get(candidate.url, auth=(FTP_USERNAME, FTP_PASSWORD), timeout=30, stream=True) as response:
            response.raise_for_status()

            with open(temp_path, "wb") as file_obj:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_obj.write(chunk)

        # Atomic promotion makes completed files visible only after the full
        # response body has been written successfully.
        os.replace(temp_path, final_path)
        return final_path

    except Exception as exc:
        logger.error(f"Failed to download {candidate.url}: {exc}")

        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

        return None


def save_manifest(records):
    """Write a CSV manifest for downloaded random frames.

        Args:
            records: Manifest rows containing local path, source URL, and metadata.

        Returns:
            Path to the created manifest, or ``None`` when manifest writing is
            disabled by configuration.
        """
    if not SAVE_RANDOM_FRAMES_CSV:
        return None

    os.makedirs(LOCAL_OUTPUT_ROOT, exist_ok=True)
    manifest_path = unique_local_path(LOCAL_OUTPUT_ROOT, "selected_random_frames.csv")

    # Keep the manifest schema explicit so downstream scripts can rely on the
    # column order and names.
    fieldnames = [
        "local_path",
        "source_url",
        "alias",
        "continent",
        "country",
        "state",
        "city",
        "day_night",
        "vehicle",
        "video_id",
        "file_name",
    ]

    with open(manifest_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return manifest_path


# =============================================================================
# Main sampling function
# =============================================================================

def get_random_frames_from_common_config():
    """Run random frame sampling using values from the shared config module.

        Returns:
            Tuple ``(records, manifest_path)``. ``records`` contains one dictionary
            per successfully downloaded frame. ``manifest_path`` is the CSV path or
            ``None`` when manifest saving is disabled.

        Raises:
            ValueError: If required configuration values are missing or invalid.
            FileNotFoundError: If no remote images match the configured filters.
        """
    if not BASE_URL:
        raise ValueError("base_url is missing in common config.")

    num_images = int(NUM_IMAGES)

    if num_images <= 0:
        raise ValueError("num_images must be greater than 0.")

    # Normalise all user filters once near the entry point. Downstream code
    # can then assume canonical folder names or empty wildcard lists.
    location_filters = parse_location_tree(LOCATION_TREE)
    day_nights = clean_filter_list(DAY_NIGHTS, normalise_day_night)
    vehicles = clean_filter_list(VEHICLES, normalise_vehicle)
    video_ids = clean_filter_list(VIDEO_IDS, clean_video_id)

    logger.info(f"Requested images: {num_images}")
    logger.info(f"FTP username loaded: {bool(FTP_USERNAME)}")
    logger.info(f"FTP password loaded: {bool(FTP_PASSWORD)}")
    logger.info(f"Location filters: {len(location_filters)}")
    logger.info(f"Day/Night: {day_nights if day_nights else 'all'}")
    logger.info(f"Vehicles: {vehicles if vehicles else 'all'}")
    logger.info(f"Video IDs: {video_ids if video_ids else 'all'}")
    logger.info(f"Max images per video folder first pass: {MAX_IMAGES_PER_VIDEO_FOLDER}")
    logger.info(f"Max images per video folder fill pass: {FILL_IMAGES_PER_VIDEO_FOLDER}")

    # Clear previous outputs before network work so the destination reflects
    # only the current sampling run when cleanup is enabled.
    clear_previous_random_outputs(LOCAL_OUTPUT_ROOT)

    session = make_session()
    discovered = []

    # This script currently samples from tue5. Add more aliases here if the
    # random frame pool should span additional server roots.
    for alias in ["tue5"]:
        logger.info(f"\nScanning alias={alias}")

        alias_candidates = collect_candidates_for_alias(
            session=session,
            alias=alias,
            location_filters=location_filters,
            day_nights=day_nights,
            vehicles=vehicles,
            video_ids=video_ids,
            target_count=num_images,
        )

        logger.info(f"Sampled {len(alias_candidates)} candidate frame image(s) on alias={alias}")
        discovered.extend(alias_candidates)

    unique_by_url = {}

    # Deduplicate after all aliases are scanned so the final shuffle has unique
    # source images only.
    for item in discovered:
        unique_by_url[item.url] = item

    discovered = list(unique_by_url.values())

    logger.info(f"\nTotal unique matching frames: {len(discovered)}")

    if not discovered:
        raise FileNotFoundError("No frame images were found for the given config.")

    # Shuffle the full candidate pool immediately before selection so all
    # matching frames have equal chance after discovery and deduplication.
    random.shuffle(discovered)

    selected_count = min(num_images, len(discovered))
    selected = discovered[:selected_count]

    if selected_count < num_images:
        logger.info(
            f"Requested {num_images} image(s), but only {selected_count} "
            "matching image(s) are available. "
            f"Downloading those {selected_count}."
        )

    records = []

    logger.info(f"Downloading {selected_count} random frame image(s)...")

    # Download selected candidates one by one so failures can be logged while
    # the remaining frames still proceed.
    for item in tqdm(selected, unit="frame"):
        local_path = download_image(session, item)

        if local_path:
            records.append(
                {
                    "local_path": local_path,
                    "source_url": item.url,
                    "alias": item.alias,
                    "continent": item.continent,
                    "country": item.country,
                    "state": item.state,
                    "city": item.city,
                    "day_night": item.day_night,
                    "vehicle": item.vehicle,
                    "video_id": item.video_id,
                    "file_name": item.filename,
                }
            )

    manifest_path = save_manifest(records)

    logger.info(f"\nFinished. Downloaded {len(records)} random frame image(s).")
    if len(records) < num_images:
        logger.info(
            f"Only {len(records)} image(s) were saved, which is less than the requested {num_images}, "
            "because not enough matching images were available."
        )
    logger.info(f"Output root: {LOCAL_OUTPUT_ROOT}")

    if manifest_path:
        logger.info(f"CSV manifest: {manifest_path}")
    else:
        logger.info("CSV manifest: disabled")

    return records, manifest_path


def main():
    """Execute the sampler from command line entry point."""
    get_random_frames_from_common_config()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logger.error(f"Error: {error}")
