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


# vehicle type mapping
vehicle_map = {
    0: "Car",
    1: "Bus",
    2: "Truck",
    3: "Two wheeler",
    4: "Bicycle",
    5: "Automated car",
    6: "Electric scooter",
    7: "Monowheel unicycle",
    8: "Automated bus",
    9: "Automated truck",
    10: "Automated two wheeler",
    11: "Non electric scooter",
    12: "Pedestrian"
}

# time of day mapping
time_map = {
    0: "Day",
    1: "Night"
}


# ----------------------------
# General helpers
# ----------------------------
def clean_name(value):
    if value is None:
        return "unknown"

    value = str(value).strip()

    if value in ("", "[]", "None", "null", "nan"):
        return "unknown"

    value = re.sub(r'[<>:"/\\|?*]+', "_", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value if value else "unknown"


def is_missing_field(value):
    if value is None:
        return True

    value = str(value).strip()
    return value in ("", "[]", "None", "null", "nan", "unknown")


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_scalar(value):
    current = value
    while isinstance(current, list):
        if not current:
            return None
        current = current[0]
    return current


def parse_bracket_value(text):
    """
    Parse strings like:
      [a,b,c]
      [[0],[1,2],[3]]
      [10,20,30]
    into Python lists without eval.
    """
    if text is None:
        return []

    text = str(text).strip()
    if text == "":
        return []

    i = 0
    n = len(text)

    def skip_spaces():
        nonlocal i
        while i < n and text[i].isspace():
            i += 1

    def parse_value():
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
    items = ensure_list(value)
    result = []

    for item in items:
        if item is None or item == "":
            continue
        result.append(int(float(item)))

    return result


def normalise_scalar_values(value):
    result = []

    def visit(item):
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
def build_video_index(root_folder):
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
    file_name = os.path.basename(video_path)
    stem = os.path.splitext(file_name)[0]

    video_index[file_name.lower()] = video_path
    video_index[stem.lower()] = video_path


def find_video_path(video_ref, video_index):
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
def get_video_metadata(video_path):
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
    """
    Download a specific mp4 from the file server.

    It first tries direct /files paths.
    If not found, it crawls /browse pages.

    The file is downloaded to .tmp first and then moved to the final path.
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
            return href.startswith("/v/") and "/browse" in href

        def is_file_link(href):
            return "/files/" in href

        def crawl(start_url):
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
def extract_frames_for_ranges(video_path, output_folder, ranges):
    """
    Extract frames only inside the given ranges.

    New behaviour:
      - Frames are first written to a temporary folder: <output_folder>.tmp
      - If a stale temp folder exists from an earlier interrupted run, it is deleted
      - If the final output is incomplete, the existing final frames are deleted
        and the video is processed again from zero
      - Frames are moved into the final folder only after the whole requested
        set has been extracted successfully

    Output names:
      videoName_time.png
    """
    parent_folder = os.path.dirname(output_folder)
    if parent_folder:
        os.makedirs(parent_folder, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_name = clean_name(video_name)

    requested_times = []
    seen = set()

    for start_sec, end_sec in ranges:
        current_time = start_sec
        while current_time <= end_sec:
            if current_time not in seen:
                seen.add(current_time)
                requested_times.append(current_time)
            current_time += cfg.interval_seconds

    if not requested_times:
        logger.warning(f"No frame times requested for {video_path}")
        return 0

    final_targets = []
    existing_final = []

    for current_time in requested_times:
        output_filename = f"{video_name}_{current_time}.png"
        output_path = os.path.join(output_folder, output_filename)
        final_targets.append((current_time, output_filename, output_path))

        if os.path.exists(output_path):
            existing_final.append(output_path)

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
            logger.warning(f"Promotion for {video_name} stopped. Remaining temp files are kept at {temp_output_folder}.")  # noqa: E501
            return promoted_count

        os.replace(temp_output_path, output_path)
        promoted_count += 1

    try:
        shutil.rmtree(temp_output_folder)
    except Exception as e:
        logger.warning(f"Could not remove temp folder {temp_output_folder}: {e}")

    logger.info(f"Finished {video_name}: extracted {saved_count} frame(s) to temp and promoted {promoted_count} to final")  # noqa: E501
    return promoted_count


def delete_downloaded_videos(downloaded_video_paths):
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
    raw = str(video_ref).strip().lower()
    if not raw:
        return ""
    return os.path.splitext(raw)[0]


def collect_video_usage_counts(csv_file):
    usage_counts = Counter()

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
    keys_to_remove = [key for key, value in video_index.items() if os.path.abspath(value) == os.path.abspath(video_path)]  # noqa: E501
    for key in keys_to_remove:
        video_index.pop(key, None)
    logger.info(f"Removed {len(keys_to_remove)} index entrie(s) for deleted video: {video_path}")


def delete_single_downloaded_video(video_path, downloaded_video_paths, video_index):
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


# ----------------------------
# Main
# ----------------------------
def main():
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

        for col in required_columns:
            if col not in reader.fieldnames:  # type: ignore
                raise ValueError(f"Missing required column in CSV: {col}")

        for row_num, row in enumerate(reader, start=2):
            continent = row.get("continent", "")
            country = row.get("country", "")
            state = row.get("state", "")
            locality = row.get("locality", "")

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
                logger.info(f"Row {row_num}: remaining uses before processing for '{video_key}' = {remaining_video_uses.get(video_key, 0)}")  # noqa: E501

                video_path = ensure_video_available(video_ref, video_index, downloaded_video_paths)

                if not video_path:
                    logger.warning(f"Row {row_num}: video not found and download failed -> {video_ref}")
                    if video_key in remaining_video_uses and remaining_video_uses[video_key] > 0:
                        remaining_video_uses[video_key] -= 1
                        logger.info(f"Row {row_num}: remaining uses after failed processing for '{video_key}' = {remaining_video_uses[video_key]}")  # noqa: E501
                    continue

                logger.info(f"Row {row_num}: resolved video path -> {video_path}")

                start_entry = starts[i] if i < len(starts) else []
                end_entry = ends[i] if i < len(ends) else []
                time_entry = time_of_day_values[i] if i < len(time_of_day_values) else None
                vehicle_entry = vehicle_types[i] if i < len(vehicle_types) else None

                count = 0
                grouped_ranges = build_segments_for_video(
                    start_entry=start_entry,
                    end_entry=end_entry,
                    time_entry=time_entry,
                    vehicle_entry=vehicle_entry,
                    video_ref=video_ref,
                )

                if not grouped_ranges:
                    logger.warning(f"Row {row_num}: no valid ranges for video -> {video_ref}")
                else:
                    actual_video_name = os.path.splitext(os.path.basename(video_path))[0]

                    for (time_folder, vehicle_folder), ranges in grouped_ranges.items():
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

                if video_key in remaining_video_uses and remaining_video_uses[video_key] > 0:
                    remaining_video_uses[video_key] -= 1

                remaining_after = remaining_video_uses.get(video_key, 0)
                logger.info(f"Row {row_num}: remaining uses after processing for '{video_key}' = {remaining_after}")

                if cfg.delete_downloaded_videos_after_processing:
                    if remaining_after == 0:
                        deleted_now = delete_single_downloaded_video(video_path, downloaded_video_paths, video_index)
                        logger.info(f"Row {row_num}: immediate delete attempted for '{video_key}' -> deleted={deleted_now}")  # noqa: E501
                    else:
                        logger.info(f"Row {row_num}: immediate delete skipped for '{video_key}' because future uses remain")  # noqa: E501

    logger.info("Reached end of CSV processing")
    logger.info(f"delete_downloaded_videos_after_processing={cfg.delete_downloaded_videos_after_processing}")
    logger.info(f"downloaded_video_paths_count={len(downloaded_video_paths)}")
    if downloaded_video_paths:
        for tracked_path in sorted(downloaded_video_paths):
            logger.info(f"Tracked for cleanup: {tracked_path}")

    if cfg.delete_downloaded_videos_after_processing:
        delete_downloaded_videos(downloaded_video_paths)
    else:
        logger.info("Cleanup skipped because delete_downloaded_videos_after_processing is False")

    logger.info("All done.")


if __name__ == "__main__":
    main()
