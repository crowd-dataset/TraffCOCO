import os
import re
import csv
import cv2
import common
import requests

from bs4 import BeautifulSoup
from tqdm import tqdm
from types import SimpleNamespace
from urllib.parse import urljoin, urlparse


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
    aliases=["tue1", "tue2", "tue3", "tue4"],
)

secrets = SimpleNamespace(
    username=common.get_secrets("ftp_username"),
    password=common.get_secrets("ftp_password"),
)


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
# Logging helper
# ----------------------------
def log(message, level="INFO"):
    if cfg.debug or level in ("WARNING", "ERROR"):
        print(f"[{level}] {message}")


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


def build_ranges_for_video(start_entry, end_entry):
    starts = normalise_time_values(start_entry)
    ends = normalise_time_values(end_entry)

    count = min(len(starts), len(ends))
    ranges = []

    for i in range(count):
        s = starts[i]
        e = ends[i]

        if e < s:
            log(f"Skipping invalid range: start={s}, end={e}", "WARNING")
            continue

        ranges.append((s, e))

    if len(starts) != len(ends):
        log(f"Mismatch between starts and ends: {starts} vs {ends}", "WARNING")

    return ranges


def map_time_of_day(value):
    if value is None or value == "":
        return "Unknown"

    try:
        key = int(float(value))
        return clean_name(time_map.get(key, f"Unknown_{key}"))
    except Exception:
        return clean_name(value)


def map_vehicle_type(value):
    if value is None or value == "":
        return "Unknown"

    if isinstance(value, list):
        names = []
        for item in value:
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
    locality = clean_name(locality)
    time_of_day = clean_name(time_of_day)
    vehicle_type = clean_name(vehicle_type)
    video_name = clean_name(video_name)

    if is_missing_field(state_raw):
        return os.path.join(
            cfg.frames_root,
            continent,
            country,
            locality,
            time_of_day,
            vehicle_type,
            video_name
        )

    state = clean_name(state_raw)
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
        log(f"Video already present: {final_path}")
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
        log(f"Download complete: {final_path} ({written} bytes)")
        return final_path

    except Exception as e:
        log(f"Download failed for {final_path}: {e}", "ERROR")
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
        log("Base URL is missing.", "ERROR")
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
        log(f"Already downloaded: {final_path}")
        return final_path, filename, resolution, fps

    log(f"Starting download for '{filename_with_ext}'")

    with requests.Session() as session:
        if username and password:
            session.auth = (username, password)

        session.headers.update({"User-Agent": "multi-fileserver-downloader/1.0"})

        def fetch(url, stream=False):
            try:
                r = session.get(url, timeout=cfg.timeout, params=req_params, stream=stream)
                log(f"GET {url} -> {r.status_code}")
                if r.status_code == 401:
                    log(f"Authentication failed for {url}", "ERROR")
                r.raise_for_status()
                return r
            except requests.RequestException as e:
                log(f"Request failed [{url}]: {e}", "WARNING")
                return None

        # Direct /files paths
        for alias in cfg.aliases:
            direct_url = urljoin(base, f"v/{alias}/files/{filename_with_ext}")
            log(f"Trying direct URL: {direct_url}")

            r = fetch(direct_url, stream=True)
            if r is None:
                continue

            local_path = save_response_to_tmp_then_final(r, final_path)
            if local_path is None:
                return None

            resolution, fps = get_video_metadata(local_path)
            log(f"Saved '{filename_with_ext}' (res={resolution}, fps={fps})")
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
                    log(f"Crawl stopped after {cfg.max_pages} pages.", "WARNING")
                    return None

                resp = fetch(url, stream=False)
                if resp is None:
                    continue

                try:
                    soup = BeautifulSoup(resp.text, "html.parser")
                except Exception as e:
                    log(f"HTML parse failed at {url}: {e}", "WARNING")
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
                            log(f"File located via crawl: {full}")
                            return full

                    if is_dir_link(href):
                        stack.append(full)

            return None

        for alias in cfg.aliases:
            start_url = urljoin(base, f"v/{alias}/browse")
            log(f"Crawling alias: {alias} -> {start_url}")

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
            log(f"Saved '{filename_with_ext}' via crawl (res={resolution}, fps={fps})")
            return local_path, filename, resolution, fps

    log(f"File '{filename_with_ext}' was not found in any alias.", "WARNING")
    return None


def ensure_video_available(video_ref, video_index):
    existing = find_video_path(video_ref, video_index)
    if existing:
        return existing

    result = download_video_from_fileserver(video_ref)
    if not result:
        return None

    local_path = result[0]
    add_video_to_index(local_path, video_index)
    return local_path


# ----------------------------
# Frame extraction
# ----------------------------
def extract_frames_for_ranges(video_path, output_folder, ranges):
    """
    Extract frames only inside the given ranges.

    Output names:
      videoName_time.jpg

    If a frame already exists, it is skipped.
    """
    os.makedirs(output_folder, exist_ok=True)

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
        log(f"No frame times requested for {video_path}", "WARNING")
        return 0

    times_to_extract = []
    skipped_existing = 0

    for current_time in requested_times:
        output_filename = f"{video_name}_{current_time}.jpg"
        output_path = os.path.join(output_folder, output_filename)

        if os.path.exists(output_path):
            skipped_existing += 1
        else:
            times_to_extract.append(current_time)

    if not times_to_extract:
        log(f"All requested frames already exist for {video_name}")
        log(f"Finished {video_name}: saved 0, skipped existing {skipped_existing}")
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Could not open video: {video_path}", "ERROR")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        log(f"Could not determine FPS: {video_path}", "ERROR")
        cap.release()
        return 0

    saved_count = 0

    for current_time in times_to_extract:
        target_frame = int(round(current_time * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ret, frame = cap.read()
        if not ret:
            log(f"Could not read frame at {current_time}s from {video_path}", "WARNING")
            continue

        output_filename = f"{video_name}_{current_time}.jpg"
        output_path = os.path.join(output_folder, output_filename)

        ok = cv2.imwrite(output_path, frame)
        if ok:
            log(f"Saved: {output_path}")
            saved_count += 1
        else:
            log(f"Failed to save: {output_path}", "WARNING")

    cap.release()

    log(f"Finished {video_name}: saved {saved_count}, skipped existing {skipped_existing}")
    return saved_count


# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(cfg.videos_root, exist_ok=True)
    os.makedirs(cfg.frames_root, exist_ok=True)

    video_index = build_video_index(cfg.videos_root)

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
                log(f"Row {row_num}: no videos listed", "WARNING")
                continue

            for i, video_ref in enumerate(videos):
                video_path = ensure_video_available(video_ref, video_index)

                if not video_path:
                    log(f"Row {row_num}: video not found and download failed -> {video_ref}", "WARNING")
                    continue

                start_entry = starts[i] if i < len(starts) else []
                end_entry = ends[i] if i < len(ends) else []
                time_entry = time_of_day_values[i] if i < len(time_of_day_values) else None
                vehicle_entry = vehicle_types[i] if i < len(vehicle_types) else None

                ranges = build_ranges_for_video(start_entry, end_entry)
                if not ranges:
                    log(f"Row {row_num}: no valid ranges for video -> {video_ref}", "WARNING")
                    continue

                actual_video_name = os.path.splitext(os.path.basename(video_path))[0]
                time_folder = map_time_of_day(time_entry)
                vehicle_folder = map_vehicle_type(vehicle_entry)

                output_folder = build_output_folder(
                    continent=continent,
                    country=country,
                    state_raw=state,
                    locality=locality,
                    time_of_day=time_folder,
                    vehicle_type=vehicle_folder,
                    video_name=actual_video_name,
                )

                count = extract_frames_for_ranges(
                    video_path=video_path,
                    output_folder=output_folder,
                    ranges=ranges,
                )

                log(f"Done: {video_path} -> {count} new frames")

    log("All done.")


if __name__ == "__main__":
    main()