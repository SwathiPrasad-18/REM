from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import hashlib
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "Raw_Reports"
OUTPUT_DIR = BASE_DIR / "Extracted_Data"

JSON_DIR = OUTPUT_DIR / "individual_reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REGEX PATTERNS
# ============================================================

SHA256_PATTERN = re.compile(
    r"\b[a-fA-F0-9]{64}\b"
)

MD5_PATTERN = re.compile(
    r"\b[a-fA-F0-9]{32}\b"
)

SHA1_PATTERN = re.compile(
    r"\b[a-fA-F0-9]{40}\b"
)

IP_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|"
    r"2[0-4][0-9]|"
    r"1?[0-9]{1,2})\.){3}"
    r"(?:25[0-5]|"
    r"2[0-4][0-9]|"
    r"1?[0-9]{1,2})\b"
)

MITRE_PATTERN = re.compile(
    r"\bT\d{4}(?:\.\d{3})?\b",
    re.IGNORECASE
)

URL_PATTERN = re.compile(
    r"https?://[^\s\"'<>\]]+",
    re.IGNORECASE
)

DOMAIN_PATTERN = re.compile(
    r"\b(?:[a-zA-Z0-9-]+\.)+"
    r"[a-zA-Z]{2,}\b"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_text(value):
    """
    Normalize whitespace.
    """

    if value is None:
        return ""

    value = str(value)

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def unique_list(values):
    """
    Remove duplicates while preserving order.
    """

    result = []

    seen = set()

    for value in values:

        value = clean_text(value)

        if not value:
            continue

        key = value.lower()

        if key not in seen:

            seen.add(key)

            result.append(value)

    return result


def read_html(file_path):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1"
    ]

    for encoding in encodings:

        try:

            with open(
                file_path,
                "r",
                encoding=encoding,
                errors="ignore"
            ) as file:

                return file.read()

        except Exception:
            continue

    return ""


def sha256_file(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# HTML PROCESSING
# ============================================================

def parse_html(html):

    soup = BeautifulSoup(
        html,
        "lxml"
    )

    # Remove scripts and styles
    for tag in soup(
        ["script", "style", "noscript"]
    ):

        tag.decompose()

    text = soup.get_text(
        " ",
        strip=True
    )

    text = clean_text(text)

    return soup, text


# ============================================================
# HASH EXTRACTION
# ============================================================

def extract_hashes(text):

    sha256_values = unique_list(
        SHA256_PATTERN.findall(text)
    )

    sha1_values = unique_list(
        SHA1_PATTERN.findall(text)
    )

    md5_values = unique_list(
        MD5_PATTERN.findall(text)
    )

    return {
        "sha256": sha256_values[0]
        if sha256_values else None,

        "sha1": sha1_values[0]
        if sha1_values else None,

        "md5": md5_values[0]
        if md5_values else None
    }


# ============================================================
# SANDBOX DETECTION
# ============================================================

def detect_sandboxes(text):

    text_lower = text.lower()

    detected = []

    sandbox_names = {
        "Zenbox": [
            "zenbox"
        ],

        "CAPE": [
            "cape sandbox",
            "capEv2",
            "cape"
        ],

        "Jujubox": [
            "jujubox"
        ],

        "CAPA": [
            "capa"
        ],

        "Sysinternals": [
            "sysinternals"
        ],

        "VMRay": [
            "vmray"
        ],

        "C2AE": [
            "c2ae"
        ]
    }

    for sandbox, patterns in sandbox_names.items():

        for pattern in patterns:

            if pattern.lower() in text_lower:

                detected.append(sandbox)

                break

    return unique_list(detected)


# ============================================================
# BASIC FILE INFORMATION
# ============================================================

def extract_filename(soup, text):

    patterns = [
        r"filename\s*[:\-]\s*([^\s]+)",
        r"file name\s*[:\-]\s*([^\s]+)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            return clean_text(
                match.group(1)
            )

    # HTML title fallback
    if soup.title:

        title = clean_text(
            soup.title.get_text()
        )

        if title:

            return title

    return None


def extract_file_type(text):

    patterns = [
        r"file type\s*[:\-]\s*(.{1,150}?)(?:\s{2,}|file size|$)",
        r"type\s*[:\-]\s*(.{1,150}?)(?:\s{2,}|size|$)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            return clean_text(
                match.group(1)
            )

    return None


def extract_file_size(text):

    pattern = (
        r"(?:file size|size)"
        r"\s*[:\-]\s*"
        r"([\d.,]+\s*(?:KB|MB|GB|bytes?|B))"
    )

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:

        return clean_text(
            match.group(1)
        )

    return None


# ============================================================
# MITRE ATT&CK
# ============================================================

def extract_mitre(text):

    techniques = MITRE_PATTERN.findall(text)

    techniques = [
        technique.upper()
        for technique in techniques
    ]

    return unique_list(
        techniques
    )


# ============================================================
# NETWORK INFORMATION
# ============================================================

def extract_ips(text):

    ips = IP_PATTERN.findall(text)

    return unique_list(
        ips
    )


def extract_urls(text):

    urls = URL_PATTERN.findall(text)

    cleaned = []

    for url in urls:

        url = url.rstrip(
            ".,;:)]}>"
        )

        cleaned.append(url)

    return unique_list(
        cleaned
    )


def extract_domains(text):

    domains = DOMAIN_PATTERN.findall(text)

    ignored = {
        "windows.com",
        "microsoft.com",
        "google.com",
        "virustotal.com",
        "example.com"
    }

    result = []

    for domain in domains:

        if domain.lower() in ignored:
            continue

        result.append(domain)

    return unique_list(
        result
    )


# ============================================================
# BEHAVIOR TAGS
# ============================================================

def extract_behavior_tags(text):

    possible_tags = [
        "idle",
        "detect-debug-environment",
        "anti-debugging",
        "anti-vm",
        "evasion",
        "persistence",
        "ransomware",
        "keylogging",
        "credential-access",
        "network",
        "download",
        "upload",
        "process-injection",
        "powershell",
        "command-execution",
        "registry",
        "file-modification",
        "self-delete"
    ]

    text_lower = text.lower()

    found = []

    for tag in possible_tags:

        if tag.lower() in text_lower:

            found.append(tag)

    return unique_list(
        found
    )


# ============================================================
# CAPA CAPABILITIES
# ============================================================

def extract_capa(text):

    capa_keywords = [
        "process",
        "network",
        "registry",
        "file",
        "memory",
        "execute",
        "powershell",
        "command",
        "inject",
        "persistence",
        "download",
        "create",
        "delete"
    ]

    text_lower = text.lower()

    found = []

    for keyword in capa_keywords:

        if keyword in text_lower:

            found.append(keyword)

    return unique_list(
        found
    )


# ============================================================
# PROCESS EXTRACTION
# ============================================================

def extract_processes(text):

    processes = []

    process_extensions = [
        ".exe",
        ".dll",
        ".sys"
    ]

    # Candidate process-like strings
    pattern = (
        r"\b[A-Za-z0-9_\-.]+"
        r"\.(?:exe|dll|sys)\b"
    )

    matches = re.findall(
        pattern,
        text,
        re.IGNORECASE
    )

    for match in matches:

        lower = match.lower()

        # Avoid obvious noise
        if lower in [
            "index.exe",
            "setup.exe"
        ]:
            continue

        processes.append(match)

    return unique_list(
        processes
    )


# ============================================================
# COMMAND EXECUTION
# ============================================================

def extract_commands(text):

    commands = []

    command_patterns = [
        r"cmd\.exe",
        r"powershell\.exe",
        r"wscript\.exe",
        r"cscript\.exe",
        r"rundll32\.exe",
        r"regsvr32\.exe",
        r"mshta\.exe",
        r"certutil\.exe",
        r"bitsadmin\.exe"
    ]

    for pattern in command_patterns:

        matches = re.findall(
            pattern,
            text,
            re.IGNORECASE
        )

        commands.extend(
            matches
        )

    return unique_list(
        commands
    )


# ============================================================
# REGISTRY INDICATORS
# ============================================================

def extract_registry(text):

    registry_patterns = [
        r"HKLM\\[^\s<\"]+",
        r"HKCU\\[^\s<\"]+",
        r"HKCR\\[^\s<\"]+",
        r"HKEY_LOCAL_MACHINE\\[^\s<\"]+",
        r"HKEY_CURRENT_USER\\[^\s<\"]+",
        r"HKEY_CLASSES_ROOT\\[^\s<\"]+"
    ]

    registry = []

    for pattern in registry_patterns:

        matches = re.findall(
            pattern,
            text,
            re.IGNORECASE
        )

        registry.extend(
            matches
        )

    return unique_list(
        registry
    )


# ============================================================
# FILE ACTIVITY
# ============================================================

def extract_file_activity(text):

    activity = {
        "opened": [],
        "written": [],
        "deleted": [],
        "created": [],
        "dropped": []
    }

    patterns = {

        "opened": [
            r"files?\s+opened",
            r"opened\s+files?"
        ],

        "written": [
            r"files?\s+written",
            r"written\s+files?"
        ],

        "deleted": [
            r"files?\s+deleted",
            r"deleted\s+files?"
        ],

        "created": [
            r"files?\s+created",
            r"created\s+files?"
        ],

        "dropped": [
            r"dropped\s+files?"
        ]
    }

    text_lower = text.lower()

    for category, regexes in patterns.items():

        for regex in regexes:

            if re.search(
                regex,
                text_lower
            ):

                activity[category].append(
                    regex
                )

    for category in activity:

        activity[category] = unique_list(
            activity[category]
        )

    return activity


# ============================================================
# VERDICT EXTRACTION
# ============================================================

def extract_verdict(text):

    patterns = [
        r"verdict\s*[:\-]\s*(.{1,100})",
        r"(\d+)\s*/\s*100\s*(?:Non Malicious|Malicious)",
        r"(Non Malicious|Malicious)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            return clean_text(
                match.group(0)
            )

    return None


# ============================================================
# REPORT EXTRACTION
# ============================================================

def extract_report(file_path):

    html = read_html(
        file_path
    )

    soup, text = parse_html(
        html
    )

    hashes = extract_hashes(
        text
    )

    sandboxes = detect_sandboxes(
        text
    )

    processes = extract_processes(
        text
    )

    commands = extract_commands(
        text
    )

    registry = extract_registry(
        text
    )

    ips = extract_ips(
        text
    )

    urls = extract_urls(
        text
    )

    domains = extract_domains(
        text
    )

    mitre = extract_mitre(
        text
    )

    behavior_tags = extract_behavior_tags(
        text
    )

    capa = extract_capa(
        text
    )

    file_activity = extract_file_activity(
        text
    )

    result = {

        "source_file": file_path.name,

        "source_path": str(
            file_path
        ),

        "report_sha256": sha256_file(
            file_path
        ),

        "sample_sha256": hashes[
            "sha256"
        ],

        "sample_sha1": hashes[
            "sha1"
        ],

        "sample_md5": hashes[
            "md5"
        ],

        "filename": extract_filename(
            soup,
            text
        ),

        "file_type": extract_file_type(
            text
        ),

        "file_size": extract_file_size(
            text
        ),

        "sandboxes": sandboxes,

        "processes": processes,

        "commands": commands,

        "registry": registry,

        "file_activity": file_activity,

        "network": {

            "ip_addresses": ips,

            "domains": domains,

            "urls": urls

        },

        "mitre_techniques": mitre,

        "behavior_tags": behavior_tags,

        "capa_capabilities": capa,

        "verdict": extract_verdict(
            text
        ),

        "extraction_timestamp":
            datetime.now().isoformat()

    }

    return result


# ============================================================
# FLATTEN REPORT INTO EVENTS
# ============================================================

def create_events(report):

    events = []

    sha256 = report.get(
        "sample_sha256"
    )

    sandboxes = report.get(
        "sandboxes",
        []
    )

    sandbox = (
        "|".join(sandboxes)
        if sandboxes
        else "Unknown"
    )

    # MITRE
    for technique in report.get(
        "mitre_techniques",
        []
    ):

        events.append({

            "sha256": sha256,

            "sandbox": sandbox,

            "event_type": "MITRE",

            "event_value": technique,

            "category": "ATT&CK"

        })

    # Processes
    for process in report.get(
        "processes",
        []
    ):

        events.append({

            "sha256": sha256,

            "sandbox": sandbox,

            "event_type": "PROCESS",

            "event_value": process,

            "category": "Process"

        })

    # Commands
    for command in report.get(
        "commands",
        []
    ):

        events.append({

            "sha256": sha256,

            "sandbox": sandbox,

            "event_type": "COMMAND",

            "event_value": command,

            "category": "Execution"

        })

    # Registry
    for key in report.get(
        "registry",
        []
    ):

        events.append({

            "sha256": sha256,

            "sandbox": sandbox,

            "event_type": "REGISTRY",

            "event_value": key,

            "category": "Registry"

        })

    # IP
    network = report.get(
        "network",
        {}
    )

    for ip in network.get(
        "ip_addresses",
        []
    ):

        events.append({

            "sha256": sha256,

            "sandbox": sandbox,

            "event_type": "IP",

            "event_value": ip,

            "category": "Network"

        })

    # Domains
    for domain in network.get(
        "domains",
        []
    ):

        events.append({

            "sha256": sha256,

            "sandbox": sandbox,

            "event_type": "DOMAIN",

            "event_value": domain,

            "category": "Network"

        })

    # URLs
    for url in network.get(
        "urls",
        []
    ):

        events.append({

            "sha256": sha256,

            "sandbox": sandbox,

            "event_type": "URL",

            "event_value": url,

            "category": "Network"

        })

    # Behavior tags
    for tag in report.get(
        "behavior_tags",
        []
    ):

        events.append({

            "sha256": sha256,

            "sandbox": sandbox,

            "event_type": "BEHAVIOR_TAG",

            "event_value": tag,

            "category": "Behavior"

        })

    # CAPA
    for capability in report.get(
        "capa_capabilities",
        []
    ):

        events.append({

            "sha256": sha256,

            "sandbox": sandbox,

            "event_type": "CAPA",

            "event_value": capability,

            "category": "Capability"

        })

    return events


# ============================================================
# CREATE ML FEATURES
# ============================================================

def create_feature_row(report):

    network = report.get(
        "network",
        {}
    )

    file_activity = report.get(
        "file_activity",
        {}
    )

    processes = report.get(
        "processes",
        []
    )

    commands = report.get(
        "commands",
        []
    )

    registry = report.get(
        "registry",
        []
    )

    mitre = report.get(
        "mitre_techniques",
        []
    )

    behavior_tags = report.get(
        "behavior_tags",
        []
    )

    capa = report.get(
        "capa_capabilities",
        []
    )

    tags_lower = [
        x.lower()
        for x in behavior_tags
    ]

    row = {

        "sha256":
            report.get(
                "sample_sha256"
            ),

        "filename":
            report.get(
                "filename"
            ),

        "sandbox_count":
            len(
                report.get(
                    "sandboxes",
                    []
                )
            ),

        "process_count":
            len(processes),

        "command_count":
            len(commands),

        "registry_count":
            len(registry),

        "network_ip_count":
            len(
                network.get(
                    "ip_addresses",
                    []
                )
            ),

        "domain_count":
            len(
                network.get(
                    "domains",
                    []
                )
            ),

        "url_count":
            len(
                network.get(
                    "urls",
                    []
                )
            ),

        "mitre_count":
            len(mitre),

        "capa_count":
            len(capa),

        "behavior_tag_count":
            len(behavior_tags),

        "file_opened_indicator":
            int(
                bool(
                    file_activity.get(
                        "opened"
                    )
                )
            ),

        "file_written_indicator":
            int(
                bool(
                    file_activity.get(
                        "written"
                    )
                )
            ),

        "file_deleted_indicator":
            int(
                bool(
                    file_activity.get(
                        "deleted"
                    )
                )
            ),

        "file_created_indicator":
            int(
                bool(
                    file_activity.get(
                        "created"
                    )
                )
            ),

        "file_dropped_indicator":
            int(
                bool(
                    file_activity.get(
                        "dropped"
                    )
                )
            ),

        "debug_evasion":
            int(
                any(
                    "debug" in tag
                    for tag in tags_lower
                )
            ),

        "powershell_activity":
            int(
                "powershell" in tags_lower
                or any(
                    "powershell" in x.lower()
                    for x in commands
                )
            ),

        "network_activity":
            int(
                bool(
                    network.get(
                        "ip_addresses"
                    )
                    or network.get(
                        "domains"
                    )
                    or network.get(
                        "urls"
                    )
                )
            ),

        "registry_activity":
            int(
                bool(registry)
            ),

        "mitre_activity":
            int(
                bool(mitre)
            ),

        "capa_activity":
            int(
                bool(capa)
            )

    }

    return row


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "REM - VirusTotal Behavioral Report Extractor"
    )

    print("=" * 70)

    print()

    if not RAW_DIR.exists():

        print(
            "ERROR: Raw_Reports folder not found."
        )

        print(
            "Expected:"
        )

        print(
            RAW_DIR
        )

        return

    html_files = list(
        RAW_DIR.rglob("*.html")
    )

    html_files += list(
        RAW_DIR.rglob("*.htm")
    )

    html_files = sorted(
        set(html_files)
    )

    print(
        f"Reports found: {len(html_files)}"
    )

    print()

    if not html_files:

        print(
            "No HTML reports found."
        )

        return

    all_reports = []

    all_events = []

    feature_rows = []

    extraction_logs = []

    for file_path in tqdm(
        html_files,
        desc="Extracting reports"
    ):

        try:

            report = extract_report(
                file_path
            )

            all_reports.append(
                report
            )

            events = create_events(
                report
            )

            all_events.extend(
                events
            )

            feature_row = create_feature_row(
                report
            )

            feature_rows.append(
                feature_row
            )

            # Save individual JSON
            sha = report.get(
                "sample_sha256"
            )

            if not sha:

                sha = file_path.stem

            sandbox = (
                "_".join(
                    report.get(
                        "sandboxes",
                        []
                    )
                )
                or "Unknown"
            )

            json_name = (
                f"{sha}_{sandbox}.json"
            )

            json_path = (
                JSON_DIR /
                json_name
            )

            with open(
                json_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    report,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            extraction_logs.append({

                "file":
                    str(file_path),

                "status":
                    "SUCCESS",

                "sha256":
                    sha,

                "sandbox":
                    sandbox,

                "error":
                    ""

            })

        except Exception as error:

            extraction_logs.append({

                "file":
                    str(file_path),

                "status":
                    "ERROR",

                "sha256":
                    "",

                "sandbox":
                    "",

                "error":
                    str(error)

            })

            print(
                f"\nERROR processing {file_path}: {error}"
            )

    # ========================================================
    # SAVE COMPLETE JSON
    # ========================================================

    with open(
        OUTPUT_DIR / "all_reports.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_reports,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # SAVE EVENTS
    # ========================================================

    events_df = pd.DataFrame(
        all_events
    )

    events_df.to_csv(
        OUTPUT_DIR /
        "behavior_events.csv",
        index=False
    )

    # ========================================================
    # SAVE FEATURES
    # ========================================================

    features_df = pd.DataFrame(
        feature_rows
    )

    features_df.to_csv(
        OUTPUT_DIR /
        "sample_features.csv",
        index=False
    )

    # ========================================================
    # SAVE LOG
    # ========================================================

    log_df = pd.DataFrame(
        extraction_logs
    )

    log_df.to_csv(
        OUTPUT_DIR /
        "extraction_log.csv",
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    successful = len(
        [
            x
            for x in extraction_logs
            if x["status"] == "SUCCESS"
        ]
    )

    failed = len(
        [
            x
            for x in extraction_logs
            if x["status"] == "ERROR"
        ]
    )

    print()

    print("=" * 70)

    print(
        "EXTRACTION COMPLETED"
    )

    print("=" * 70)

    print(
        f"Reports found      : {len(html_files)}"
    )

    print(
        f"Successfully parsed: {successful}"
    )

    print(
        f"Failed             : {failed}"
    )

    print()

    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Generated:"
    )

    print(
        "  all_reports.json"
    )

    print(
        "  behavior_events.csv"
    )

    print(
        "  sample_features.csv"
    )

    print(
        "  extraction_log.csv"
    )

    print(
        "  individual_reports/*.json"
    )

    print()

    print(
        "REM extraction finished."
    )


if __name__ == "__main__":

    main()