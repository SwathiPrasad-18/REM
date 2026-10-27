
from pathlib import Path
from bs4 import BeautifulSoup
import csv

base = Path(
    "../REM_Datasets/02_VirusTotal_Reports/Raw_Reports"
)

error_file = Path(
    "../REM_Datasets/02_VirusTotal_Reports/Extracted_Data/parse_errors.csv"
)

with error_file.open(encoding="utf-8-sig", newline="") as f:
    errors = list(csv.DictReader(f))

for item in errors:
    path = Path(item["file"])

    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        print(f"\nFILE NOT FOUND: {item['file']}")
        continue

    try:
        soup = BeautifulSoup(
            path.read_text(encoding="utf-8", errors="ignore"),
            "html.parser"
        )

        title = soup.title.get_text(" ", strip=True) if soup.title else "No title"
        text = " ".join(soup.get_text(" ", strip=True).split())

        print("\n" + "=" * 70)
        print("FILE:", path.name)
        print("TITLE:", title)
        print("CONTENT:", text[:500])

    except Exception as e:
        print("ERROR:", path.name, e)
