"""Step 1b of the pipeline (see README.md): converts the EthoVision / Observer
Excel exports in each group folder into the CSVs the rest of the pipeline
reads. Replaces the Sep 14 one-off scripts (convert_ethovision.py,
regenerate_csvs.py, the conversion half of convert_sort_behavior.py) and
convert_opto_xlsx.py, and writes byte-for-byte the same format as
regenerate_csvs.py did.

Each .xlsx is identified by its CONTENTS, not its name (cohort 2's tracking
and scoring exports can have identical file names):

  kind              how it's recognized                    -> output (same folder)
  ----------------  -------------------------------------  --------------------------------
  tracking          sheets "Track-Arena 1-Subject 1/2"     <stem>_subject1.csv, _subject2.csv
  ethovision score  sheet  "Manual Scoring-Arena 1"        <stem>.csv
  observer score    first row has an "Event_Type" column   <stem>.csv, with "Observation N"
                                                           renamed to "Trial N" (opto)

Every sheet is written verbatim, header lines included, with every row padded
to the sheet's widest row. Uneven rows break pd.read_csv(header=35), which is
how the pipeline reads these files.

This script never moves or deletes files. Sorting trials into blinded group
folders is a separate step.

Usage:

    python3 convert_xlsx_to_csv.py              # CHECK (default, writes nothing):
                                                #   convert in memory, compare to the CSVs on disk
    python3 convert_xlsx_to_csv.py --write      # write CSVs that are missing or different
    python3 convert_xlsx_to_csv.py --write Updated-C2/B/tracking     # only these folders
"""

import argparse
import csv
import glob
import io
import re
from pathlib import Path

import openpyxl

from pipeline_paths import ETHO_ROOT

# Folders (relative to data/Ethovision) holding the pipeline's source .xlsx files.
SOURCE_FOLDERS = [
    "Updated-C1/A/tracking",
    "Updated-C1/A/scoring",
    "Updated-C1/C/tracking",
    "Updated-C1/C/scoring",
    "Updated-C2/B/tracking",
    "Updated-C2/B/scoring",
    "Opto/opto_etho_tracking",
    "Opto/opto-scored",
]


def classify(wb):
    """Returns 'tracking', 'ethovision_scoring', 'observer_scoring' or None."""
    sheets = wb.sheetnames
    if len(sheets) == 2 and all(s.startswith("Track-") for s in sheets):
        return "tracking"
    if len(sheets) == 1 and sheets[0].startswith("Manual Scoring"):
        return "ethovision_scoring"
    first_row = next(wb[sheets[0]].iter_rows(min_row=1, max_row=1, values_only=True), ())
    if "Event_Type" in first_row:
        return "observer_scoring"
    return None


def sheet_to_csv_text(ws):
    """Same output as the Sep 14 regenerate_csvs.py: openpyxl cell values,
    empty cells as "", every row padded to the widest row."""
    rows = [["" if v is None else v for v in row] for row in ws.iter_rows(values_only=True)]
    max_col = max(len(row) for row in rows)
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    for row in rows:
        writer.writerow(row + [""] * (max_col - len(row)))
    return buf.getvalue()


def planned_outputs(xlsx_path, kind, wb):
    """[(output_path, worksheet)] for one source file."""
    if kind == "tracking":
        return [(xlsx_path.with_name(f"{xlsx_path.stem}_subject{i}.csv"), wb[name])
                for i, name in enumerate(wb.sheetnames, start=1)]

    stem = xlsx_path.stem
    if kind == "observer_scoring":
        stem = re.sub(r"Observation\s*(\d+)", r"Trial \1", stem)
    return [(xlsx_path.with_name(f"{stem}.csv"), wb[wb.sheetnames[0]])]


def main(folders, write):
    counts = {"identical": 0, "different": 0, "missing": 0}
    written = 0
    seen_outputs = {}

    for folder in folders:
        xlsx_files = sorted(glob.glob(str(ETHO_ROOT / folder / "*.xlsx")))
        print(f"\n== {folder}  ({len(xlsx_files)} xlsx)")

        for xlsx_file in xlsx_files:
            xlsx_path = Path(xlsx_file)
            wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
            kind = classify(wb)
            if kind is None:
                print(f"  SKIP (unrecognized contents: sheets {wb.sheetnames})  {xlsx_path.name!r}")
                wb.close()
                continue

            for out_path, ws in planned_outputs(xlsx_path, kind, wb):
                if out_path in seen_outputs:
                    raise RuntimeError(f"{xlsx_path.name!r} and {seen_outputs[out_path]!r} both map to {out_path.name!r}")
                seen_outputs[out_path] = xlsx_path.name

                text = sheet_to_csv_text(ws)
                if not out_path.exists():
                    status = "missing"
                elif out_path.read_bytes() == text.encode("utf-8"):
                    status = "identical"
                else:
                    status = "different"
                counts[status] += 1

                action = ""
                if write and status != "identical":
                    with open(out_path, "w", encoding="utf-8", newline="") as f:   # newline="": keep the CSV's own line endings
                        f.write(text)
                    written += 1
                    action = "  -> written"
                print(f"  {status:9s} {kind:18s} {xlsx_path.name!r} -> {out_path.name!r}{action}")
            wb.close()

    print(f"\nsummary: {counts}")
    print(f"wrote {written} CSV(s)" if write else "check only: nothing written (pass --write to write missing/different CSVs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="*", default=SOURCE_FOLDERS,
                        help="folders under data/Ethovision to convert (default: all pipeline groups)")
    parser.add_argument("--write", action="store_true",
                        help="write CSVs that are missing or differ (default: check only)")
    args = parser.parse_args()
    main(args.folders, args.write)
