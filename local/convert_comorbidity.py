"""Shrink the comorbidity CSV so it can be uploaded to Databricks.

microbiology_cultures_comorbidity.csv is ~19.7 GB, and the Databricks upload page takes at most
5 GB per file. This script rewrites it as compressed Parquet files of about 1 GB or less each.

Nothing is filtered or changed: every row and column is kept, and every value stays text. The one
clean-up is the same as notebook 01's: the text "null" (any case) or an empty cell becomes a real
missing value.

Usage (from the flemingrx folder, with the .venv active):
    python local/convert_comorbidity.py <path to microbiology_cultures_comorbidity.csv> <empty output folder>

Then upload every .parquet file in the output folder to the volume folder
/Volumes/workspace/armd/raw/comorbidity/ and run notebook 01b.
"""
import argparse
import pathlib
import time

import duckdb


def sql_text(value) -> str:
    """A string literal for SQL, with any single quotes doubled."""
    return "'" + str(value).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=pathlib.Path, help="microbiology_cultures_comorbidity.csv")
    parser.add_argument("out_dir", type=pathlib.Path, help="new or empty folder for the Parquet files")
    parser.add_argument("--max-file-size", default="900MB", help="largest Parquet file to write (default 900MB)")
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Not found: {args.csv}")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"Output folder is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.out_dir.parent / f"{args.out_dir.name}_duckdb_tmp"

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")        # lets DuckDB stream instead of buffer
    con.execute(f"SET temp_directory = {sql_text(tmp_dir)}")    # spill to disk if memory runs low

    source = (
        f"read_csv({sql_text(args.csv)}, header = true, all_varchar = true, "
        f"quote = '\"', escape = '\"', nullstr = ['null', 'Null', 'NULL', ''])"
    )
    columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]
    print("Columns:", ", ".join(columns))

    print(f"Converting {args.csv.name} ({args.csv.stat().st_size / 1e9:.1f} GB). This can take 10-30 minutes...")
    start = time.time()
    rows = con.execute(
        f"COPY (SELECT * FROM {source}) TO {sql_text(args.out_dir)} "
        f"(FORMAT parquet, COMPRESSION zstd, FILE_SIZE_BYTES {sql_text(args.max_file_size)}, "
        f"FILENAME_PATTERN 'comorbidity_{{i}}')"
    ).fetchone()[0]

    files = sorted(args.out_dir.glob("*.parquet"))
    total = sum(f.stat().st_size for f in files)
    print(f"Done in {(time.time() - start) / 60:.1f} min: {rows:,} rows in {len(files)} files, "
          f"{total / 1e9:.2f} GB in total (largest {max(f.stat().st_size for f in files) / 1e9:.2f} GB)")
    print(f"Write this row count down; notebook 01b checks against it: {rows:,}")


if __name__ == "__main__":
    main()
