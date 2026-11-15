"""Run a Databricks source notebook (.py) on your own computer, without Spark or Java.

It uses sqlframe, which runs PySpark DataFrame code on DuckDB, and swaps the
Databricks-only pieces for local stand-ins:
  - `spark`      -> a sqlframe session backed by a DuckDB file named workspace.duckdb,
                    so tables like `workspace.armd.bronze_cohort` work unchanged
  - `dbutils`    -> only `dbutils.fs.ls` (lists a local folder)
  - `display`    -> prints the first rows as a text table
  - `RAW_DIR`    -> points at the local folder of CSVs
  - `COMORBIDITY_DIR` -> the `comorbidity` folder inside it (notebook 01b)
  - `CREATE VOLUME` statements are skipped (volumes only exist on Databricks)

A notebook that passes here will very likely work on Databricks, but it isn't a
guarantee: sqlframe is a stand-in for Spark, not Spark itself.

Usage (from the repo root):
    python local/run_notebook.py notebooks/01_ingest.py --data ../data_raw --db ../local_db/workspace.duckdb
"""

import argparse
import pathlib
import re
import sys
import time

import duckdb


class _FileInfo:
    def __init__(self, path: pathlib.Path):
        self.path = path.as_posix()
        self.name = path.name
        self.size = path.stat().st_size


class _LocalFS:
    def ls(self, directory):
        return [_FileInfo(p) for p in sorted(pathlib.Path(directory).iterdir())]


class LocalDbutils:
    fs = _LocalFS()


class LocalReader:
    """Translates Spark CSV options that DuckDB's read_csv doesn't accept."""

    def __init__(self, reader):
        self._reader = reader

    def option(self, key, value):
        k = key.lower()
        if k == "multiline":
            return self  # DuckDB handles quoted fields itself
        if k == "inferschema":
            if not value:
                self._reader = self._reader.option("all_varchar", True)  # load every column as text
            return self
        self._reader = self._reader.option(key, value)
        return self

    def __getattr__(self, name):
        return getattr(self._reader, name)


class LocalSpark:
    """Forwards everything to the sqlframe session, except Databricks-only SQL."""

    @property
    def read(self):
        return LocalReader(self._session.read)

    def __init__(self, session, conn):
        self._session = session
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._session, name)

    def sql(self, query, *args, **kwargs):
        if re.match(r"\s*CREATE\s+VOLUME", query, re.IGNORECASE):
            print(f"[local] skipped (Databricks only): {query.strip()}")
            return None
        if re.match(r"\s*CREATE\s+SCHEMA", query, re.IGNORECASE):
            self._conn.execute(query)  # sqlframe doesn't run plain DDL; DuckDB understands this as-is
            return None
        return self._session.sql(query, *args, **kwargs)


def display(obj, max_rows: int = 60, max_width: int = 32):
    if not hasattr(obj, "collect"):
        print(obj)
        return
    columns = obj.columns
    rows = [[("" if v is None else str(v))[:max_width] for v in row] for row in obj.limit(max_rows).collect()]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(columns)]
    line = " | ".join(c.ljust(w) for c, w in zip(columns, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(v.ljust(w) for v, w in zip(r, widths)))
    print(f"({len(rows)} rows shown)")


def notebook_cells(source: str):
    """Yield (cell_number, title, code) for each code cell; markdown-only cells are skipped."""
    title = ""
    for number, cell in enumerate(source.split("# COMMAND ----------"), start=1):
        lines = cell.strip().splitlines()
        markdown = [l[len("# MAGIC"):].strip() for l in lines if l.startswith("# MAGIC")]
        headings = [m.lstrip("#").strip() for m in markdown if m.startswith("#")]
        if headings:
            title = headings[0]
        code = [l for l in lines if not l.startswith("# MAGIC") and l.strip() != "# Databricks notebook source"]
        if any(l.strip() for l in code):
            yield number, title, "\n".join(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook", help="path to a Databricks .py notebook")
    parser.add_argument("--data", required=True, help="folder holding the ARMD CSV files")
    parser.add_argument("--db", required=True, help="DuckDB file to store tables in; must be named workspace.duckdb")
    parser.add_argument("--no-pandas", action="store_true",
                        help="don't load pandas (use when Windows blocks it)")
    args = parser.parse_args()

    db_path = pathlib.Path(args.db).resolve()
    if db_path.stem != "workspace":
        sys.exit("--db file must be named workspace.duckdb so table names match the notebooks")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = pathlib.Path(args.data).resolve()

    if args.no_pandas:
        # Windows Smart App Control can block pandas' compiled files. sqlframe only uses pandas
        # optionally, but a half-loaded pandas crashes it, so mark pandas as unavailable up front.
        sys.modules["pandas"] = None

    conn = duckdb.connect(str(db_path))
    from sqlframe import activate
    activate(engine="duckdb", conn=conn)
    from pyspark.sql import SparkSession  # resolved to sqlframe by activate()
    from pyspark.sql import functions as F
    if not hasattr(F, "broadcast"):
        F.broadcast = lambda df: df  # a Spark join-speed hint; meaningless on DuckDB

    namespace = {
        "__name__": "__main__",
        "spark": LocalSpark(SparkSession.builder.getOrCreate(), conn),
        "dbutils": LocalDbutils(),
        "display": display,
    }

    notebook = pathlib.Path(args.notebook)
    source = notebook.read_text(encoding="utf-8")
    started = time.time()
    for number, title, code in notebook_cells(source):
        code = re.sub(r"^RAW_DIR\s*=.*$", f"RAW_DIR = {data_dir.as_posix()!r}", code, flags=re.MULTILINE)
        code = re.sub(r"^COMORBIDITY_DIR\s*=.*$", f"COMORBIDITY_DIR = {(data_dir / 'comorbidity').as_posix()!r}",
                      code, flags=re.MULTILINE)
        print(f"\n===== cell {number}: {title} =====", flush=True)
        cell_start = time.time()
        exec(compile(code, f"{notebook.name} [cell {number}]", "exec"), namespace)
        print(f"[{time.time() - cell_start:.1f}s]", flush=True)
    print(f"\nFinished {notebook.name} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
