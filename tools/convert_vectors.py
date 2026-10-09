"""Convert an existing book.db's embeddings from float32 to int8 in place
(or vice versa), then VACUUM to reclaim disk space.

Usage: python tools/convert_vectors.py <db_path> [--to i8|f32]
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.db import quantize_vector, dequantize_vector
from app.log import get_logger

logger = get_logger("index")


def _ensure_columns(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(embeddings)")}
    if "dtype" not in cols:
        conn.execute("ALTER TABLE embeddings ADD COLUMN dtype TEXT DEFAULT 'f32'")
    if "scale" not in cols:
        conn.execute("ALTER TABLE embeddings ADD COLUMN scale REAL")
    book_cols = {row[1] for row in conn.execute("PRAGMA table_info(books)")}
    if "vector_dtype" not in book_cols:
        conn.execute("ALTER TABLE books ADD COLUMN vector_dtype TEXT DEFAULT 'f32'")
    conn.commit()


def convert(db_path: str, to: str = "i8") -> dict:
    import struct

    import numpy as np

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_columns(conn)

    size_before = Path(db_path).stat().st_size

    rows = conn.execute(
        "SELECT rowid, para_id, lang, model, vector, COALESCE(dtype,'f32') AS dtype, scale FROM embeddings"
    ).fetchall()

    n_converted = 0
    for r in rows:
        cur_dtype = r["dtype"]
        if cur_dtype == to:
            continue
        if cur_dtype == "f32":
            n = len(r["vector"]) // 4
            vec = np.array(struct.unpack(f"<{n}f", r["vector"]), dtype=np.float32)
        else:  # i8 -> f32
            vec = dequantize_vector(r["vector"], r["scale"] or 1.0)

        if to == "i8":
            blob, scale = quantize_vector(vec)
        else:
            blob = struct.pack(f"<{len(vec)}f", *[float(x) for x in vec])
            scale = None

        conn.execute(
            "UPDATE embeddings SET vector = ?, dtype = ?, scale = ? WHERE rowid = ?",
            (blob, to, scale, r["rowid"]),
        )
        n_converted += 1

    conn.execute("UPDATE books SET vector_dtype = ?", (to,))
    conn.commit()
    conn.execute("VACUUM")
    conn.commit()
    conn.close()

    size_after = Path(db_path).stat().st_size
    result = {
        "db_path": db_path,
        "to": to,
        "n_converted": n_converted,
        "n_total": len(rows),
        "size_before_bytes": size_before,
        "size_after_bytes": size_after,
    }
    logger.info(
        "convert_vectors(%s -> %s): %d/%d rows, %.1f MB -> %.1f MB",
        db_path, to, n_converted, len(rows), size_before / 1e6, size_after / 1e6,
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db_path")
    ap.add_argument("--to", choices=["i8", "f32"], default="i8")
    args = ap.parse_args()
    result = convert(args.db_path, args.to)
    import json

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
