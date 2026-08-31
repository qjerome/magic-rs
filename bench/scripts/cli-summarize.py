#!/bin/env python
import re, glob

rows = []
for f in glob.glob("target/benchmarks/cli/wiza_vs_file_*.md"):
    n = int(re.search(r'_(\d+)\.md$', f).group(1))
    text = open(f).read()
    unit_match = re.search(r'Mean \[(\w+)\]', text)
    unit = unit_match.group(1)
    mult = 1000.0 if unit == 's' else 1.0

    wiza_line = re.search(r'\| `[^`]*wiza`\s*\|\s*([\d.]+)\s*±\s*([\d.]+)', text)
    file_line = re.search(r'\| `[^`]*\bfile`\s*\|\s*([\d.]+)\s*±\s*([\d.]+)', text)

    wiza_ms = float(wiza_line.group(1)) * mult
    wiza_err = float(wiza_line.group(2)) * mult
    file_ms = float(file_line.group(1)) * mult
    file_err = float(file_line.group(2)) * mult

    rows.append((n, wiza_ms, wiza_err, file_ms, file_err))

rows.sort()

print(f"| {'files':>6} | {'wiza (ms)':>16} | {'file (ms)':>16} | {'wiza/file':>10} | {'wiza ms/file':>12} | {'file ms/file':>12} | {'faster':>6} |")
print(f"| {'-'*6}-|-{'-'*16}-|-{'-'*16}-|-{'-'*10}-|-{'-'*12}-|-{'-'*12}-|-{'-'*6} |")
for n, w, we, f, fe in rows:
    ratio = w / f
    faster = "wiza" if ratio < 1 else "file"
    print(f"| {n:>6} | {w:>8.1f} ± {we:<5.1f} | {f:>8.1f} ± {fe:<5.1f} | {ratio:>9.2f}x | {w/n:>11.3f}  | {f/n:>11.3f}  | {faster:>6} |")
