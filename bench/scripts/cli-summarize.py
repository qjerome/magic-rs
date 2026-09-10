#!/bin/env python
import re, glob

rows = []
for f in glob.glob("target/benchmarks/cli/wiza_vs_file_*.md"):
    n = int(re.search(r'_(\d+)\.md$', f).group(1))
    text = open(f).read()
    unit_match = re.search(r'Mean \[(\w+)\]', text)
    unit = unit_match.group(1)
    mult = 1000.0 if unit == 's' else 1.0

    wiza_line = re.search(r'\| `[^`]*wiza.*?`\s*\|\s*([\d.]+)\s*±\s*([\d.]+)', text)
    file_line = re.search(r'\| `[^`]*\bfile`\s*\|\s*([\d.]+)\s*±\s*([\d.]+)', text)

    wiza_ms = float(wiza_line.group(1)) * mult
    wiza_err = float(wiza_line.group(2)) * mult
    file_ms = float(file_line.group(1)) * mult
    file_err = float(file_line.group(2)) * mult

    rows.append((n, wiza_ms, wiza_err, file_ms, file_err))

rows.sort()

print(f"| {'files':>6} | {'wiza (ms)':>16} | {'file (ms)':>16} | {'file/wiza':>10} | {'wiza ms/file':>12} | {'file ms/file':>12} | {'faster':>6} |")
print(f"| {'-'*6}-|-{'-'*16}-|-{'-'*16}-|-{'-'*10}-|-{'-'*12}-|-{'-'*12}-|-{'-'*6} |")
for n, w, we, f, fe in rows:
    ratio = f / w
    faster = "wiza" if ratio > 1 else "file"

    w_str = f"{w:.1f} ± {we:<5.1f}".strip()
    f_str = f"{f:.1f} ± {fe:<5.1f}".strip()
    w_per_file = f"{w/n:.3f}".strip()
    f_per_file = f"{f/n:.3f}".strip()
    if faster == "wiza":
        w_str = f"**{w_str}**"
        w_per_file = f"**{w_per_file}**"
    else:
        f_str = f"**{f_str}**"
        f_per_file = f"**{f_per_file}**"

    print(f"| {n:>6} | {w_str:>16} | {f_str:>16} | {ratio:>9.2f}x | {w_per_file:>12} | {f_per_file:>12} | {faster:>6} |")
