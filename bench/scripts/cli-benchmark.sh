#!/bin/bash

set -e

cargo build -r --bin wiza
git_root=$(git rev-parse --show-toplevel)
root=$(realpath --relative-to="$PWD" "$git_root")
bench_out="$root/target/benchmarks/cli"
corpus_dir="$bench_out/corpus"

mkdir -p $corpus_dir

find $bench_out -type f -delete

find ~ -maxdepth 5 -type f -print0 2>/dev/null | shuf -z > $corpus_dir/corpus.txt

for n in 50 100 250 500 1000 2000 4000 8000
do
    corpus="$corpus_dir/corpus_$n.txt"
    head -z -n $n $corpus_dir/corpus.txt > $corpus

    hyperfine \
    --warmup 3 \
    --export-markdown $bench_out/wiza_vs_file_$n.md \
    "xargs -0 -a $corpus $root/target/release/wiza" \
    "xargs -0 -a $corpus file"
    rm $corpus
done

rm $corpus_dir/corpus.txt
rmdir $corpus_dir
