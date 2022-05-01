#!/usr/bin/sh

folder="$1"
d=$(date +"%Y-%m-%d-%T")
cat "$1"/*.csv | awk -F "," 'NR==1 {print $0}; $1 != "id" {print $0}' | sed 's/,/\t/g' > "$folder"/CompiledReport_"$d".tsv