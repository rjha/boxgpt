#!/usr/bin/env bash

set -euo pipefail

SOURCE_DIR="$HOME/code/github/boxgpt/scripts/x.out/excel"
BACKUP_DIR="$HOME/backup/excel"
TS=$(date +%d_%m_%H%M)
TMP_DIR="/tmp/excel_${TS}"
ARCHIVE="/tmp/excel_${TS}.tar.gz"

# --- Execution ---
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist." >&2
    exit 1
fi

cp -R "$SOURCE_DIR" "$TMP_DIR"
tar -cvzf "$ARCHIVE" -C /tmp "excel_${TS}"
mv "$ARCHIVE" "$BACKUP_DIR/"
rm -rf "$TMP_DIR"

# delete source files while keeping subfolders
find "$SOURCE_DIR" -type f -delete
# we are done.
echo "Backup complete: ${BACKUP_DIR}/excel_${TS}.tar.gz"

