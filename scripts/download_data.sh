#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-all}"

download_dataset() {
    local repo_id="$1"
    local revision="$2"
    local destination="$3"

    echo "=================================================="
    echo "Downloading: ${repo_id}"
    echo "Revision:    ${revision}"
    echo "Destination: ${destination}"
    echo "=================================================="

    python -m recsys.data.hub download \
        --repo-id "${repo_id}" \
        --revision "${revision}" \
        --destination "${destination}"

    echo "Successfully downloaded and verified ${destination}."
    echo ""
}

if [ "${TARGET}" = "all" ] || [ "${TARGET}" = "diginetica" ]; then
    download_dataset \
        "CSCI-566-SB/srgnn-processed-diginetica" \
        "ac9f069fb5305de9885a566301ddcc8f9317669e" \
        "data/processed/diginetica"
fi

if [ "${TARGET}" = "all" ] || [ "${TARGET}" = "retailrocket" ]; then
    download_dataset \
        "CSCI-566-SB/srgnn-processed-retailrocket" \
        "cb1487aa0566c97bc0b8a2def9a39f4fae45d8c6" \
        "data/processed/retailrocket"
fi

echo "Dataset download complete."
