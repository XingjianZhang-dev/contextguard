#!/usr/bin/env bash
# Download the run records from the Zenodo archive (latest version of 10.5281/zenodo.22124598),
# verify the SHA-256 checksum and unpack them into records/.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=contextguard_records_v3.0.0.zip
EXPECTED=$(cut -d' ' -f1 scripts/records.sha256)
URL=$(curl -sSL https://zenodo.org/api/records/22124598/versions/latest | python3 -c "
import json, sys
files = json.load(sys.stdin)['files']
print(next(f['links']['self'] for f in files if f['key'] == '$NAME'))")
curl -sSL -o "$NAME" "$URL"
python3 -c "
import hashlib, sys
digest = hashlib.sha256(open('$NAME', 'rb').read()).hexdigest()
sys.exit(0 if digest == '$EXPECTED' else f'checksum mismatch: {digest}')"
echo "$NAME: checksum OK"
mkdir -p records
unzip -q -o "$NAME" -d records
rm "$NAME"
echo "records/ ready: $(find records -name manifest.json | wc -l) run manifests"
