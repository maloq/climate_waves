#!/usr/bin/env bash

PUBLIC_LINK="$1"          # e.g. https://disk.yandex.ru/d/AbCdEfGhIjKlMn
OUTFILE="${2:-features_2023.nc}"  # optional second arg = output filename

if [ -z "$PUBLIC_LINK" ]; then
  echo "Usage: $0 <public_link> [output_filename]"
  exit 1
fi

# 1) Get direct download URL via Yandex public API
DIRECT_URL=$(curl -sG \
  --data-urlencode "public_key=$PUBLIC_LINK" \
  'https://cloud-api.yandex.net/v1/disk/public/resources/download' \
  | grep -oP '(?<="href":")[^"]+')

if [ -z "$DIRECT_URL" ]; then
  echo "Could not get direct download url. Check your public link."
  exit 1
fi

# 2) Download the file
curl -L "$DIRECT_URL" -o "$OUTFILE"

echo "Saved to $OUTFILE"