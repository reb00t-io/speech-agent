#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# --- prompt ---

read -rp "PORT [29000]: " port
port="${port:-29000}"
if ! [[ "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
  echo "invalid port: ${port}"
  exit 1
fi

default_name="$(basename "$PWD")"
read -rp "Project name [${default_name}]: " project_name
project_name="${project_name:-$default_name}"
if ! [[ "$project_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "invalid project name: only alphanumerics, hyphens, and underscores allowed"
  exit 1
fi

while true; do
  read -rp "PUBLIC_URL (e.g. https://myapp.example.com): " public_url
  if [[ "$public_url" =~ ^https?:// ]]; then
    break
  fi
  echo "PUBLIC_URL must start with https:// or http://"
done

# strip protocol to get the hostname
hostname="${public_url#https://}"
hostname="${hostname#http://}"
hostname="${hostname%%/*}"

if ! [[ "$hostname" =~ ^[a-zA-Z0-9.-]+$ ]]; then
  echo "invalid hostname: ${hostname}"
  exit 1
fi

# --- .envrc ---

# Match any PORT= and PUBLIC_URL= lines regardless of current value
sed -i.bak \
  -e "s|^export PORT=.*|export PORT=${port}|" \
  -e "s|^export PUBLIC_URL=.*|export PUBLIC_URL=\"${public_url}\"|" \
  .envrc
rm -f .envrc.bak

echo "updated .envrc"

# --- pyproject.toml ---

sed -i.bak "s|^name = .*|name = \"${project_name}\"|" pyproject.toml
rm -f pyproject.toml.bak

echo "updated pyproject.toml"

# --- docker ---

sed -i.bak "s|-t [a-zA-Z0-9_-]*|-t ${project_name}|" scripts/build.sh
rm -f scripts/build.sh.bak

sed -i.bak "s|^IMAGE_NAME=.*|IMAGE_NAME=\"${project_name}\"|" scripts/deploy.sh
rm -f scripts/deploy.sh.bak

sed -i.bak "s|^IMAGE_NAME=.*|IMAGE_NAME=\"${project_name}\"|" scripts/get_logs.sh
rm -f scripts/get_logs.sh.bak

sed -i.bak \
  -e "s|^  bootstrap-template:|  ${project_name}:|" \
  -e "s|image: .*|image: ${project_name}|" \
  docker-compose.yml
rm -f docker-compose.yml.bak

echo "updated docker-compose.yml, build.sh, deploy.sh"

# --- nginx config ---

# Find the current nginx conf (any .conf file in config/nginx/)
old_conf=$(find config/nginx -name '*.conf' -print -quit 2>/dev/null || true)
new_conf="config/nginx/${hostname}.conf"

if [[ -z "$old_conf" ]]; then
  echo "no nginx config found in config/nginx/"
  exit 1
fi

# Update server_name and proxy_pass port inside the file
sed -i.bak \
  -e "s|server_name .*;|server_name ${hostname};|" \
  -e "s|proxy_pass http://localhost:[0-9]*;|proxy_pass http://localhost:${port};|" \
  "$old_conf"
rm -f "${old_conf}.bak"

# Rename if the hostname changed
if [[ "$old_conf" != "$new_conf" ]]; then
  mv "$old_conf" "$new_conf"
  echo "renamed $(basename "$old_conf") -> $(basename "$new_conf")"
fi

echo "updated nginx config"

# --- nginx configure script ---

old_script=$(find config/nginx -name 'configure-nginx-*.sh' -print -quit 2>/dev/null || true)
new_script="config/nginx/configure-nginx-${hostname}.sh"

if [[ -n "$old_script" ]]; then
  sed -i.bak "s|configure-nginx\.sh .*|configure-nginx.sh ${hostname}|" "$old_script"
  rm -f "${old_script}.bak"

  if [[ "$old_script" != "$new_script" ]]; then
    mv "$old_script" "$new_script"
    echo "renamed $(basename "$old_script") -> $(basename "$new_script")"
  fi

  echo "updated nginx configure script"
fi

# --- AGENTS.md ---

rm -f AGENTS.md
if [[ -f AGENTS_TEMPLATE.md ]]; then
  mv AGENTS_TEMPLATE.md AGENTS.md
  echo "updated AGENTS.md"
fi

# --- direnv ---

read -rp "Run 'direnv allow' now? [Y/n]: " allow_direnv
allow_direnv="${allow_direnv:-Y}"
if [[ "$allow_direnv" =~ ^[Yy]$ ]]; then
  direnv allow
  echo "direnv allowed"
else
  echo "skipped — run 'direnv allow' to reload"
fi

echo "done"
