#!/bin/sh
# Render the nginx config at container start and hand off to the real command.
#
# Two things the stock nginx entrypoint cannot do for us here:
#   1. Build the `resolver` directive from the container's DNS settings, which
#      differ between Docker Compose and Railway private networking.
#   2. Substitute only BACKEND_HOST / BACKEND_PORT, leaving nginx variables
#      such as $backend and $host untouched for request-time DNS resolution.
set -eu

RESOLVERS=$(awk '$1 == "nameserver" { printf " %s", $2 }' /etc/resolv.conf)
if [ -z "$RESOLVERS" ]; then
    # Docker's embedded DNS; Railway always injects at least one nameserver.
    RESOLVERS=" 127.0.0.11"
    echo "[hermes] no nameserver in /etc/resolv.conf; using $RESOLVERS" >&2
fi
printf 'resolver%s valid=10s;\n' "$RESOLVERS" > /etc/nginx/resolver.conf

envsubst '${BACKEND_HOST} ${BACKEND_PORT}' \
    < /etc/nginx/templates/default.conf.template \
    > /etc/nginx/conf.d/default.conf

exec "$@"
