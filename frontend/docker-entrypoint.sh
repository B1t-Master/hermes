#!/bin/sh
# Render the nginx config at container start and hand off to the real command.
#
# Two things the stock nginx entrypoint cannot do for us here:
#   1. Build the `resolver` directive from the container's DNS settings, which
#      differ between Docker Compose and Railway private networking.
#   2. Substitute only BACKEND_HOST / BACKEND_PORT, leaving nginx variables
#      such as $backend and $host untouched for request-time DNS resolution.
set -eu

RESOLVERS=""
for ns in $(awk '$1 == "nameserver" { print $2 }' /etc/resolv.conf); do
    # Railway uses a bare IPv6 nameserver (fd12::10); nginx requires brackets
    # around IPv6 literals or it fails with "invalid port in resolver".
    ns="${ns%%%*}"          # drop any zone index, e.g. %eth0
    case "$ns" in
        *:*) ns="[$ns]" ;;  # IPv6 -> [fd12::10]; IPv4 passes through unchanged
    esac
    RESOLVERS="$RESOLVERS $ns"
done
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
