#!/bin/sh
set -eu

STACK_DIR="${SARA_STACK_DIR:-/opt/sara/deploy}"
COMPOSE="docker compose --project-directory $STACK_DIR -f $STACK_DIR/compose.yml"

case "${1:-}" in
  status) $COMPOSE ps ;;
  health) curl --fail --silent --show-error http://127.0.0.1:7788/healthz ;;
  logs) $COMPOSE logs --tail "${2:-200}" sara ;;
  restart) $COMPOSE restart sara ;;
  kill) $COMPOSE stop sara ;;
  resume) $COMPOSE up -d sara ;;
  delivery-off|delivery-on|delivery-status)
    action="${1#delivery-}"
    $COMPOSE exec -T sara python -m herandhim --config /data/herandhim.json proactive-delivery "$action"
    ;;
  backup) python3 "$STACK_DIR/backup.py" ;;
  restore-check) python3 "$STACK_DIR/restore.py" --archive "${2:?archive path required}" --verify-only ;;
  *)
    echo "usage: $0 {status|health|logs [n]|restart|kill|resume|delivery-off|delivery-on|delivery-status|backup|restore-check ARCHIVE}" >&2
    exit 2
    ;;
esac

