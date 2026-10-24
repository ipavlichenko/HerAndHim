#!/bin/sh
set -eu

STACK_DIR="${SARA_STACK_DIR:-/opt/sara/deploy}"

install -m 0644 "$STACK_DIR/00-00-sara-sshd.conf" /etc/ssh/sshd_config.d/00-00-sara.conf
rm -f /etc/ssh/sshd_config.d/99-sara.conf
sshd -t
systemctl reload ssh

if ! swapon --show=NAME --noheadings | grep -Fxq /swapfile; then
  if [ ! -f /swapfile ]; then
    fallocate -l 1G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile >/dev/null
  fi
  swapon /swapfile
fi
if ! grep -Fq '/swapfile none swap sw 0 0' /etc/fstab; then
  printf '%s\n' '/swapfile none swap sw 0 0' >> /etc/fstab
fi

install -m 0644 "$STACK_DIR/sara-backup.service" /etc/systemd/system/sara-backup.service
install -m 0644 "$STACK_DIR/sara-backup.timer" /etc/systemd/system/sara-backup.timer
systemctl daemon-reload
systemctl enable --now sara-backup.timer >/dev/null

echo "host provisioning complete"

