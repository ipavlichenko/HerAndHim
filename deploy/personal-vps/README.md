# Personal VPS pilot deployment

This bundle deploys one owner-only Sarah instance. It deliberately exposes no
public application port: the dashboard is bound to VPS loopback and is reached
only through an SSH tunnel.

## Host layout

- `/opt/sara/source` — transient build context, removable after image build.
- `/opt/sara/deploy` — this operations bundle.
- `/srv/sara/data` — persistent runtime state, owned by UID/GID 10001.
- `/srv/sara/backups` — seven verified local snapshots; provider backup is the
  off-host layer.
- `/etc/herandhim/secrets` — root-only secret files, each mode `0600`.

Required secret filenames are documented by `render_pilot_config.py`. Secret
values must never be passed on the command line or saved in the source tree.

## Build and first start

1. Build the image from the transferred dirty working tree:
   `docker build -t herandhim-sara:pilot -f deploy/docker/Dockerfile .`
2. Initialise `/srv/sara/data` with the image's packaged templates using an
   overridden entrypoint, then run `install_identity.py`. It archives the old
   identity, removes every conflicting `.md`/`.txt` template from the three
   active identity directories, installs the canonical Sarah files (including
   the visual description required by the selfie pipeline), and adds the
   Moscow city profile used by weather and planning.
3. Copy the currently approved pilot reference to
   `/srv/sara/data/context/appearance/reference.jpg`. This is a reversible
   deployment hypothesis, not a decision that the reference is Sarah's final
   canonical appearance.
4. Render `/srv/sara/data/herandhim.json` as root, then start `compose.yml`.
5. Enable `sara-backup.timer`; create one backup and pass `restore.py` against it.

## Operator commands

`sara-ops.sh` provides status, health, bounded logs, restart, a full process
kill/resume switch, the durable proactive-delivery switch, and backup/restore
verification. It never prints configuration or secrets.

## Image retention after a successful deployment

The deployment operator must clean up obsolete Sarah images after the new
release has passed health, restart, and transport checks. This is part of each
deployment; a separate owner approval is not required.

- Keep the running image and the two most recent **successfully deployed**
  rollback images. Resolve and record their full image IDs before cleanup;
  candidate tags or creation timestamps do not prove a successful release.
- Protect every image referenced by any container and keep the Python base image.
- Inventory obsolete `herandhim-sara` tags and untagged Sarah runtime/builder
  images. Confirm ownership from image configuration/build history and review
  the exact deletion list before applying it.
- Remove only those image references using `docker image rm` without `--force`.
  Never use broad system/volume pruning or delete data/backups during image
  retention. Defer cleanup while another build or deployment is active.
- Verify the running container identity, health, both rollback image IDs, and
  disk space afterwards. Record the result in the project handoff.

This is an operator deployment step, not a scheduled background cleanup job.
Removed old artifacts require a rebuild from their source revision if needed;
without pinned dependencies such a rebuild may not be byte-identical.

For private dashboard access:

```text
ssh -L 7788:127.0.0.1:7788 root@VPS_IP
```

Then open `http://127.0.0.1:7788`. Read the dashboard token directly from the
root-only VPS file; do not put it in shell history, chat, screenshots, or logs.

