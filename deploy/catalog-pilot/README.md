# Visual catalogue pilot — local candidate

This increment extends the current Sarah bot. The production deployment has not
been changed. The public catalogue is a separate static-only service; never
publish the private dashboard on port 7788.

## Local preview

From the candidate checkout with its installed Python environment:

```sh
python -m uvicorn herandhim.catalog.web:app --host 127.0.0.1 --port 7789
```

`GET /healthz`, `GET /catalog.json`, and static assets are public. There are no
account, conversation, admin, configuration, or mutable API endpoints. The web
process does not need the owner's data directory, Telegram token or LLM key.

Portraits are shipped as WebP. Twenty delivery assets total about 1.7 MB;
original generated PNGs are retained separately as design evidence.

## Configuration to merge into the existing bot settings

Only add this block. Replace the two placeholders before enabling it:

```json
{
  "catalog": {
    "enabled": true,
    "url": "https://YOUR-APPROVED-DOMAIN/",
    "legacyOwnerId": "REPLACE_WITH_EXISTING_NUMERIC_OWNER_ID",
    "publicAccess": false,
    "maxUsers": 50
  }
}
```

`legacyOwnerId` must be the numeric Telegram user ID of the current Sarah owner,
not the bot ID. A string placeholder deliberately fails validation. Preserve
the existing `channels.telegram.allowedUsers`, provider settings, credentials,
cost limits and privacy contract. Do not run the full Sarah installer: it would
also rewrite identity/configuration that this rollout must preserve.

Start with `publicAccess: false`. Only set it to true after an approved rollout
and acceptance in Telegram. Setting it true admits new private-chat users up to
`maxUsers`; it does not change or increase provider budgets. The owner's legacy
slot counts towards the limit. No new bot token is required.

## HTTPS service

`compose.catalog.yaml` starts only the catalogue from an explicit candidate
image. It overrides the normal image entry point so it cannot start a second
polling bot. It mounts no owner data or secrets and publishes only loopback
7789. Point the approved domain's TLS reverse proxy at this service, retaining
the upstream CSP. Do not expose 7788 or proxy paths to the dashboard.

The selected heroine is sent through a Telegram keyboard Mini App `sendData`
message. The bot validates a bounded `{v: 1, character_id: ...}` payload and gets
the user ID from the private Telegram update; a supplied user ID is rejected.
Outside Telegram the landing page opens the existing bot; selection happens
from its catalogue button after the name and adult confirmation.

## Rollout and acceptance checklist

1. Read the current production revision/config read-only and confirm the actual
   owner ID. Obtain an approved HTTPS domain and concrete rollout authorization.
2. Back up the entire existing data/config directory while the bot is stopped;
   verify restoration to a separate path and record checksums of identity,
   owner history and active scene. Retain the current image for rollback.
3. Build an immutable candidate from this branch without updating dependencies.
   Validate imports, the static app, health endpoint, and package portrait files.
4. Start the static service and verify HTTPS on a phone. Merge only the catalogue
   block with public access still false, then restart the single existing bot.
5. In Telegram verify the actual keyboard Mini App: name → 18+ → selection → ten
   story buttons; reopening `/start` keeps the owner scene; switching heroines
   resumes each current episode; new story has no earlier episode memory. Check
   `/hide`, `/menu`, `/pause`, and `/resume`. Model inference is a separate paid
   acceptance step; no live provider calls were made during local development.
6. Verify two test accounts cannot see each other's state, media is not generated,
   duplicate updates do not run twice, and pending initiative from an inactive
   heroine is not sent. Check shared provider ledger and error logs.
7. Open public access only under the agreed pilot limit, with a rollout record.
   Advertising eligibility is not established by the existence of a creative.

## Rollback

Stop the bot before changing image/config. Restore the prior image and the prior
configuration (or disable `catalog.enabled`), then start the existing owner-only
bot. Do not delete catalogue player directories or overwrite the owner's history
with an old backup merely to roll back code. New catalogue data stays available
for recovery; the old runtime ignores it. Verify the expected owner scene and
delivery settings after restart.

## Known limits

- Local synthetic tests do not prove real model dialogue quality or Telegram
  webview integration. Those checks remain before a public pilot.
- Per-user locking is within one bot process. Do not run several polling workers.
- Only the current episode per heroine is selectable; old independent episode
  files are retained. Retention/deletion UX is not in this increment.
- Domain/name is still being selected. Registry absence is not a price quote,
  a purchase, or a trademark clearance.
- Private chat means isolated from other users, not end-to-end encrypted or
  anonymous: Telegram, the service operator and the AI route process the data.

Relevant protocol documentation:
[keyboard Mini Apps](https://core.telegram.org/bots/webapps#keyboard-button-mini-apps),
[untrusted WebAppData](https://core.telegram.org/bots/api#webappdata).
