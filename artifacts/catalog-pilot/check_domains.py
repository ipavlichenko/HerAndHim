"""Read-only .com registry check; absence is not a registrar checkout quote."""
import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def check(domain):
    url = 'https://rdap.verisign.com/com/v1/domain/' + domain
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            return {'domain': domain, 'http': response.status, 'status': 'registered', 'source': url}
    except urllib.error.HTTPError as error:
        return {'domain': domain, 'http': error.code,
                'status': 'no_registry_record' if error.code == 404 else 'unknown', 'source': url}
    except Exception as error:
        return {'domain': domain, 'status': 'unknown', 'error': str(error), 'source': url}


domains = sys.argv[1:]
if not domains or any(not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,61}[a-z0-9]\.com', d) for d in domains):
    raise SystemExit('Provide .com domains only')
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(check, domains))
stamp = datetime.now(timezone.utc)
report = {'checked_at': stamp.isoformat(),
          'method': 'Verisign .com RDAP; 404 means no registry record; price/checkout not verified',
          'results': results}
path = Path(__file__).with_name('domain-check-' + stamp.strftime('%H%M%S') + '.json')
path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({'evidence': str(path), 'checked_at': report['checked_at'],
                  **{status: [r['domain'] for r in results if r['status'] == status]
                     for status in ['no_registry_record', 'registered', 'unknown']}}, ensure_ascii=False, indent=2))
