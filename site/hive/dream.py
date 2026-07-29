'''
DREAM - nightly offline memory maintenance ("sleep consolidation")

Runs inside the server process from the runserver daemon loop: once per night
(local time), for every device with memory fragments:

  1. Mechanical confidence decay on stale non-core fragments (deterministic).
  2. Embedding-based near-duplicate candidate detection (deterministic).
  3. One LLM pass that reviews the full fragment listing plus the duplicate
     candidates and emits memory ops: merge duplicates, retire stale trivia,
     distill old episodes into semantic facts, fix contradictions.
  4. A dated sqlite backup of the whole database, pruned to the last 14.

Everything is offline - latency does not matter here, so the strongest
reasonable model does the judgment work.
'''
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime

from django.conf import settings
from django.utils import timezone

from .memory import _cosine, apply_memory_ops
from .models import ChatTranscript, HiveConfiguration, MemoryFragment, MoxieDevice
from .mqtt.ai_factory import create_openai, set_openai_key
from .mqtt.conversations import inference_token_params
from .mqtt.util import run_db_atomic

logger = logging.getLogger(__name__)

DREAM_MODEL = 'gpt-5.6-terra'
DREAM_HOUR = 3               # local time (settings.TIME_ZONE)
CORE_BANKS = ('profile', 'magic_tricks')
DECAY_AFTER_DAYS = 45
DUP_THRESHOLD = 0.88
BACKUP_KEEP = 14
_MARKER_FILE = 'last_dream.json'


def _decay_atomic(device_id):
    now = timezone.now()
    decayed = 0
    stale = MemoryFragment.objects.filter(device__device_id=device_id, active=True) \
                                  .exclude(bank__in=CORE_BANKS)
    for f in stale:
        days = (now - f.updated).days
        if days > DECAY_AFTER_DAYS:
            new_conf = max(0.15, f.confidence - 0.05 * (days // 30))
            if new_conf < f.confidence:
                # .update() skips auto_now so decay does not look like recency
                MemoryFragment.objects.filter(pk=f.pk).update(confidence=new_conf)
                decayed += 1
    return decayed


def duplicate_pairs(frags):
    embedded = [f for f in frags if f.embedding]
    pairs = []
    for i in range(len(embedded)):
        for j in range(i + 1, len(embedded)):
            a, b = embedded[i], embedded[j]
            if len(a.embedding) == len(b.embedding) \
                    and _cosine(a.embedding, b.embedding) > DUP_THRESHOLD:
                pairs.append((f'{a.bank}/{a.key}', f'{b.bank}/{b.key}'))
    return pairs


def _listing_atomic(device_id):
    now = timezone.now()
    frags = list(MemoryFragment.objects.filter(device__device_id=device_id, active=True))
    lines = [f'{f.bank} | {f.key} | conf={f.confidence:.2f} | {(now - f.updated).days}d old | {f.text}'
             for f in frags]
    return frags, lines


def _dream_llm(prompt):
    resp = create_openai().chat.completions.create(
        model=DREAM_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        **inference_token_params(DREAM_MODEL, 3000))
    return resp.choices[0].message.content


def run_dream(device_id):
    decayed = run_db_atomic(_decay_atomic, device_id)
    frags, lines = run_db_atomic(_listing_atomic, device_id)
    if not frags:
        return {'decayed': decayed, 'applied': 0}
    dups = duplicate_pairs(frags)
    prompt = ('You are performing nightly memory maintenance for Moxie, a robot friend of a child '
              'who is an aspiring magician. ALL MEMORY FRAGMENTS are listed below, one per line as: '
              'bank | key | confidence | age | text. '
              + ('LIKELY DUPLICATE PAIRS (by embedding similarity): '
                 + '; '.join(f'{a} ~ {b}' for a, b in dups) + '. ' if dups else '')
              + 'Reply with ONLY a JSON array of maintenance operations: '
                '[{"op": "add"|"update"|"retire", "bank": str, "key": str, "text": str, "confidence": float}]. '
                'Merge duplicate fragments (update the better key with the combined text, retire the other). '
                'Distill episodes older than two weeks into durable facts in the right banks, then retire '
                'those episodes. Retire trivial or clearly stale fragments. Correct contradictions, keeping '
                'the newer information. Do NOT change fragments that are fine as-is; an empty array is fine.'
                '\n\nMEMORY FRAGMENTS:\n' + '\n'.join(lines))
    try:
        raw = _dream_llm(prompt)
    except Exception as e:
        return {'decayed': decayed, 'error': str(e)}
    try:
        cleaned = raw
        if '```' in cleaned:
            cleaned = cleaned.split('```')[1]
            if cleaned.startswith('json'):
                cleaned = cleaned[4:]
        ops = json.loads(cleaned)
    except Exception:
        return {'decayed': decayed, 'error': 'unparseable: ' + raw[:200]}
    result = apply_memory_ops(device_id, ops) if isinstance(ops, list) else {'applied': 0, 'skipped': 0}
    return {'decayed': decayed, 'dup_candidates': len(dups), **result}


def backup_database(src_path=None, dest_dir=None):
    backups = dest_dir or os.path.join(settings.DATA_STORE_DIR, 'backups')
    os.makedirs(backups, exist_ok=True)
    dest = os.path.join(backups, 'db-' + timezone.localtime().strftime('%Y-%m-%d') + '.sqlite3')
    src = sqlite3.connect(str(src_path or settings.DATABASES['default']['NAME']))
    dst = sqlite3.connect(dest)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    existing = sorted(f for f in os.listdir(backups) if f.startswith('db-'))
    for old in existing[:-BACKUP_KEEP]:
        os.remove(os.path.join(backups, old))
    return dest


def _devices_with_fragments_atomic():
    return list(MemoryFragment.objects.values_list('device__device_id', flat=True).distinct())


def _openai_key_atomic():
    cfg = HiveConfiguration.objects.filter(name='default').first()
    return cfg.openai_api_key if cfg else None


# Devices with transcript activity since the given ISO timestamp (all, when None) -
# the gate that keeps dream/backup from running on an unchanged system.
def _active_devices_atomic(since_iso):
    from django.utils.dateparse import parse_datetime
    q = ChatTranscript.objects.all()
    since = parse_datetime(since_iso) if since_iso else None
    if since:
        q = q.filter(timestamp__gt=since)
    return list(q.values_list('device__device_id', flat=True).distinct())


def nightly_maintenance(only_devices=None):
    key = run_db_atomic(_openai_key_atomic)
    if key:
        set_openai_key(key)
    results = {}
    device_ids = run_db_atomic(_devices_with_fragments_atomic)
    if only_devices is not None:
        device_ids = [d for d in device_ids if d in only_devices]
    for device_id in device_ids:
        results[device_id] = run_dream(device_id)
        logger.info(f'dream: {device_id} -> {results[device_id]}')
    backup = backup_database()
    logger.info(f'dream: database backed up to {backup}')
    return results


def _safe_maintenance(only_devices=None):
    try:
        nightly_maintenance(only_devices)
    except Exception as e:
        logger.error(f'dream: nightly maintenance failed: {e}')


# Called once a minute from the runserver daemon loop; fires the nightly maintenance
# in a background thread once per day at DREAM_HOUR local time - and only when there
# has been conversation activity since the last run (otherwise dreaming and backup
# would burn LLM calls and disk on an unchanged system).
def nightly_tick(now=None):
    now = now or timezone.localtime()
    if now.hour != DREAM_HOUR:
        return False
    marker = os.path.join(settings.DATA_STORE_DIR, _MARKER_FILE)
    today = now.strftime('%Y-%m-%d')
    state = {}
    try:
        with open(marker) as f:
            state = json.load(f)
    except Exception:
        state = {}
    if state.get('date') == today:
        return False
    active = run_db_atomic(_active_devices_atomic, state.get('last_run'))
    state['date'] = today
    if active:
        state['last_run'] = now.isoformat()
    with open(marker, 'w') as f:
        json.dump(state, f)
    if not active:
        logger.info('dream: no conversation activity since last run; skipping maintenance')
        return False
    threading.Thread(target=_safe_maintenance, args=(active,), daemon=True).start()
    return True
