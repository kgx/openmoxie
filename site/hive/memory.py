'''
MEMORY - Fragment-based long-term memory (memory v3)

Memory is stored as per-device rows (MemoryFragment) grouped into named banks,
updated by DIFFS (operation lists emitted by an LLM extractor) rather than
whole-state rewrites, and assembled into prompt context per volley within a
budget. Ground truth remains ChatTranscript; fragments are derived data.

Both entry points are safe to call from worker threads (DB access is wrapped
in run_db_atomic) and are intended to be used from conversation `code` hooks:

    from hive.memory import apply_memory_ops, assemble_memory
'''
import logging
import math
import re
import threading

from .models import MemoryFragment, MoxieDevice
from .mqtt.util import run_db_atomic

logger = logging.getLogger(__name__)

_VALID_OPS = ('add', 'update', 'retire', 'restore')
# control banks: consumed by the director, never injected as "memories"
CONTROL_BANKS = ('objectives', 'seeds')
_EMBED_MODEL = 'text-embedding-3-small'
_EMBED_DIM = 256
_TOPIC_MIX = 0.3   # weight of the newest turn in the rolling topic vector

# per-device rolling conversation-topic vectors; in-memory only (rewarms in one turn)
_topic_vectors = {}
# per-device recent (monotonic_seconds, vector) samples for dwell detection
_topic_history = {}
_TOPIC_HISTORY_MAX = 60
_DWELL_SIMILARITY = 0.80
_topic_lock = threading.Lock()


def _embed_texts(texts):
    # returns a list of vectors, or None when embeddings are unavailable (no key, offline,
    # API error) - callers must treat None as 'fall back to lexical'
    try:
        from .mqtt.ai_factory import create_openai
        resp = create_openai().embeddings.create(model=_EMBED_MODEL, input=list(texts),
                                                 dimensions=_EMBED_DIM)
        return [d.embedding for d in resp.data]
    except Exception as e:
        logger.debug(f'embeddings unavailable: {e}')
        return None


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# Fold a just-spoken turn into the device's rolling topic vector. Called asynchronously
# (worker pool) as notify records arrive - never from the response path.
def update_topic_vector(device_id, text):
    if not text or not text.strip():
        return
    vecs = _embed_texts([text])
    if not vecs:
        return
    new = vecs[0]
    import time as _time
    with _topic_lock:
        old = _topic_vectors.get(device_id)
        if old and len(old) == len(new):
            _topic_vectors[device_id] = [(1.0 - _TOPIC_MIX) * o + _TOPIC_MIX * n
                                         for o, n in zip(old, new)]
        else:
            _topic_vectors[device_id] = new
        hist = _topic_history.setdefault(device_id, [])
        hist.append((_time.monotonic(), _topic_vectors[device_id]))
        del hist[:-_TOPIC_HISTORY_MAX]


def get_topic_vector(device_id):
    with _topic_lock:
        return _topic_vectors.get(device_id)


# How long (minutes) the conversation has stayed on its current topic: the age of the
# oldest consecutive topic sample still similar to the current one. None when unknown.
def topic_dwell_minutes(device_id):
    import time as _time
    with _topic_lock:
        hist = list(_topic_history.get(device_id, []))
        current = _topic_vectors.get(device_id)
    if not current or len(hist) < 2:
        return None
    oldest_similar = None
    for ts, vec in reversed(hist):
        if len(vec) == len(current) and _cosine(vec, current) >= _DWELL_SIMILARITY:
            oldest_similar = ts
        else:
            break
    if oldest_similar is None:
        return 0.0
    return (_time.monotonic() - oldest_similar) / 60.0
_WORD_RE = re.compile(r"[a-z0-9']+")
# words too common to signal relevance
_STOP = set('the a an and or but so to of in on at for with is are was were be been do does did '
            'you your i my me we our he she it they them this that these those what when where '
            'who how why not no yes have has had can could will would like about'.split())


def _apply_ops_atomic(device_id, ops):
    device = MoxieDevice.objects.filter(device_id=device_id).first()
    if not device:
        return {'error': f'unknown device {device_id}'}, []
    applied, skipped, changed = 0, 0, []
    for op in ops:
        if not isinstance(op, dict) or op.get('op') not in _VALID_OPS \
                or not op.get('bank') or not op.get('key'):
            skipped += 1
            continue
        kind = op['op']
        bank, key = str(op['bank'])[:40], str(op['key'])[:120].strip().casefold()
        frag = MemoryFragment.objects.filter(device=device, bank=bank, key=key).first()
        if kind in ('add', 'update'):
            if not op.get('text'):
                skipped += 1
                continue
            if frag:
                frag.text = str(op['text'])
                frag.times_seen += 1
                if 'confidence' in op:
                    try:
                        frag.confidence = min(1.0, max(0.0, float(op['confidence'])))
                    except (TypeError, ValueError):
                        pass
                frag.active = True
                frag.save()
            else:
                try:
                    conf = min(1.0, max(0.0, float(op.get('confidence', 0.5))))
                except (TypeError, ValueError):
                    conf = 0.5
                frag = MemoryFragment.objects.create(device=device, bank=bank, key=key,
                                                     text=str(op['text']), confidence=conf)
            changed.append((frag.pk, frag.text))
            applied += 1
        elif kind in ('retire', 'restore'):
            if frag:
                frag.active = (kind == 'restore')
                frag.save()
                applied += 1
            else:
                skipped += 1
    return {'applied': applied, 'skipped': skipped}, changed


def _store_embeddings_atomic(pk_vecs):
    for pk, vec in pk_vecs:
        # .update() deliberately skips auto_now so embedding writes don't fake recency
        MemoryFragment.objects.filter(pk=pk).update(embedding=vec)


# Apply an extractor-emitted operation list to a device's memory fragments.
# ops: [{op: add|update|retire|restore, bank, key, text?, confidence?}, ...]
# Invalid operations are skipped, never fatal. Returns {'applied': n, 'skipped': m}.
# Changed fragments are (re-)embedded outside the transaction; embedding failure
# degrades to lexical-only retrieval, never blocks the ops.
def apply_memory_ops(device_id, ops):
    try:
        result, changed = run_db_atomic(_apply_ops_atomic, device_id, list(ops or []))
        if changed:
            vecs = _embed_texts([text for _, text in changed])
            if vecs and len(vecs) == len(changed):
                run_db_atomic(_store_embeddings_atomic,
                              [(pk, vec) for (pk, _), vec in zip(changed, vecs)])
        return result
    except Exception as e:
        logger.warning(f'apply_memory_ops failed: {e}')
        return {'error': str(e)}


def _list_fragments_atomic(device_id, include_retired):
    q = MemoryFragment.objects.filter(device__device_id=device_id)
    if not include_retired:
        q = q.filter(active=True)
    return [f'{f.bank} | {f.key} | {f.text}' + ('' if f.active else ' [retired]')
            for f in q.order_by('bank', 'key')]


# Compact one-line-per-fragment listing for extractor prompts.
def list_fragments_compact(device_id, include_retired=False):
    try:
        return '\n'.join(run_db_atomic(_list_fragments_atomic, device_id, include_retired))
    except Exception as e:
        logger.warning(f'list_fragments_compact failed: {e}')
        return ''


def _stem(w):
    # cheap singular/plural folding so 'twirly' cues 'twirlies', 'bunnies' cues 'bunny'
    if len(w) > 4 and w.endswith('ies'):
        return w[:-3] + 'y'
    if len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
        return w[:-1]
    return w


def _words(text):
    return {_stem(w) for w in _WORD_RE.findall(text.casefold())} - _STOP


# details must earn their way into context: score at/above this = cued by conversation
CUE_SCORE = 1.0
# small novelty slot: un-cued details rotated in, least-recently-surfaced first
ROTATION_SLOTS = 2
_DISTANT_PAST = None  # placeholder; computed lazily to keep tz handling in one place


def _long_ago():
    import datetime as _dt
    return _dt.datetime(2000, 1, 1, tzinfo=_dt.timezone.utc)


def _assemble_atomic(device_id, utterance, budget_chars, core_banks, topic_vec, include_details):
    device = MoxieDevice.objects.filter(device_id=device_id).first()
    if not device:
        return '', []
    frags = [f for f in MemoryFragment.objects.filter(device=device, active=True)
             if f.bank not in CONTROL_BANKS]
    if not frags:
        return '', []
    newest = max(f.updated for f in frags)
    spoken = _words(utterance or '')

    def score(f):
        overlap = len(spoken & _words(f.text + ' ' + f.key)) if spoken else 0
        recency = 1.0 - min(1.0, (newest - f.updated).total_seconds() / (30 * 86400))
        semantic = 0.0
        if topic_vec and f.embedding and len(f.embedding) == len(topic_vec):
            semantic = _cosine(topic_vec, f.embedding)
        return overlap * 2.0 + semantic * 2.0 + f.confidence * 0.5 + recency * 0.25

    core = [f for f in frags if f.bank in core_banks]
    details = [f for f in frags if f.bank not in core_banks]
    cued, rotation = [], []
    if include_details:
        cued = sorted((f for f in details if score(f) >= CUE_SCORE), key=score, reverse=True)
        # habituation: novelty slot prefers never/least-recently surfaced details
        fresh_pool = sorted((f for f in details if f not in cued),
                            key=lambda f: (f.last_surfaced or _long_ago(), -f.confidence))
        rotation = fresh_pool[:ROTATION_SLOTS]
    picked, used, featured = [], 0, []
    for f in core + cued + rotation:
        cost = len(f.text) + len(f.bank) + 4
        if f not in core and used + cost > budget_chars:
            continue
        picked.append(f)
        used += cost
        if f not in core:
            featured.append(f.pk)
    by_bank = {}
    for f in picked:
        by_bank.setdefault(f.bank, []).append(f.text)
    lines = []
    for bank in sorted(by_bank, key=lambda b: (b not in core_banks, b)):
        lines.append(bank.replace('_', ' ').upper() + ':')
        lines.extend('- ' + t for t in by_bank[bank])
        lines.append('')
    return '\n'.join(lines).strip(), featured


def _mark_surfaced_atomic(pks):
    from django.db.models import F
    from django.utils import timezone as djtz
    MemoryFragment.objects.filter(pk__in=pks).update(
        last_surfaced=djtz.now(), surfaced_count=F('surfaced_count') + 1)


# Build the memory context block for a prompt, mimicking human recall:
# gist-level banks (profile, gists) are ALWAYS present; detail fragments appear only
# when cued by the utterance/topic vector, plus a small novelty-rotation slot of
# least-recently-surfaced details. Featured details are recorded in the habituation
# ledger so they naturally rest before coming up again. No API calls in this path.
def assemble_memory(device_id, utterance='', budget_chars=6000,
                    core_banks=('profile', 'gists'), include_details=True):
    try:
        text, featured = run_db_atomic(_assemble_atomic, device_id, utterance, budget_chars,
                                       tuple(core_banks), get_topic_vector(device_id),
                                       include_details)
        if featured:
            run_db_atomic(_mark_surfaced_atomic, featured)
        return text
    except Exception as e:
        logger.warning(f'assemble_memory failed: {e}')
        return ''


def _recently_surfaced_atomic(device_id, hours, limit):
    import datetime as _dt

    from django.utils import timezone as djtz
    cutoff = djtz.now() - _dt.timedelta(hours=hours)
    q = MemoryFragment.objects.filter(device__device_id=device_id, active=True,
                                      last_surfaced__gte=cutoff) \
                              .exclude(bank__in=CONTROL_BANKS) \
                              .order_by('-last_surfaced')[:limit]
    return [f.text for f in q]


# Texts of memories featured in the last N hours - fuel for the director's
# freshness rule ("don't bring these up again unless the child does").
def recently_surfaced(device_id, hours=48, limit=5):
    try:
        return run_db_atomic(_recently_surfaced_atomic, device_id, hours, limit)
    except Exception as e:
        logger.warning(f'recently_surfaced failed: {e}')
        return []


def _pick_anchor_atomic(device_id):
    import random
    frags = [f for f in MemoryFragment.objects.filter(device__device_id=device_id, active=True)
             if f.bank not in CONTROL_BANKS and f.bank not in ('profile', 'gists')]
    if not frags:
        return None
    frags.sort(key=lambda f: (f.last_surfaced or _long_ago(), -f.confidence))
    pick = random.choice(frags[:2])
    _mark_surfaced_atomic([pick.pk])
    return pick.text


# Choose one detail memory to feature (e.g. in an opener), novelty-weighted:
# random among the least-recently-surfaced few, then marked as surfaced.
def pick_opener_anchor(device_id):
    try:
        return run_db_atomic(_pick_anchor_atomic, device_id)
    except Exception as e:
        logger.warning(f'pick_opener_anchor failed: {e}')
        return None
