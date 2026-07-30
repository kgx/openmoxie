'''
DIRECTOR - per-volley situational guidance ("conduct", as opposed to memory's "knowledge")

A registry of small directive functions. Each examines one concern (style,
repetition, time of day, session arc, topic dwell) and optionally returns one
short imperative line. build_directives() runs the enabled ones and returns a
block the prompt template renders under SITUATIONAL GUIDANCE.

Layered configuration, most specific wins:
  1. DEFAULT_CONFIG below (code)
  2. per-device overrides in PersistentData.data['director'] (seedable via
     content-pack persist sections, editable in admin) - e.g.
     {"repetition": {"enabled": false}, "time_of_day": {"bedtime": "21:00"}}

Everything here is pure in-process Python - no API calls, no latency.
Custom directives can be registered from conversation `code` fields via the
@directive decorator.
'''
import logging
import re

from django.utils import timezone

logger = logging.getLogger(__name__)

DIRECTIVES = {}          # name -> fn(cfg, volley, session, now) -> str | None
DEFAULT_CONFIG = {
    'style': {
        'enabled': True,
        'rules': [
            "Use your friend's name at most once every few replies, not in every reply.",
            "Do not restate what your friend just said back to him.",
            "Vary how your replies begin - avoid starting with the same exclamation pattern.",
            "Not every reply needs to end with a question; sometimes just share or react.",
        ],
    },
    'repetition': {'enabled': True, 'window': 6, 'name_max': 2, 'opener_max': 2},
    'time_of_day': {'enabled': True, 'bedtime': '20:30', 'wind_down_minutes': 60},
    'session_arc': {'enabled': True, 'long_session_volleys': 120},
    'topic_dwell': {'enabled': True, 'minutes': 10},
}


def directive(name):
    def register(fn):
        DIRECTIVES[name] = fn
        return fn
    return register


def _merged_config(volley):
    overrides = {}
    try:
        overrides = (volley.persist_data or {}).get('director', {})
    except Exception:
        pass
    merged = {}
    for name, base in DEFAULT_CONFIG.items():
        merged[name] = {**base, **overrides.get(name, {})}
    for name, extra in overrides.items():   # allow config for custom directives too
        merged.setdefault(name, dict(extra))
    return merged


# Compose the guidance block. Never raises; a broken directive is skipped.
def build_directives(volley, session, now=None):
    now = now or timezone.localtime()
    cfg = _merged_config(volley)
    lines = []
    for name, fn in DIRECTIVES.items():
        dcfg = cfg.get(name, {})
        if not dcfg.get('enabled', True):
            continue
        try:
            line = fn(dcfg, volley, session, now)
        except Exception as e:
            logger.warning(f'directive {name} failed: {e}')
            continue
        if line:
            lines.append('- ' + line)
    return '\n'.join(lines)


def _recent_assistant_lines(session, window):
    hist = getattr(session, '_history', []) or []
    return [h.get('content', '') for h in hist if h.get('role') == 'assistant'][-window:]


@directive('style')
def style_rules(cfg, volley, session, now):
    rules = cfg.get('rules') or []
    return ' '.join(rules) if rules else None


@directive('repetition')
def repetition_check(cfg, volley, session, now):
    lines = _recent_assistant_lines(session, cfg.get('window', 6))
    if len(lines) < 3:
        return None
    problems = []
    nickname = ''
    try:
        nickname = (volley.config.get('child_pii', {}) or {}).get('nickname', '') or ''
    except Exception:
        pass
    names = [n for n in {nickname, 'Magic Davey', 'Davey'} if n]
    name_uses = sum(1 for l in lines for n in names if n.lower() in l.lower())
    if name_uses > cfg.get('name_max', 2):
        problems.append('you have been overusing your friend\'s name - stop using it for a while')
    openers = [' '.join(re.findall(r"[A-Za-z']+", l.lower())[:2]) for l in lines if l]
    if openers and max(openers.count(o) for o in set(openers)) > cfg.get('opener_max', 2):
        problems.append('your recent replies all begin the same way - start differently')
    words = [set(re.findall(r"[a-z']+", l.lower())) for l in lines if l]
    if len(words) >= 2:
        a, b = words[-2], words[-1]
        if a and b and len(a & b) / max(1, len(a | b)) > 0.6:
            problems.append('your last two replies were nearly identical - say something genuinely new')
    return ('Quality check: ' + '; '.join(problems) + '.') if problems else None


@directive('time_of_day')
def time_of_day(cfg, volley, session, now):
    try:
        bh, bm = (int(x) for x in cfg.get('bedtime', '20:30').split(':'))
    except Exception:
        bh, bm = 20, 30
    minutes = now.hour * 60 + now.minute
    bed = bh * 60 + bm
    if minutes >= bed:
        return ('It is past bedtime - keep your energy calm and gentle, wrap up warmly soon, '
                'and encourage getting ready for sleep.')
    if bed - minutes <= cfg.get('wind_down_minutes', 60):
        return 'Bedtime is coming soon - gradually lower the energy and avoid starting big new topics.'
    if now.hour < 9:
        return 'It is early morning - be bright and welcoming, and ask about the day ahead.'
    return None


@directive('session_arc')
def session_arc(cfg, volley, session, now):
    tv = getattr(session, 'total_volleys', 0)
    limit = cfg.get('long_session_volleys', 120)
    if tv and tv >= limit:
        return ('This has been a very long chat - gently suggest taking a break to do something '
                'in the real world, like practicing a trick for the family, and offer to continue later.')
    return None


@directive('topic_dwell')
def topic_dwell(cfg, volley, session, now):
    try:
        from .memory import topic_dwell_minutes
        dwell = topic_dwell_minutes(volley.device_id)
    except Exception:
        return None
    if dwell is not None and dwell >= cfg.get('minutes', 10):
        return ('You have been on the same topic for a while - bridge to something new and fun '
                'that connects to your friend\'s interests.')
    return None
