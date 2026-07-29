import json
import sys

from deepmerge import always_merger
from django.core.management.base import BaseCommand, CommandError

from ...data_import import import_content
from ...memory import apply_memory_ops
from ...models import MoxieDevice, PersistentData


class Command(BaseCommand):
    help = ("Load content pack JSON files into the database. Packs use the import/export schema "
            "(globals, schedules, conversations) plus an optional 'persist' list of per-device "
            "PersistentData seeds. All records in the pack are applied unconditionally; use "
            "high source_version values so init_data never overwrites them. The running MQTT "
            "service must be told to reload afterward (/hive/reload_database).")

    def add_arguments(self, parser):
        parser.add_argument('files', nargs='+', help="content pack paths, or '-' to read from stdin")

    def handle(self, *args, **options):
        # command runs in its own process; set the OpenAI key so fragment seeding can embed
        from ...models import HiveConfiguration
        from ...mqtt.ai_factory import set_openai_key
        cfg = HiveConfiguration.objects.filter(name='default').first()
        if cfg and cfg.openai_api_key:
            set_openai_key(cfg.openai_api_key)
        for path in options['files']:
            if path == '-':
                data = json.load(sys.stdin)
            else:
                with open(path) as f:
                    data = json.load(f)
            message = import_content(data,
                                     list(range(len(data.get('globals', [])))),
                                     list(range(len(data.get('schedules', [])))),
                                     list(range(len(data.get('conversations', [])))))
            self.stdout.write(f'{path}: {message}')
            for seed in data.get('persist', []):
                device = MoxieDevice.objects.filter(device_id=seed['device_id']).first()
                if not device:
                    raise CommandError(f"Unknown device_id {seed['device_id']}")
                pdata, _ = PersistentData.objects.get_or_create(device=device, defaults={'data': {}})
                if seed.get('replace', False):
                    pdata.data = seed['data']
                else:
                    pdata.data = always_merger.merge(pdata.data, seed['data'])
                pdata.save()
                mode = 'replaced' if seed.get('replace', False) else 'merged'
                self.stdout.write(f"{path}: {mode} persist data for {seed['device_id']}")
            for seed in data.get('fragments', []):
                device_id = seed['device_id']
                if not MoxieDevice.objects.filter(device_id=device_id).exists():
                    raise CommandError(f'Unknown device_id {device_id}')
                result = apply_memory_ops(device_id, seed.get('ops', []))
                self.stdout.write(f'{path}: fragments for {device_id}: {result}')
