import json

from django.core.management.base import BaseCommand

from ...dream import nightly_maintenance


class Command(BaseCommand):
    help = ('Run the nightly memory maintenance ("dream") immediately: confidence decay, '
            'embedding-flagged duplicate merging, episode distillation, and a dated DB backup. '
            'Normally fired automatically by the server at DREAM_HOUR local time.')

    def handle(self, *args, **options):
        self.stdout.write(json.dumps(nightly_maintenance(), indent=2))
