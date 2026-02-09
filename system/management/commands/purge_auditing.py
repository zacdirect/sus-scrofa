# Sus Scrofa - Copyright (C) 2026 Sus Scrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

from django.core.management.base import BaseCommand

from users.models import Activity


class Command(BaseCommand):
    """Purge auditing table."""

    help = "Purge auditing table"

    def handle(self, *args, **options):
        """Runs command."""

        print("Audit log purge")
        print("WARNING: this will permanently delete all your audit logs!")

        ans = input("Do you want to continue? [y/n] ")

        if ans.strip().lower() == "y":
            print("Purging audit log... (it could take several minutes)")
            Activity.objects.all().delete()
            print("Done.")
        else:
            print("Please use only y/n")