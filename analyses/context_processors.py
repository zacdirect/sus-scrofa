# Sus Scrofa - Copyright (C) 2026 Sus Scrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

from django.conf import settings
from analyses.models import Case, Analysis

def dashboard_data(request):
    """Populate dashboard data showed on top of each page."""
    open_cases_count = Case.objects.filter(state="O").count()
    analyses_complete_count = Analysis.objects.filter(state="C").count()
    analyses_wait_count = Analysis.objects.filter(state="W").count()

    return {"open_cases_count": open_cases_count,
            "analyses_complete_count": analyses_complete_count,
            "analyses_wait_count": analyses_wait_count}

def sus_scrofa_release(request):
    """Context processor used to populate the sus_scrofa release label in all pages."""
    return {"sus_scrofa_release": settings.SUS_SCROFA_VERSION}