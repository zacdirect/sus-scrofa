# Sus Scrofa - Copyright (C) 2026 Sus Scrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_safe
from django.shortcuts import render
from django.http import HttpResponse

from lib.utils import deps_check


@require_safe
@login_required
def dependencies_list(request):
    return render(request, "admin/dependencies.html",
                              {"dependencies": deps_check()})