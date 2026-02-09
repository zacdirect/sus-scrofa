# Sus Scrofa - Copyright (C) 2026 Sus Scrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

from django.conf.urls import patterns, url

urlpatterns = patterns("",
    url(r"^dependencies/$", "system.views.dependencies_list"),
)