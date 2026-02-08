# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

from django.urls import include, re_path
from analyses import views as analyses_views

urlpatterns = [
    re_path(r"^$", analyses_views.dashboard),
    re_path(r"^users/", include("users.urls")),
    re_path(r"^analyses/", include("analyses.urls")),
    re_path(r"^hashes/", include("hashes.urls")),
    re_path(r"^manage/", include("manage.urls")),
    re_path(r"^api/", include("api.urls")),
]
