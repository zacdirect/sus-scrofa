# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r"^dependencies/$", views.dependencies_list, name="dependencies_list"),
]