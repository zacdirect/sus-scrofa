# Ghiro - Copyright (C) 2013-2015 Ghiro Developers.
# This file is part of Ghiro.
# See the file 'docs/LICENSE.txt' for license terms.

from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r"^dependencies/$", views.dependencies_list, name="dependencies_list"),
]