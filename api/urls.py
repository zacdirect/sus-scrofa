# Ghiro - Copyright (C) 2013-2015 Ghiro Developers.
# This file is part of Ghiro.
# See the file 'docs/LICENSE.txt' for license terms.

from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r"^cases/new$", views.new_case),
    re_path(r"^images/new$", views.new_image),
]
