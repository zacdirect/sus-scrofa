# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r"^index/$", views.list_hashes, name="list_hashes"),
    re_path(r"^new/", views.new_hashes, name="new_hashes"),
    re_path(r"^show/(?P<list_id>[\d]+)/", views.show_hashes, name="show_hashes"),
    re_path(r"^delete/(?P<list_id>[\d]+)/", views.delete_hashes, name="delete_hashes"),
]