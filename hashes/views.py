# Sus Scrofa - Copyright (C) 2026 Sus Scrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_safe
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse

import hashes.forms as forms
from hashes.models import List, Hash
from sus_scrofa.common import log_activity

@require_safe
@login_required
def list_hashes(request):
    """Hash list index."""

    my_lists = List.objects.filter(owner=request.user)
    public_lists = List.objects.filter(public=True)

    # Set sidebar active tab.
    request.session["sidebar_active"] = "side-hashes"

    return render(request, "hashes/index.html",
                              {"my_lists": my_lists, "public_lists": public_lists})

@login_required
def new_hashes(request):
    """New hash list."""

    if request.method == "POST":
        form = forms.ListForm(request.POST, request.FILES)

        if form.is_valid():
            list = form.save(commit=False)
            list.owner = request.user
            list.save()
            # Read file.
            with open(request.FILES["hash_list"].temporary_file_path(), "r", encoding='utf-8', errors='ignore') as file:
                for row in file.readlines():
                    row = row.strip()
                    # Skip comments and empty lines
                    if row.startswith("#") or len(row) == 0:
                        continue
                    Hash.objects.get_or_create(value=row, list=list)

            # Auditing.
            log_activity("H",
                         "Created new hash list %s" % list.name,
                         request)

            return HttpResponseRedirect(reverse("show_hashes", args=(list.id,)))
    else:
        form = forms.ListForm()

    return render(request, "hashes/new.html",
                              {"form": form})

@require_safe
@login_required
def show_hashes(request, list_id):
    hash_list = get_object_or_404(List, pk=list_id)

    # Security check.
    if request.user != hash_list.owner:
        return render(request, "error.html",
                                  {"error": "You are not authorized to show this."})

    return render(request, "hashes/show.html",
                              {"hash_list": hash_list})

@require_safe
@login_required
def delete_hashes(request, list_id):
    hash_list = get_object_or_404(List, pk=list_id)

    # Security check.
    if request.user != hash_list.owner:
        return render(request, "error.html",
                                  {"error": "You are not authorized to delete this."})

    hash_list.delete()

    # Auditing.
    log_activity("H",
                 "Deleted hash list %s" % hash_list.name,
                 request)

    return HttpResponseRedirect(reverse("list_hashes"))