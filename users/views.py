# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_safe
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from users.models import Profile, Activity
from sus_scrofa.common import log_activity

import users.forms as forms

@require_safe
@login_required
def profile(request):
    """User's profile."""
    # Set sidebar no active tab.
    request.session["sidebar_active"] = None

    return render(request, "users/profile.html")

@require_safe
@login_required
def admin_list_users(request):
    """Show users list."""

    # Set sidebar active tab.
    request.session["sidebar_active"] = "side-admin"

    # Security check.
    if not request.user.is_superuser:
        return render(request, "error.html",
                                  {"error": "You must be superuser"})

    users = Profile.objects.all()

    return render(request, "admin/index.html",
                              {"users": users})

@login_required
def admin_new_user(request):
    """Create new users."""

    # Security check.
    if not request.user.is_superuser:
        return render(request, "error.html",
                                  {"error": "You must be superuser"})

    if request.method == "POST":
        form = forms.ProfileCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Auditing.
            log_activity("A",
                         "Created new user %s" % user.username,
                         request)
            return HttpResponseRedirect(reverse("admin_show_user", args=(user.id,)))
    else:
        form = forms.ProfileCreationForm()

    return render(request, "admin/new_user.html",
                              {"form": form})

@login_required
def admin_show_user(request, user_id):
    """Show user details."""

    # Security check.
    if not request.user.is_superuser:
        return render(request, "error.html",
                                  {"error": "You must be superuser"})

    user = get_object_or_404(Profile, pk=user_id)

    return render(request, "admin/show_user.html",
                              {"user": user})

@login_required
def admin_edit_user(request, user_id):
    """Edit user."""

    # Security check.
    if not request.user.is_superuser:
        return render(request, "error.html",
                                  {"error": "You must be superuser"})

    user = get_object_or_404(Profile, pk=user_id)

    if request.method == "POST":
        form = forms.ProfileForm(request.POST, instance=user)
        if form.is_valid():
            user = form.save()
            # Auditing.
            log_activity("A",
                         "Edited user %s" % user.username,
                         request)
            return HttpResponseRedirect(reverse("admin_show_user", args=(user.id,)))
    else:
        form = forms.ProfileForm(instance=user)

    return render(request, "admin/edit_user.html",
                              {"form": form, "user": user})

@login_required
def admin_disable_user(request, user_id):
    """Disable user."""

    # Security check.
    if not request.user.is_superuser:
        return render(request, "error.html",
                                  {"error": "You must be superuser"})

    user = get_object_or_404(Profile, pk=user_id)

    if request.user == user:
        return render(request, "error.html",
                                  {"error": "You can not disable yourself"})

    user.is_active = False
    user.save()
    # Auditing.
    log_activity("A",
                 "Disabled user %s" % user.username,
                 request)

    return HttpResponseRedirect(reverse("admin_list_users"))

@login_required
def admin_list_activity(request):
    """Activity index."""

    # Security check.
    if not request.user.is_superuser:
        return render(request, "error.html",
                                  {"error": "You must be superuser"})

    activities = Activity.objects.all()

    return render(request, "admin/activity.html",
                              {"activities": activities})

@login_required
def admin_show_activity(request, user_id):
    """Show user activity."""

    # Security check.
    if not request.user.is_superuser:
        return render(request, "error.html",
                                  {"error": "You must be superuser"})

    user = get_object_or_404(Profile, pk=user_id)
    activities = Activity.objects.filter(user=user)

    return render(request, "admin/user_activity.html",
                              {"activities": activities})