# Ghiro - Copyright (C) 2013-2015 Ghiro Developers.
# This file is part of Ghiro.
# See the file 'docs/LICENSE.txt' for license terms.

from django.urls import re_path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    re_path(r"^login/$", auth_views.LoginView.as_view(template_name="users/login.html"), name="login"),
    re_path(r"^logout/$", auth_views.LogoutView.as_view(next_page="/"), name="logout"),
    re_path(r"^profile/$", views.profile, name="profile"),
    re_path(r"^password_change/$", auth_views.PasswordChangeView.as_view(template_name="users/password_change.html"), name="password_change"),
    re_path(r"^password_change_done/$", auth_views.PasswordChangeDoneView.as_view(template_name="users/password_change_done.html"), name="password_change_done"),

    # Admin
    re_path(r"^management/index/$", views.admin_list_users, name="admin_list_users"),
    re_path(r"^management/activity/$", views.admin_list_activity, name="admin_list_activity"),
    re_path(r"^management/new/$", views.admin_new_user, name="admin_new_user"),
    re_path(r"^management/show/(?P<user_id>[\d]+)/$", views.admin_show_user, name="admin_show_user"),
    re_path(r"^management/show/(?P<user_id>[\d]+)/activity/$", views.admin_show_activity, name="admin_show_activity"),
    re_path(r"^management/edit/(?P<user_id>[\d]+)/$", views.admin_edit_user, name="admin_edit_user"),
    re_path(r"^management/disable/(?P<user_id>[\d]+)/$", views.admin_disable_user, name="admin_disable_user"),
]