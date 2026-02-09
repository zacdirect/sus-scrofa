# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r"^cases/$", views.list_cases, name="list_cases"),
    re_path(r"^cases/new/", views.new_case, name="new_case"),
    re_path(r"^cases/edit/(?P<case_id>[\d]+)/", views.edit_case, name="edit_case"),
    re_path(r"^cases/close/(?P<case_id>[\d]+)/", views.close_case, name="close_case"),
    re_path(r"^cases/delete/(?P<case_id>[\d]+)/", views.delete_case, name="delete_case"),
    re_path(r"^cases/reprocess/(?P<case_id>[\d]+)/$", views.reprocess_case, name="reprocess_case"),
    re_path(r"^cases/show/(?P<case_id>[\d]+)/(?P<page_name>[\w]+)/", views.show_case, name="show_case"),
    re_path(r"^cases/(?P<case_id>[\d]+)/images/new/$", views.new_image, name="new_image"),
    re_path(r"^cases/(?P<case_id>[\d]+)/folder/new/$", views.new_folder, name="new_folder"),
    re_path(r"^cases/(?P<case_id>[\d]+)/url/new/$", views.new_url, name="new_url"),
    re_path(r"^cases/count_ajax/(?P<case_id>[\d]+)/(?P<analysis_id>[\d]+)/$", views.count_new_analysis, name="count_new_analysis"),
    re_path(r"^show/(?P<analysis_id>[\d]+)/$", views.show_analysis, name="show_analysis"),
    re_path(r"^show/(?P<analysis_id>[\d]+)/hexdump/$", views.hex_dump, name="hex_dump"),
    re_path(r"^delete/(?P<analysis_id>[\d]+)/$", views.delete_analysis, name="delete_analysis"),
    re_path(r"^reprocess/(?P<analysis_id>[\d]+)/$", views.reprocess_analysis, name="reprocess_analysis"),
    re_path(r"^images/(?P<page_name>[\w]+)/$", views.list_images, name="list_images"),
    re_path(r"^images/file/(?P<id>[\d\w]+)/$", views.image, name="image"),
    re_path(r"^images/favorite/(?P<id>[\d\w]+)/$", views.favorite, name="favorite"),
    re_path(r"^images/comment/(?P<id>[\d]+)/$", views.add_comment, name="add_comment"),
    re_path(r"^images/comment/delete/(?P<id>[\d]+)/$", views.delete_comment, name="delete_comment"),
    re_path(r"^search/(?P<page_name>[\w]+)/$", views.search, name="search"),
    re_path(r"^images/tag/add/(?P<id>[\d]+)/$", views.add_tag, name="add_tag"),
    re_path(r"^images/tag/delete/(?P<id>[\d]+)/$", views.delete_tag, name="delete_tag"),
    re_path(r"^show/(?P<analysis_id>[\d]+)/report/(?P<report_type>[\w]+)/$", views.static_report, name="static_report"),
    re_path(r"^show/(?P<analysis_id>[\d]+)/export/json/$", views.export_json, name="export_json"),
]