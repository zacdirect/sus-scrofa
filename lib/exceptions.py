# SusScrofa - Copyright (C) 2013-2016 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

class GhiroException(Exception):
    """Base SusScrofa exception."""
    pass

class GhiroValidationException(GhiroException):
    """Validation error."""
    pass

class GhiroPluginException(GhiroException):
    """An error occurred when running the plugin."""
    pass