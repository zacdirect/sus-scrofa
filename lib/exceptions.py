# Sus Scrofa - Copyright (C) 2026 Sus Scrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

class SusScrofaException(Exception):
    """Base SusScrofa exception."""
    pass

class SusScrofaValidationException(SusScrofaException):
    """Validation error."""
    pass

class SusScrofaPluginException(SusScrofaException):
    """An error occurred when running the plugin."""
    pass