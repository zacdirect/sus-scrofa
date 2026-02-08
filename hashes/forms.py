# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

import re
from django import forms

from hashes.models import List

# Validation patterns for each hash type
# Format: (regex_pattern, expected_description)
HASH_PATTERNS = {
    "md5":    (r"^([a-fA-F\d]{32})$",  "MD5 hash format: [a-fA-F\\d]{32}"),
    "crc32":  (r"^([a-fA-F\d]{8})$",   "CRC32 hash format: [a-fA-F\\d]{8}"),
    "sha1":   (r"^([a-fA-F\d]{40})$",  "SHA1 hash format: [a-fA-F\\d]{40}"),
    "sha224": (r"^([a-fA-F\d]{56})$",  "SHA224 hash format: [a-fA-F\\d]{56}"),
    "sha256": (r"^([a-fA-F\d]{64})$",  "SHA256 hash format: [a-fA-F\\d]{64}"),
    "sha384": (r"^([a-fA-F\d]{96})$",  "SHA384 hash format: [a-fA-F\\d]{96}"),
    "sha512": (r"^([a-fA-F\d]{128})$", "SHA512 hash format: [a-fA-F\\d]{128}"),
    # Perceptual hashes: hex string, length depends on hash_size (default 8 = 16 hex chars)
    # We accept 8-64 hex chars to accommodate different hash sizes
    "a_hash": (r"^([a-fA-F\d]{8,64})$", "Average Hash: 8-64 hex characters"),
    "p_hash": (r"^([a-fA-F\d]{8,64})$", "Perceptual Hash: 8-64 hex characters"),
    "d_hash": (r"^([a-fA-F\d]{8,64})$", "Difference Hash: 8-64 hex characters"),
    "w_hash": (r"^([a-fA-F\d]{8,64})$", "Wavelet Hash: 8-64 hex characters"),
}


class ListForm(forms.ModelForm):
    """Hash list form."""
    hash_list = forms.FileField(required=True)

    class Meta:
        model = List
        fields = "__all__"
        fields = "__all__"

    def clean_hash_list(self):
        file = self.cleaned_data["hash_list"]
        cipher = self.cleaned_data["cipher"].lower()

        pattern_info = HASH_PATTERNS.get(cipher)
        if not pattern_info:
            raise forms.ValidationError(f"Unknown hash cipher type: {cipher}")

        regex, description = pattern_info

        # Checks file for validation line by line.
        for row in file.readlines():
            # Decode bytes to string if needed (Python 3 compatibility)
            if isinstance(row, bytes):
                row = row.decode('utf-8', errors='ignore')
            
            # Strip whitespace
            row = row.strip()
            
            # Skip comments.
            if row.startswith("#"):
                continue
            # Skip empty lines.
            if len(row) == 0:
                continue
            # Validate against the pattern for this cipher type.
            if not re.match(regex, row):
                raise forms.ValidationError(
                    f"Uploaded file does not meet {description}"
                )

        return file