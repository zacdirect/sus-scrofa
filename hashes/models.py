# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

from django.db import models

from analyses.models import Profile, Analysis

class List(models.Model):
    """Hashes list."""
    # Cryptographic hash types
    CRYPTO_CIPHERS = (
            ("sha1", "SHA1"),
            ("sha224", "SHA224"),
            ("sha384", "SHA384"),
            ("crc32", "CRC32"),
            ("sha256", "SHA256"),
            ("sha512", "SHA512"),
            ("md5", "MD5"),
    )
    # Perceptual hash types (distance-based matching)
    PERCEPTUAL_CIPHERS = (
            ("a_hash", "Average Hash"),
            ("p_hash", "Perceptual Hash"),
            ("d_hash", "Difference Hash"),
            ("w_hash", "Wavelet Hash"),
    )
    CIPHERS = CRYPTO_CIPHERS + PERCEPTUAL_CIPHERS

    PERCEPTUAL_CIPHER_KEYS = {c[0] for c in PERCEPTUAL_CIPHERS}

    name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)
    cipher = models.CharField(max_length=10, choices=CIPHERS, db_index=True, null=False, blank=False, default="MD5")
    public = models.BooleanField(default=False)
    owner = models.ForeignKey(Profile, null=False, blank=False, on_delete=models.CASCADE, db_index=True, editable=False, related_name="owned_hashes")
    created_at = models.DateTimeField(auto_now_add=True, editable=False, db_index=True)
    matches = models.ManyToManyField(Analysis, blank=True)
    # Distance threshold for perceptual hash matching (0 = exact only)
    distance_threshold = models.IntegerField(default=0, help_text="Hamming distance threshold for perceptual hash matching (0=exact, 5=near-duplicate, 10=similar)")

    def is_owner(self, user):
        """Checks if a user is the owner of this object.
        @param user: user instance
        @return: boolean permission
        """
        return user == self.owner

    def can_read(self, user):
        """Checks if a user is allowed to read this object.
        @param user: user instance
        @return: boolean permission
        """
        return user.is_superuser or self.is_owner(user) or self.public

    def can_write(self, user):
        """Checks if a user is allowed to write (create, edit, delete) this object.
        @param user: user instance
        @return: boolean permission
        """
        return user.is_superuser or self.is_owner(user)

    @property
    def is_perceptual(self):
        """Whether this list uses perceptual (distance-based) hashing."""
        return self.cipher in self.PERCEPTUAL_CIPHER_KEYS

class Hash(models.Model):
    """Hashes."""
    value = models.CharField(max_length=255)
    list = models.ForeignKey(List, null=False, blank=False, on_delete=models.CASCADE, db_index=True, editable=False)