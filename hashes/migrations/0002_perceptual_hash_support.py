# Generated migration for perceptual hash support

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hashes', '0001_initial'),
    ]

    operations = [
        # Widen cipher field to accommodate perceptual hash type names (e.g. "a_hash")
        migrations.AlterField(
            model_name='list',
            name='cipher',
            field=models.CharField(
                choices=[
                    ('sha1', 'SHA1'), ('sha224', 'SHA224'), ('sha384', 'SHA384'),
                    ('crc32', 'CRC32'), ('sha256', 'SHA256'), ('sha512', 'SHA512'),
                    ('md5', 'MD5'),
                    ('a_hash', 'Average Hash'), ('p_hash', 'Perceptual Hash'),
                    ('d_hash', 'Difference Hash'), ('w_hash', 'Wavelet Hash'),
                ],
                db_index=True,
                default='MD5',
                max_length=10,
            ),
        ),
        # Add distance threshold for perceptual hash matching
        migrations.AddField(
            model_name='list',
            name='distance_threshold',
            field=models.IntegerField(
                default=0,
                help_text='Hamming distance threshold for perceptual hash matching (0=exact, 5=near-duplicate, 10=similar)',
            ),
        ),
    ]
