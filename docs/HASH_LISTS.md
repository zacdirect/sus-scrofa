# Hash Lists - Watchlist Matching

Hash lists allow you to upload curated databases of known image hashes to automatically flag matching images during analysis. This is essential for forensic investigations, OSINT, brand protection, and content moderation.

## Overview

**Hash Lists** = External watchlists you create and maintain  
**Cross-Case Linking** = Automatic matching across your own case database

Both features work together to provide comprehensive image correlation:
- Hash lists match against **external intelligence** (law enforcement databases, watchlists)
- Cross-case linking finds **internal connections** (same image appearing in multiple investigations)

## Use Cases

### Law Enforcement & Forensics
- **CSAM Detection**: Match against NCMEC or ICMEC hash databases
- **Terrorist Propaganda**: Flag known extremist imagery
- **Evidence Correlation**: Link images across different investigations
- **Missing Persons**: Identify photos of persons of interest

### Corporate & Brand Protection
- **Leaked Assets**: Detect unauthorized distribution of confidential images
- **Copyright Enforcement**: Flag stolen branded content
- **Product Counterfeiting**: Identify fake product images
- **Employee Misconduct**: Match against internal incident databases

### OSINT & Research
- **Profile Tracking**: Identify same person across platforms using profile photos
- **Stock Image Detection**: Flag generic/fake content from stock libraries
- **Reverse Image Search**: Build custom databases of notable imagery
- **Propaganda Analysis**: Track distribution of disinformation campaigns

### Deduplication
- **Previously Analyzed**: Skip redundant analysis of known images
- **Batch Processing**: Identify duplicates before expensive ML inference

## Hash Types

### Cryptographic Hashes (Exact Match Only)
Perfect for identifying **identical** files:

| Hash Type | Length | Use Case |
|-----------|--------|----------|
| **MD5** | 32 hex | Legacy databases (fast but weak) |
| **SHA1** | 40 hex | Git-style versioning |
| **SHA256** | 64 hex | **Recommended** - secure and standard |
| **SHA512** | 128 hex | Maximum security |
| **CRC32** | 8 hex | Quick integrity checks |
| **SHA224/384** | 56/96 hex | Niche applications |

**Limitation**: Even 1-pixel change or JPEG re-encoding breaks the match.

### Perceptual Hashes (Fuzzy Matching)
Detect **variations** of the same image:

| Hash Type | Detects | Resistant To |
|-----------|---------|--------------|
| **pHash** (Perceptual) | Crops, resizing, rotation | Geometric transforms |
| **aHash** (Average) | Brightness/contrast changes | Color adjustments |
| **dHash** (Difference) | Edge manipulation | Content-preserving edits |
| **wHash** (Wavelet) | JPEG re-compression | Lossy encoding artifacts |

**Distance Thresholds**:
- `0` = Exact perceptual match (identical visual content)
- `≤5` = Near-duplicate (minor edits, re-encoding)
- `≤10` = Similar (filters, color shift, moderate crop)
- `≤15` = Related (heavy editing, composites)

## Creating Hash Lists

### File Format
Plain text file, one hash per line:

```
# Known CSAM Database (NCMEC)
abc123def456789... 
789def012abc345...

# Comments start with # and are ignored
# Blank lines are also skipped

fed654cba987654...
```

### Web Interface
1. Navigate to **Hashes** in sidebar
2. Click **"New Hash List"**
3. Fill form:
   - **Name**: "NCMEC CSAM Database 2026-02"
   - **Description**: "Child exploitation imagery hashes from NCMEC public database"
   - **Cipher**: Select hash type (SHA256, pHash, etc.)
   - **Distance Threshold** (perceptual only): `0` = exact, `5` = near-duplicate
   - **Public**: Check to share with other users (optional)
   - **File**: Upload `.txt` file with hashes
4. Click **"Submit"**

### Permissions
- **Owner**: Can view, edit, delete
- **Public Lists**: Viewable by all users (created by admins)
- **Private Lists**: Only owner can access

Hash lists are **per-user** by default. Administrators can mark lists as public for organization-wide sharing.

## How Matching Works

### Analysis Pipeline
When an image is analyzed:

1. **Compute all hashes** (crypto + perceptual) - see [plugins/static/perceptual_hash.py](../plugins/static/perceptual_hash.py)
2. **Check uploaded hash lists**:
   - Compare against lists matching the hash type
   - For perceptual hashes, calculate Hamming distance
   - Match if distance ≤ configured threshold
3. **Check cross-case database** (automatic internal linking)
4. **Report matches** in "Key Indicators" section

### Performance
- **Cryptographic**: O(1) hash table lookup - instant
- **Perceptual**: O(n) distance calculations per list
  - Small lists (<1000): negligible overhead
  - Large lists (>10,000): consider segmenting by category

### Match Recording
When a hash list match occurs:
- Badge appears in analysis report: **"Perceptual Hash List Match"**
- Shows list name(s), match count, and distance
- Match is recorded on the hash list model (`hash_list.matches` relation)
- Users can view all matching images on the hash list page

## Integration with Cross-Case Linking

Both features use the **same perceptual hash algorithms** and appear together in analysis reports:

```
Key Indicators:
┌─────────────────────────────────┬──────────────────────────────────┐
│ Perceptual Hash List Match      │ 1 match: "Known CSAM Database"   │
│ Cross-Case Image Link           │ 3 exact + 2 near-duplicate       │
│                                  │ matches across 2 other cases     │
└─────────────────────────────────┴──────────────────────────────────┘
```

**Difference**:
- **Hash Lists**: Match against YOUR uploaded external watchlists
- **Cross-Case**: Match against ALL images in YOUR database automatically

Together they provide complete coverage:
1. "Is this a **known bad** image?" → Hash Lists
2. "Have **I seen this** before?" → Cross-Case
3. "Where else does it appear?" → Both

## Security & Privacy

### Data Protection
- Hash lists are **stored in MongoDB** (not filesystem)
- Access control enforced per-user with public sharing option
- No image data stored - only hash values
- Hashes are one-way: cannot reconstruct image from hash

### Operational Security
- Use **SHA256 or higher** for secure identification
- **Never share CSAM hash lists** outside secure channels
- Mark sensitive lists as **private** (unchecked public box)
- Regularly update lists from authoritative sources
- Log hash list uploads in audit trail (automatic)

### Legal Considerations
- Hash matching is **legally sound** for identification (no privacy violation)
- Commonly accepted in court as evidence linking
- NCMEC and ICMEC provide free hash databases for LEO
- Consult legal counsel for jurisdiction-specific requirements

## Example Workflows

### Scenario 1: Child Exploitation Investigation
```
1. Obtain NCMEC hash database (SHA256 hashes)
2. Upload as "NCMEC-2026-Q1" hash list (private)
3. Analyze suspect's seized device images
4. System flags 47 matches → immediate evidence for warrant
5. Cross-case linking shows same images in 3 other investigations
   → Connects trafficking network
```

### Scenario 2: Brand Protection
```
1. Hash all official product photos (pHash with threshold=5)
2. Upload as "Authentic Product Catalog" (public)
3. Monitor e-commerce scrapes for counterfeit listings
4. Flagged images show near-duplicates with altered logos
5. Builds case for trademark enforcement action
```

### Scenario 3: OSINT Profile Correlation
```
1. Collect profile photos from known extremist accounts
2. Hash with pHash for variations (threshold=10)
3. Upload as "ISIS Propagandists" watchlist
4. Scan new social media accounts
5. Match links new account to known operator despite photo editing
```

## Troubleshooting

### No Matches Despite Known Duplicates
- **Cryptographic hashes**: Even minor changes break exact match - use perceptual hashes instead
- **Perceptual threshold too low**: Try increasing from 0 → 5 → 10
- **Wrong hash type**: Verify uploaded hashes match selected cipher type
- **Format errors**: Check file uses one hash per line (no commas, spaces)

### Slow Analysis
- **Too many hash lists**: Each list adds comparison overhead
- **Very large lists**: Break into smaller categorical lists (e.g., by year/category)
- **High thresholds**: Lower threshold = faster rejection of non-matches

### False Positives
- **Threshold too high**: Reduce from 15 → 10 → 5
- **Generic content**: Stock images may match unrelated images - use descriptive list names
- **Wrong hash algorithm**: Try different perceptual hash type (pHash vs aHash)

## Best Practices

1. **Use descriptive names**: "NCMEC-2026-Q1" not "List1"
2. **Document sources**: Add description with origin, date, authority
3. **Version your lists**: Include date/version in name for updates
4. **Separate by sensitivity**: Keep CSAM separate from general watchlists
5. **Start conservative**: Begin with threshold=0, increase if needed
6. **Regular updates**: Refresh from authoritative sources quarterly
7. **Test before deployment**: Upload small test list first
8. **Monitor match rates**: High match rate may indicate overly broad list

## Related Documentation

- [ENGINE_ARCHITECTURE.md](ENGINE_ARCHITECTURE.md) - Analysis pipeline overview
- [PLUGIN_CONTRACT.md](PLUGIN_CONTRACT.md) - How plugins expose hash data
- [plugins/static/perceptual_hash.py](../plugins/static/perceptual_hash.py) - Implementation details
