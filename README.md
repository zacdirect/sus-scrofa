Sus Scrofa is a fork of a long dormant project [Ghiro](https://github.com/ghirensics/ghiro).

It is being modernized and moved from its strictly forensic analysis roots to something more generally purposed for determining what's sus about images or a collection of images.

Sometimes forensic investigators need to process digital images as evidence.
There are some tools around, otherwise it is difficult to deal with forensic
analysis with lots of images involved.
Images contain tons of information, Sus Scrofa extracts this information from
provided images and displays them in a nicely formatted report.
Dealing with tons of images is pretty easy, Sus Scrofa is designed to scale to
support gigs of images.
All tasks are totally automated, you have just to upload your images and let
Sus Scrofa do the work.
Understandable reports, and great search capabilities allow you to find a
needle in a haystack.
Sus Scrofa is a multi-user environment, different permissions can be assigned to each
user.
Cases allow you to group image analysis by topic, you can choose which user
can see your case with a permission schema. 

Community and reports
---------------------

[Coming Soon?](https://duckduckgo.com/?q=Katsuhiro+Harada+shirt&iar=images)

GPU/CUDA Detection
------------------

Sus Scrofa uses unified GPU/CUDA detection to automatically optimize AI/ML installations.

**Check your system:**
```bash
make detect-system
```

This will:
- Detect NVIDIA GPU hardware
- Check driver installation
- Identify CUDA version
- Recommend CPU or GPU PyTorch installation

**AI/ML Setup:**
```bash
make photoholmes-setup    # Photoholmes forgery detection
make ai-setup             # AI Detection (SPAI, SDXL)
```

Both automatically install appropriate PyTorch version:
- **CPU-only**: ~190MB (no GPU detected)
- **CUDA-enabled**: ~2GB+ (GPU + drivers detected)

**Documentation:** See `docs/GPU_DETECTION.md` for details.

