---
title: Landslide Type Classifier
emoji: ⛰️
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Landslide type from a ground photo, with field guidance
---

# Landslide type classifier (online demo)

Take or upload a ground-level photo of a slope; the app names the landslide type (Hungr et al. 2014), shows how
sure it is, grades the photo quality and guides the next steps in the field (English / Hindi).

- This free demo runs on 2 CPU cores: a photo takes about 15-20 s, longer when several people use it at once.
- **Size estimates are switched off here**: on a CPU one estimate takes several minutes. Run the app on a laptop
  with a GPU for sizes (see the GitHub repository).
- Uploaded photos are not stored.
- The first visit after a quiet period wakes the app, which takes a minute or two.

Code, method and evaluation: https://github.com/DhmalTPS/Landslide-Category-Classifier
