# Instructions given to AI annotators (reviewer `claude_ai`)

Kept here so the report can state exactly how the provisional AI labels were produced.
The same rules as `labeling_guide.md`, plus the conventions below.

## Task
You label landslide photos for a classifier that will receive **phone photos taken by an
engineer standing on the ground, usually on a road or valley floor, looking up/across at a
slope**. You get contact sheets (`sheet_###.jpg`, 4x3 numbered tiles, made with
`python tools/contact_sheet.py OUT --cols 4 --rows 3 --tile 320 [--ids ids.txt | --skip-labeled]`) and title lists
(`sheet_###.txt`, `cell: image_id | title`). Look at every tile, read its title, and write one
line per sheet into your codes file:

```
003 0f 1x 2tv 3n 4x 5x 6r 7d 8e 9n 10x 11f
```
Codes: `f` fall_topple, `r` rotational_slide, `t` translational_slide, `d` debris_flow,
`e` earth_mud_flow, `n` no_landslide, `x` reject. Append `v` when the (usable) photo is taken
on the ground but looking **down** onto the feature from above (e.g. from a cliff top).

## Reject (`x`)
- aerial / drone / helicopter / satellite / map / diagram / painting / postcard / stereo card /
  archival scan with frame or handwriting / black-and-white archive photo
- signs, tunnels, galleries or buildings dominate and the slope is barely visible
- glaciers, volcanoes, waterfalls, scenic landscapes where the slope is far away (> ~1 km) or tiny
- close-ups of rock/soil without the slope (cannot tell the movement type)
- people/vehicles dominate; heavily blurred; collages

## Class conventions (what we decided on ambiguous cases)
- **fall_topple:** talus/boulders/blocks at the foot of a cliff or on a road/beach, fresh light
  scar on a rock face with debris below, a single fallen boulder on a road with its source slope
  visible, rock-fall chutes with debris cones. Title says "rockfall" but nothing visible -> `n`.
- **translational_slide:** shallow elongated scar on a vegetated/soil slope (often exposing soil
  or rock beneath), sheet of soil/debris slid onto a road from a cut slope, rock slabs that slid
  along a bedding plane, large rock-slide scar + blocky deposit (e.g. Marocche di Dro type)
  when the source scar is visible.
- **rotational_slide:** curved/spoon-shaped scarp, back-tilted turf/blocks, stepped slumps,
  slumped river-bank blocks with tilted trees, curved crack/drop in a road.
- **debris_flow:** channel/gully + unsorted boulders in mud, levees, fan at a gully mouth,
  buried road/bridge at a stream crossing, lahar deposits.
- **earth_mud_flow:** fine wet soil/mud lobe or tongue, thaw-slump mud flows, coastal mudslides
  with a wet muddy lobe, few large boulders.
- **no_landslide:** stable cliffs, crags, road cuts (with or without nets/fences/barriers),
  quarry faces, slopes with protection works and no visible failure, outcrops.
- Label what is **visible**; the title can help interpret the scene but cannot add features.
- When unsure between two landslide types after a careful look, use `x` rather than guessing
  (clean data matters more than quantity).
