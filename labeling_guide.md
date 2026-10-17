# Labeling guide

The app receives **one phone photo taken by a junior engineer standing on the ground, usually on
a road or in a valley, looking up or across at a slope** (a "downhill" view). Every label follows
from that: we only train and test on photos that could have come from such a phone.

## Step 1: is the photo usable? (otherwise label `reject`)

Keep (go to step 2):
- Taken from the ground by a person: from a road, path, valley floor, beach or riverbank,
  looking **up** or **across** at the slope. This is the main case.
- Taken on the ground from the side or from the top of the slide looking down (still a person
  on foot). Record the viewpoint as `down`; these are used for training but reported separately.

Reject (`R` in review.py):
- Aerial, drone, helicopter or aeroplane view; satellite image; map, diagram, chart, painting,
  drawing, poster, screenshot.
- Close-up of rocks or soil with no slope visible (cannot tell what kind of movement it was).
- Slope too small or too far away to judge (roughly more than ~1 km away, e.g. scenic views).
- Glacier or ice dominates the scene.
- People, vehicles or buildings dominate and the slope is barely visible.
- Heavily edited (text overlays, collages, side-by-side before/after), stereo cards, and
  black-and-white or framed archive scans (phones take colour photos).

## Step 2: what is the main slope feature?

Pick the **dominant** visible process. If two are visible, pick the one that covers most of the
photo or caused the damage in the foreground.

| Key | Class | Choose when you see | Do not confuse with |
|---|---|---|---|
Hungr et al. (2014) types per class (same as `src/common.py`):
fall_topple = rock/ice fall, boulder/debris fall, rock block topple, flexural topple;
rotational_slide = rock/clay/silt rotational slide; translational_slide = rock planar/wedge slide,
gravel/sand/debris slide; debris_flow = debris flow, debris flood, debris avalanche;
earth_mud_flow = earthflow, mud flow; no_landslide = no visible slope failure.

| 1 | `fall_topple` | steep rock face or cliff + loose blocks, boulders or talus at its foot or on the road; blocks clearly fell/rolled | translational rock slide (a slab slid on a plane) |
| 2 | `rotational_slide` | curved, spoon-shaped scarp; displaced block tilted back; stepped or terraced head; bulging toe; tilted trees; cracked road with vertical drop | flow (no coherent block) |
| 3 | `translational_slide` | shallow, elongated scar parallel to the slope; flat sliding surface (bedding/joint plane or soil-rock contact) exposed; sheet of soil/debris or rock slabs moved down without rotating. Most monsoon road-side "landslides" on Himalayan cut slopes are this | fall (no sliding plane, blocks fell freely) |
| 4 | `debris_flow` | material came down a gully/channel; unsorted boulders in mud; levees; fan at the gully mouth; buried road/bridge at a stream crossing | earth/mud flow (fine, few boulders) |
| 5 | `earth_mud_flow` | fine-grained wet soil/mud as a lobe or tongue; smooth or hummocky surface; few large boulders; coastal mudslides | debris flow (coarse, channelised) |
| 6 | `no_landslide` | stable slope, stable road cut, cliff with no fresh failure, quarry face, construction cut, protection works (nets, rock sheds, walls) without a visible failure | old revegetated scar: label the failure only if it is still clearly recognisable |

## Rules
- Label what is **visible**, not what the title or caption says. Captions may help to understand
  the scene but cannot add features you cannot see.
- If you cannot decide between two landslide types after a careful look, press **R** (reject).
  This is the rule all existing labels follow (it keeps the type labels clean). Its cost: the
  hardest photos are not in the scored set, so the app's behaviour on them is measured
  separately (`tools/hard_case_eval.py`, `results/hard_cases.md`).
- Test-split photos are always labeled without the VLM suggestion shown.
- Viewpoint is recorded with each label: `up` (from below or level, the deployment case; the
  default) or `down` (on the ground, from above: press **V** in review.py before the class key);
  blank for `reject`.
