# How to film a scene so the point cloud comes out right

The pipeline turns each keyframe's depth into a sheet of 3D points and stacks the sheets in one world. The sheets only land
on the same surface if the camera poses are right, and poses are only reliable when **neighbouring moments of the video show
mostly the same things from a slightly different position**. Everything below serves that one rule.

## What we measured (two videos, same pipeline)

| | Hotel room (slow pan) | Mouse on a tablecloth (fast orbit) |
|---|---|---|
| Rotation between neighbouring keyframes | 2.8 deg median | 7.8 deg median, 26.7 deg for the worst 10% |
| Feature matches shared by neighbours | 929 (fewest 546) | 134 (fewest 22) |
| Depth agreeing between neighbouring views | 99% | 36% (84% after filming denser keyframes) |
| Result | coherent room cloud | pile of tilted sheets |

## Do

1. **Move slowly and smoothly.** The hotel clip turned roughly 110 degrees in 14 s (about 8 deg/s) and worked. The mouse clip
   swung far more per keyframe. If in doubt, go slower than feels natural.
2. **Keep at least 70-80% of the picture the same from one moment to the next.** The pipeline's default target is that 80% of a
   keyframe's tracked features are still shared with the next keyframe.
3. **Film one continuous pass over the whole area.** Finish where you started (a loop) if you can. This is standard practice
   against drift; the pipeline now uses it (loop closure between keyframes that revisit a place), but it cannot rescue a capture whose
   neighbouring frames share too little to track.
4. **Film a scene with structure and varied texture**: walls, furniture, objects at different distances, distinct details.
5. **Avoid a flat, repeating pattern filling the frame** (checked cloth, tiles, blinds, fences). Look-alike corners confuse
   matching. This is our best explanation for the mouse clip failing, not a tested proof. If the subject sits on such a
   surface, put a few distinct, non-repeating objects around it or use a plain surface.
6. **Small objects: orbit slowly at a constant distance**, keep the whole object in view, make two slow loops at different
   heights, and do not zoom.
7. **Include some sideways movement, not only turning on the spot.** Pure rotation gives no parallax. (The hotel pan moved the
   camera only about 13% of the scene depth in total, which depth estimation tolerated, but triangulating edges across
   views is weak with so little baseline.)
8. **Keep frames sharp**: good light, steady hands, no motion blur. Keep zoom, focus and exposure fixed if the phone lets you.
   (General practice.)
9. **Keep the scene still**: no people or pets moving, no large mirrors or windows dominating. (General practice.)

## Resolution and length

1080p is plenty (the hotel clip was 1080x1920; the mouse clip 1280x720). 15-60 s is typical. Longer clips cost more time:
the pipeline took about 5 minutes on a 12-14 s clip (stages 1-10, RTX 500 Ada, 4 GB).

## Let the pipeline tell you whether the capture is good

```
python run_pipeline.py --video path/to/video.mp4 --stages 1-3      # quick check of the capture only (about 1-3 minutes)
python run_pipeline.py --video path/to/video.mp4                   # the whole thing; the capture check runs automatically
```

It chooses the number of keyframes itself (a new one whenever the shared content drops below the overlap target), measures
the result, and retries with stricter overlap if needed. It prints a verdict and advice, and repeats them in `summary.json`:

| Verdict | Meaning (depth consistency, refined pose error) | Calibrated on |
|---|---|---|
| GOOD | at least 95% and at most 1.0 px | hotel: 99%, 0.24 px |
| FAIR | at least 80% and at most 4.0 px | mouse with 150 keyframes: 84%, 3.1 px |
| POOR | otherwise | mouse with 40 keyframes: 36%, 19 px |

Those thresholds come from two videos only. If the verdict is POOR after the retries, **re-film following the printed advice**
(slower, more overlap, more structure) instead of expecting later stages to repair it.
