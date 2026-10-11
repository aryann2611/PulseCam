# Recording protocol (15–20 consenting volunteers)

1. **Signed consent first** (`docs/consent_form.md`). Scan/photo it into `data/own/consent/` (private; gitignored).
2. Seat the volunteer ~1 m from the camera, face fully visible, no mask or cap (glasses fine). Daylight from a window.
3. Note: ID (P01, P02…), Monk shade (1–10), age group, glasses. **Never use names in file names.**
4. Rest 2 minutes, then record four 1-minute clips at 30 fps:
   | Clip | file suffix | Instruction |
   |---|---|---|
   | Sitting still | `still` | look at the camera |
   | Slow head movement | `move` | turn head slowly left–right |
   | Talking | `talk` | read a paragraph aloud |
   | Dim light | `dim` | lamp off / curtain partly closed |
5. **Camera:** highest quality available, 30 fps; turn OFF beauty filters, HDR, portrait mode (they erase the pulse). Phone: lock exposure/focus if possible. **Never re-encode or "compress for sharing" before analysis:** re-encoding a clean UBFC clip to a default-quality mp4 made POS read 62 BPM instead of 107 (12 Oct 2026 test) because compression smooths away the tiny colour change. Copy the original file as it came off the camera.
6. **Reference HR during each clip:** oximeter or smartwatch reading every 10 s, or one manual 60-second wrist count. State the method in `reference_method`.
7. Save under one name:
   - `data/own/videos/P01_still.mp4`
   - `data/own/videos/P01_still_hr.csv` with header `time_s,hr_bpm` (e.g. `0,72` / `10,74` …; a manual count is a single row)
8. Add one row per clip to `data/own/metadata.csv`:
   `id,monk_shade,age_group,glasses,condition,reference_hr,reference_method`
   e.g. `P01,5,18-25,N,still,73,smartwatch`
9. Tell the volunteer PulseCam is not medical. Store ID→name list separately and privately.

Aim for a spread of Monk shades; recruit early (family, hostel, classmates) and report sample size honestly.
