# Dataset instructions

Use the PhysioNet/CinC Challenge 2016 heart-sound dataset from its official source. Do not commit the recordings to Git.

Expected local layout:

```text
data/raw/
  recordings/
    <recording_id>.wav
  labels.csv
```

`labels.csv` should contain:

```csv
recording_id,label,patient_id,path
a0001,normal,p001,recordings/a0001.wav
a0002,abnormal,p002,recordings/a0002.wav
```

Before training:

1. Record the dataset version and download date.
2. Verify sample rates and file readability.
3. Check duplicate recording and patient identifiers.
4. Split by patient where patient IDs are available; otherwise split by recording.
5. Create fixed-length segments only after the split.
6. Keep `data/raw/` and generated arrays outside version control.

The original challenge label format may need conversion. That conversion script will be added when the exact downloaded archive is inspected.
