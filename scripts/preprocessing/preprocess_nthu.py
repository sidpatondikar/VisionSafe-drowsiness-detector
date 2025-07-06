import os
import csv

NTHU_DIR = 'data/nthu'
OUTPUT_CSV = os.path.join(NTHU_DIR, 'nthu_metadata.csv')

def extract_metadata(filename):
    label = 1 if filename.endswith('_drowsy.jpg') else 0
    parts = filename.split('_')
    frame_id = parts[-2] if label == 1 else parts[-2]
    return {
        'filename': filename,
        'label': label,
        'glasses': parts[1],
        'state': parts[2],
        'frame_id': frame_id
    }

if __name__ == "__main__":
    rows = []
    for subdir in ['drowsy', 'notdrowsy']:
        path = os.path.join(NTHU_DIR, subdir)
        for fname in os.listdir(path):
            if fname.endswith('.jpg'):
                meta = extract_metadata(fname)
                rows.append(meta)

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'label', 'glasses', 'state', 'frame_id'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Metadata written to {OUTPUT_CSV}")
