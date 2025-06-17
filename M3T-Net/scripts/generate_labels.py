import os
import csv

base_dir = '/Volumes/Sumit_HD/dataset/archive (1)'
output_csv = os.path.join(base_dir, 'labels.csv')

folders = {
    'Celeb-real': 0,
    'YouTube-real': 0,
    'Celeb-synthesis': 1
}

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['video_id', 'label'])
    for folder, label in folders.items():
        folder_path = os.path.join(base_dir, folder)
        for fname in os.listdir(folder_path):
            if fname.endswith('.mp4') and not fname.startswith('._'):
                video_id = fname[:-4]  # remove .mp4
                writer.writerow([video_id, label])

print(f"labels.csv generated at {output_csv}") 