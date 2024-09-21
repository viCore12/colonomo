# Monoloco library  &nbsp;&nbsp;  [![Downloads](https://pepy.tech/badge/monoloco)](https://pepy.tech/project/monoloco)

## IMAGE
### MONOLOCO
```rb
python3 -m monoloco.run predict docs/002282.png --path_gt monoloco/names-kitti-200615-1022.json -o output_directory --long-edge 2500 --show_all
```

### STEREO
```rb
python3 -m monoloco.run predict --glob docs/000840*.png --path_gt monoloco/names-kitti-200615-1022.json -o output_directory --long-edge 2500 --show_all --mode stereo
```
## VIDEO
```rb
# A) 3D Localization
python3 -m monoloco.run predict --video_in --video_path_in docs/22456.mp4 --video_path_out output_directory/output_video.mp4 --output_types multi --path_gt monoloco/names-kitti-200615-1022.json
# B) Social Distancing (and Talking activity)
python3 -m monoloco.run predict --video_in --activities social_distance --video_path_in docs/22456.mp4 --video_path_out output_directory/output_video.mp4 --output_types multi
# C) Hand-raising detection
python3 -m monoloco.run predict --video_in --activities raising_hand --video_path_in docs/22456.mp4 --video_path_out output_directory/output_video.mp4 --output_types multi
```
