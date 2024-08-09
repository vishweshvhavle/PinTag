#!/bin/bash

# Define the configurations
configs=(
    "data/pdf/PinTag_12-34-56.pdf random_frame_configs.json data/pin_tags_random/"
    "data/pdf/AprilTag.pdf random_frame_configs.json data/april_tags_random/"
    "data/pdf/ArUco.pdf random_frame_configs.json data/aruco_tags_random/"
)

# Loop through each configuration and run the command
for config in "${configs[@]}"
do
    IFS=' ' read -r -a args <<< "$config"
    pdf_file=${args[0]}
    config_file=${args[1]}
    output_dir=${args[2]}

    python test/test_data_gen_config.py --pdf_file "$pdf_file" --axis -1 0 0 --angle 270 --config_file "$config_file" --output_dir "$output_dir"
done

echo "Data generation completed."
