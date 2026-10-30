#!/usr/bin/env bash
set -e

# ====== Configuration ======
ASSETS_DIR="./3D_assets"
OUTPUT_ROOT="./3D_converted_models"
SCRIPT_PATH="robocasa/scripts/model_zoo/import_glb_model.py"

# ====== Preparation ======
mkdir -p "$OUTPUT_ROOT"
new_object_path=()

# ====== Convert GLB files ======
# Use process substitution to avoid subshell (preserves array)
while read -r glb_path; do
    # Filename without extension
    base_name=$(basename "$glb_path" .glb)

    # Output path
    output_path="$OUTPUT_ROOT/$base_name"
    if [ -d "$output_path" ]; then
        echo "Skipping existing output: $output_path"
        continue
    fi
    mkdir -p "$output_path"

    echo "================================================="
    echo "Converting: $glb_path"
    echo "Output dir: $output_path"

    python "$SCRIPT_PATH" \
        --prescale \
        --center \
        --no_cached_coll \
        --path "$glb_path" \
        --output_path "$output_path"

    echo "Done: $base_name"
    new_object_path+=("$output_path")
done < <(find "$ASSETS_DIR" -type f -name "*.glb")

echo "All GLB files converted."

# Copy converted models to lrs_objs
for obj_path in "${new_object_path[@]}"; do
    obj_name=$(basename "$obj_path")
    echo "Copying $obj_name to lrs_objs"
    cp -r "$obj_path" ./robocasa/models/assets/objects/lrs_objs/
done

echo "Done! Converted ${#new_object_path[@]} models."
