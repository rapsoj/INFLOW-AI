# # Run inference
# maps = run_inference_pipeline()

# Load prediction CSV
pred_df, folder_title = load_latest_prediction_csv()

# Determine example time index
assert sorted(maps.keys())[-1] + 13 == len(pred_df), "Mismatch in prediction timeline"
example_t = sorted(maps.keys())[-1]

# Extract flood thresholds
flood_percentages = {
    "Worst Case": pred_df.iloc[-1]['upper_bound_95'],
    "Average Case": pred_df.iloc[-1]['percent_inundation'],
    "Best Case": pred_df.iloc[-1]['lower_bound_95']
}

# Generate flood masks
masks = {
    scenario: generate_flood_mask(maps[example_t], pct)
    for scenario, pct in flood_percentages.items()
}

# Load spatial reference and current extent
with h5py.File('data/historic/inundation.h5', 'r') as f:
    current_extent = f['inundation'][-1]

transform, crs, regions_gdf, _ = load_spatial_ref()

# Create output directory
output_dir = Path(f"predictions/{folder_title}/spatial_predictions")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot full country
title = f"Predicted South Sudan Flood Extent Change from {folder_title[23:33]} to {folder_title[-10:]}\n"
fig = plot_flood_change_map(masks, current_extent, transform, crs, regions_gdf, title=title)
fig.savefig(output_dir / f"south_sudan_spatial_prediction_{folder_title[23:33]}_to_{folder_title[-10:]}.png", dpi=300)
plt.close(fig)

# Plot regions
for region in regions_gdf['region']:
    if region.strip().lower() == "south sudan":
        continue

    region_clean = "_".join(region.lower().split())
    title = f"Predicted {region} Flood Extent Change from {folder_title[23:33]} to {folder_title[-10:]}\n"

    try:
        fig = plot_flood_change_map(
            masks, current_extent, transform, crs, regions_gdf,
            region_name=region, title=title
        )
        filename = f"{region_clean}_spatial_prediction_{folder_title[23:33]}_to_{folder_title[-10:]}.png"
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting {region}: {e}")

# Export for QGIS
export_qgis_files(masks, current_extent, transform, crs, regions_gdf, folder_title)