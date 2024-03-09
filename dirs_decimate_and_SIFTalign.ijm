// process an entire experiment day
exp_dir = getDirectory("Choose top-level experiment dir");
print("\\Clear");
print("Start");
// List all folders in the top-level experiment dir
list_top_lvl = getFileList(exp_dir);

for(i=0; i<lengthOf(list_top_lvl); i++){
	possible_dir = exp_dir + list_top_lvl[i];
	print(list_top_lvl[i]);
	if (File.isDirectory(possible_dir)){
		list = getFileList(possible_dir);
		if (dir_processed(list) == false){
			for (j = 0; j < lengthOf(list); j++) {
						if (endsWith(list[j], ".tif") || endsWith(list[j], ".tiff")) {
							file_path = possible_dir + list[j];
							print(file_path);
							processTiff(file_path);
							print("\\Clear");
						}
			}

		}
	}
		
}
print("Finished");

function dir_processed(file_list){
	// if aligned tif already in dir, do not process it
	for(k=0; k<lengthOf(file_list); k++){
		if (startsWith(file_list[k], "aligned")){
			print(file_list[k] + " exists, skipping...");
			return true;
		}	
	}
	return false;
}

function processTiff(fp){
	open(fp);
	original_title = getTitle();
	filename = File.getName(fp);
	dir = File.getParent(fp);
	
	// Determine if the number of slices is a multiple of 8
	if (nSlices > 1) {
	    remainder = nSlices % 8;
	    if (remainder != 0) {
	        // Calculate number of slices to delete
	        slicesToDelete = remainder;
	        // Delete the extra slices from the end
	        for (a = 0; a < slicesToDelete; a++) {
	            // Always delete the last slice in the current stack
	            setSlice(nSlices);
	            run("Delete Slice");
	        }
	    }
	
	run("Grouped Z Project...", "projection=[Average Intensity] group=8");
	run("Linear Stack Alignment with SIFT", "initial_gaussian_blur=1.60 steps_per_scale_octave=3 minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size=4 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=5 inlier_ratio=0.05 expected_transformation=Translation interpolate show_info show_transformation_matrix");
	
	new_filename = "aligned_ZProject8_" + filename;
	
	file_path_out = dir + File.separator + new_filename;
	
	selectWindow("Aligned " + nSlices + " of " + nSlices);
	// Save the averaged image
	saveAs("Tiff", file_path_out);
	
	// close original stack and aligned stack
	//close("*");
	print("Closing images");
	//close(original_title);
	//close("Aligned " + nSlices + " of " + nSlices);
	//close("Alignment info");
	close("*");
	return;
}
