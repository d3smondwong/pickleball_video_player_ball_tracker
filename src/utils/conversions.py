def convert_pixel_distance_to_feet(pixel_distance, reference_height_in_feet, reference_height_in_pixels):
    return (pixel_distance * reference_height_in_feet) / reference_height_in_pixels

def convert_feet_to_pixel_distance(feet, reference_height_in_feet, reference_height_in_pixels):
    return (feet * reference_height_in_pixels) / reference_height_in_feet