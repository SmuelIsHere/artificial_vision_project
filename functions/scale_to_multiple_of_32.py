def scale_to_multiple_of_32(value, scale_factor):
    scaled_value = int(value // scale_factor)
    return (scaled_value + 31) // 32 * 32
