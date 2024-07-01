from __future__ import annotations
import matplotlib.colors as mcolors
from difflib import get_close_matches

class MatplotlibColorError(Exception):
    pass

def suggest_color(color_name):
    all_colors = list(mcolors.CSS4_COLORS.keys())
    suggestions = get_close_matches(color_name, all_colors, n=3, cutoff=0.6)
    return suggestions

def from_matplotlib(color_name: str | int, alpha=1.0):
    try:
        # Convert the color name to RGB
        rgb = mcolors.to_rgb(color_name)

        # Convert RGB (0-1 scale) to 0-255 scale and include full alpha
        r, g, b = [int(x * 255) for x in rgb]
        a = int(alpha * 255)

        # Format as 0xAARRGGBB
        return int(f"0x{a:02X}{r:02X}{g:02X}{b:02X}", 16)
    except ValueError as e:
        suggestions = suggest_color(color_name)
        if suggestions:
            raise MatplotlibColorError(
                f"Invalid color name: {color_name}. Did you mean {', '.join(suggestions)}?"
            ) from e
        else:
            raise MatplotlibColorError(f"Invalid color name: {color_name}.") from e

def format_color(color_name: str | int, alpha=1.0):
    if isinstance(color_name, int):
        return color_name
    if color_name.startswith("0x"):
        return int(color_name, 16)
    return from_matplotlib(color_name, alpha)

# Example usage
if __name__ == "__main__":    

    print("Matplotlib colors converted to 0xAARRGGBB format:")
    for color in ["red", "green", "blue",
                  "0xFFFF00FF"]:
        hex_color = format_color(color)

        if hex_color:
            as_hex = "0x" + hex(hex_color)[2:].zfill(8).upper()
            print(f"{color}: {as_hex}")
        else:
            print(f"{color}: Invalid color name")

    # Test with user input
    while True:
        user_color = input(
            "\nEnter a matplotlib color name (or 'quit' to exit): "
        ).strip()
        if user_color.lower() == "quit":
            break
        hex_color = format_color(user_color)
        
        if hex_color:
            as_hex = '0x'+hex(hex_color)[2:].zfill(8).upper()
            print(f"{user_color}: {as_hex}")
        else:
            print(f"{user_color}: Invalid color name")
