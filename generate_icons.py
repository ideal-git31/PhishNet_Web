"""
Run this once to generate placeholder icons for the Chrome extension.
pip install Pillow
python3 generate_icons.py
"""
import os

os.makedirs("extension/icons", exist_ok=True)

try:
    from PIL import Image, ImageDraw, ImageFont

    for size in [16, 48, 128]:
        img  = Image.new("RGBA", (size, size), (10, 14, 26, 255))
        draw = ImageDraw.Draw(img)
        # Blue circle background
        margin = size // 8
        draw.ellipse([margin, margin, size-margin, size-margin],
                     fill=(59, 130, 246, 255))
        # Shield character
        font_size = int(size * 0.55)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        draw.text((size//2, size//2), "🛡", fill=(255,255,255,255),
                  font=font, anchor="mm")
        img.save(f"extension/icons/icon{size}.png")
        print(f"✅ icon{size}.png created")

except ImportError:
    # Pillow not available — create minimal valid 1x1 PNG manually
    import struct, zlib

    def make_png(size):
        def chunk(name, data):
            c = struct.pack(">I", len(data)) + name + data
            return c + struct.pack(">I", zlib.crc32(name + data) & 0xffffffff)

        # 1x1 blue pixel PNG
        header = b'\x89PNG\r\n\x1a\n'
        ihdr   = chunk(b'IHDR', struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw    = b'\x00\x3b\x82\xf6'   # filter byte + RGB blue
        idat   = chunk(b'IDAT', zlib.compress(raw))
        iend   = chunk(b'IEND', b'')
        return header + ihdr + idat + iend

    for size in [16, 48, 128]:
        with open(f"extension/icons/icon{size}.png", "wb") as f:
            f.write(make_png(size))
        print(f"✅ icon{size}.png created (minimal)")

print("\nIcons ready in extension/icons/")