#!/usr/bin/env python3
"""
View ManTraNet detection results - shows original image + forgery heatmap side by side
"""
import sys
import json
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO
import subprocess

def visualize_result(result_file: Path):
    """Load result JSON and display original + mask"""
    with open(result_file) as f:
        content = f.read()
        # Find JSON start (after stderr messages)
        json_start = content.find('{')
        if json_start < 0:
            print("❌ No JSON found in result file")
            return None
        data = json.loads(content[json_start:])

    if not data['success']:
        print(f"❌ Inference failed: {data.get('error', 'Unknown error')}")
        return

    analysis = data['analysis']
    print(f"✓ Manipulated: {analysis['manipulated_percentage']:.2f}%")
    print(f"  Regions: {analysis['region_count']}")
    print(f"  Max confidence: {analysis['max_confidence']:.3f}")

    # Decode mask
    if 'mask_bytes' in data:
        mask_data = base64.b64decode(data['mask_bytes'])
        mask_image = Image.open(BytesIO(mask_data))

        # Save mask
        output_path = result_file.with_suffix('.mask.png')
        mask_image.save(output_path)
        print(f"  Mask saved: {output_path}")
        return output_path
    else:
        print("  ⚠ No mask data in result")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: view_mantranet_results.py <result.txt>")
        print("\nExample:")
        print("  view_mantranet_results.py /tmp/fake_result.txt")
        sys.exit(1)

    result_file = Path(sys.argv[1])
    if not result_file.exists():
        print(f"Error: {result_file} not found")
        sys.exit(1)

    print(f"\nAnalyzing: {result_file.name}")
    print("-" * 50)
    mask_path = visualize_result(result_file)

    if mask_path:
        print(f"\n✓ Forgery heatmap saved!")
        print(f"  View with: display {mask_path}")
        print(f"  Or open in browser: file://{mask_path.absolute()}")
