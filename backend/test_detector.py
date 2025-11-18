"""
Test utilities for image_detector_enhanced.py
"""

import io
from PIL import Image, ImageDraw, ImageFilter
from image_detector_enhanced import analyze_image, selftest


def create_test_images():
    """Generate various test images to verify heuristics"""
    
    tests = []
    
    # Test 1: Clean photo-like image
    img = Image.new("RGB", (1920, 1080), color=(120, 140, 160))
    draw = ImageDraw.Draw(img)
    for i in range(0, 1920, 100):
        draw.rectangle([i, 0, i+50, 1080], fill=(100+i%100, 130, 150))
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    tests.append(("realistic_photo", buf.getvalue()))
    
    # Test 2: AI-typical resolution (512x512)
    img = Image.new("RGB", (512, 512), color=(200, 150, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    tests.append(("ai_resolution", buf.getvalue()))
    
    # Test 3: Over-sharpened image
    img = Image.new("RGB", (800, 800), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    for i in range(0, 800, 20):
        draw.line([(0, i), (800, i)], fill=(255, 255, 255), width=2)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.SHARPEN)
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=100)
    tests.append(("oversharpened", buf.getvalue()))
    
    # Test 4: Extremely smooth/uniform image
    img = Image.new("RGB", (1024, 1024), color=(180, 180, 181))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    tests.append(("extremely_smooth", buf.getvalue()))
    
    # Test 5: Normal photo with noise
    img = Image.new("RGB", (1280, 720))
    pixels = img.load()
    import random
    for x in range(img.width):
        for y in range(img.height):
            noise = random.randint(-30, 30)
            val = 128 + noise
            pixels[x, y] = (val, val+10, val-10)
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    tests.append(("noisy_photo", buf.getvalue()))
    
    return tests


def run_comprehensive_tests():
    """Run full test suite"""
    print("=" * 60)
    print("IMAGE DETECTOR v0.6 - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # 1. Self-test
    print("\n[1/3] Running module self-test...")
    selftest_result = selftest()
    print(f"Status: {selftest_result['status']}")
    print(f"Summary: {selftest_result['summary']}")
    
    for test in selftest_result['tests']:
        status_icon = "✓" if test['status'] == 'pass' else "✗"
        print(f"  {status_icon} {test['test']}: {test['details']}")
    
    # 2. Test images
    print("\n[2/3] Testing various image types...")
    test_images = create_test_images()
    
    for name, data in test_images:
        result = analyze_image(data)
        
        if result['status'] == 'success':
            print(f"\n--- {name.upper()} ---")
            print(f"  Dimensions: {result['dimensions']['width']}x{result['dimensions']['height']}")
            print(f"  File size: {result['file_size_kb']} KB")
            print(f"  Pre-filter score: {result['prefilter_score']}")
            print(f"  Detection level: {result['detection_level']}")
            print(f"  Needs ML: {result['ml_required']}")
            print(f"  Recommendation: {result['recommendation']}")
            
            if result['flags']:
                print(f"  Flags:")
                for flag in result['flags']:
                    print(f"    - {flag}")
        else:
            print(f"\n--- {name.upper()} ---")
            print(f"  ERROR: {result['error_message']}")
    
    # 3. Edge cases
    print("\n[3/3] Testing edge cases...")
    
    edge_cases = [
        ("empty_data", b""),
        ("invalid_format", b"this is not an image"),
        ("tiny_image", create_tiny_image()),
    ]
    
    for name, data in edge_cases:
        result = analyze_image(data)
        status_icon = "✓" if result['status'] == 'error' else "✗"
        print(f"  {status_icon} {name}: {result.get('error_message', 'Unexpected success')}")
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)


def create_tiny_image():
    """Create image smaller than minimum dimensions"""
    img = Image.new("RGB", (10, 10), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def demo_usage():
    """Demonstrate typical usage patterns"""
    print("\n" + "=" * 60)
    print("USAGE DEMONSTRATION")
    print("=" * 60)
    
    # Create a sample image
    img = Image.new("RGB", (512, 512), color=(150, 150, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_data = buf.getvalue()
    
    # Analyze
    result = analyze_image(image_data)
    
    print("\nExample 1: Basic Analysis")
    print("-" * 40)
    print(f"Status: {result['status']}")
    print(f"Pre-filter Score: {result['prefilter_score']}")
    print(f"Detection Level: {result['detection_level']}")
    
    print("\nExample 2: Decision Flow")
    print("-" * 40)
    if result['status'] == 'error':
        print(f"❌ Error: {result['error_message']}")
    elif result['needs_ml_verification']:
        print(f"⚠️  Inconclusive - ML verification needed")
        print(f"   Pre-filter score: {result['prefilter_score']}")
    elif result['detection_level'] == 'obvious_ai':
        print(f"🤖 High confidence AI detection")
        print(f"   Flags: {', '.join(result['flags'][:3])}")
    else:
        print(f"📸 Likely authentic photo")
    
    print("\nExample 3: Detailed Breakdown")
    print("-" * 40)
    if 'scores_breakdown' in result:
        for check_name, check_data in result['scores_breakdown'].items():
            print(f"{check_name.capitalize():12s}: {check_data['score']:.3f}")


if __name__ == "__main__":
    # Run all tests
    run_comprehensive_tests()
    
    # Show usage demo
    demo_usage()
