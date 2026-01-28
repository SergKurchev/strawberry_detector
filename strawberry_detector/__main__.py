"""
CLI interface for strawberry detection.

Usage:
    python -m strawberry_detector --image photo.jpg --output results.json
    python -m strawberry_detector --folder images/ --output results/
    python -m strawberry_detector --image https://example.com/img.jpg --output results.json --visualize
"""

import argparse
import json
import sys
from pathlib import Path

from .detector import StrawberryDetector


def main():
    parser = argparse.ArgumentParser(
        description="🍓 Strawberry Detection with Depth Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python -m strawberry_detector --image photo.jpg --output results.json
  python -m strawberry_detector --image photo.jpg --output results.json --visualize
  
  # Folder of images
  python -m strawberry_detector --folder images/ --output results/
  python -m strawberry_detector --folder images/ --output results/ --visualize
  
  # From URL
  python -m strawberry_detector --image "https://example.com/strawberries.jpg" --output results.json
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", "-i",
        help="Path or URL to input image"
    )
    input_group.add_argument(
        "--folder", "-f",
        help="Path to folder containing images"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to save output (JSON file for single image, folder for batch)"
    )
    
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Save visualization image(s)"
    )
    
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🍓 Strawberry Detection with Depth Estimation")
    print("=" * 60)
    
    try:
        # Initialize detector
        detector = StrawberryDetector(device=args.device)
        
        if args.folder:
            # Folder mode
            print(f"📁 Input folder: {args.folder}")
            print(f"📁 Output folder: {args.output}")
            print()
            
            results = detector.detect_folder(
                args.folder,
                output_folder=args.output,
                conf_threshold=args.conf,
                visualize=args.visualize
            )
            
            # Save combined results
            output_folder = Path(args.output)
            combined_path = output_folder / "all_results.json"
            with open(combined_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print()
            print("=" * 60)
            print(f"✅ Batch processing complete!")
            print(f"📊 Processed {len(results)} images")
            
            # Aggregate stats
            total_strawberries = sum(r.get("detections_count", 0) for r in results if "detections_count" in r)
            print(f"🍓 Total strawberries found: {total_strawberries}")
            print(f"💾 Results saved to: {args.output}")
            print("=" * 60)
            
        else:
            # Single image mode
            print(f"📷 Image: {args.image}")
            print(f"💾 Output: {args.output}")
            print()
            
            if args.visualize:
                output_path = Path(args.output)
                vis_path = output_path.parent / f"{output_path.stem}_vis.jpg"
                result, vis = detector.detect_and_visualize(
                    args.image,
                    output_image_path=vis_path,
                    conf_threshold=args.conf
                )
                print(f"🖼️ Visualization saved: {vis_path}")
            else:
                result = detector.detect(args.image, conf_threshold=args.conf)
            
            # Save JSON
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print()
            print("=" * 60)
            print(f"✅ Detection complete!")
            print(f"📊 Found {result['detections_count']} strawberries")
            
            # Print summary
            stats = result["statistics"]
            print(f"   - Ripe: {stats.get('total_ripe', 0)}")
            print(f"   - Unripe: {stats.get('total_unripe', 0)}")
            
            if "closest_distance_cm" in stats:
                print(f"   - Closest: {stats['closest_distance_cm']:.1f} cm")
            if "furthest_distance_cm" in stats:
                print(f"   - Furthest: {stats['furthest_distance_cm']:.1f} cm")
            
            print(f"💾 Results saved to: {args.output}")
            print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
