"""Command-line interface for the SAM3 fursuit recognition system."""

import argparse
import sys
from pathlib import Path

from PIL import Image

from sam3_pursuit.api.identifier import SAM3FursuitIdentifier
from sam3_pursuit.config import Config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SAM3 Fursuit Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Identify character in an image
  python -m sam3_pursuit.api.cli identify photo.jpg

  # Identify with segmentation (for multi-character images)
  python -m sam3_pursuit.api.cli identify photo.jpg --segment

  # Add images for a character
  python -m sam3_pursuit.api.cli add --character "CharacterName" image1.jpg image2.jpg

  # Process unindexed images from download database
  python -m sam3_pursuit.api.cli index

  # Show statistics
  python -m sam3_pursuit.api.cli stats
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Identify command
    identify_parser = subparsers.add_parser("identify", help="Identify character in an image")
    identify_parser.add_argument("image", help="Path to image file")
    identify_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    identify_parser.add_argument("--segment", "-s", action="store_true", help="Use segmentation")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add images for a character")
    add_parser.add_argument("--character", "-c", required=True, help="Character name")
    add_parser.add_argument("images", nargs="+", help="Image paths")
    add_parser.add_argument("--segment", "-s", action="store_true", help="Use segmentation")

    # Index command
    index_parser = subparsers.add_parser("index", help="Process unindexed images")
    index_parser.add_argument("--images-dir", default=Config.IMAGES_DIR, help="Images directory")
    index_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")

    # Stats command
    subparsers.add_parser("stats", help="Show system statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize identifier
    identifier = SAM3FursuitIdentifier()

    if args.command == "identify":
        identify_command(identifier, args)
    elif args.command == "add":
        add_command(identifier, args)
    elif args.command == "index":
        index_command(identifier, args)
    elif args.command == "stats":
        stats_command(identifier)


def identify_command(identifier: SAM3FursuitIdentifier, args):
    """Handle identify command."""
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    image = Image.open(image_path)
    results = identifier.identify(image, top_k=args.top_k, use_segmentation=args.segment)

    if not results:
        print("No matches found.")
        return

    print(f"\nTop {len(results)} matches for {image_path.name}:")
    print("-" * 60)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.character_name or 'Unknown'}")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Distance: {result.distance:.4f}")
        print(f"   Post ID: {result.post_id}")
        print()


def add_command(identifier: SAM3FursuitIdentifier, args):
    """Handle add command."""
    # Verify images exist
    valid_paths = []
    for img_path in args.images:
        if Path(img_path).exists():
            valid_paths.append(img_path)
        else:
            print(f"Warning: Image not found: {img_path}")

    if not valid_paths:
        print("Error: No valid images provided.")
        sys.exit(1)

    added = identifier.add_images(
        character_name=args.character,
        image_paths=valid_paths,
        use_segmentation=args.segment
    )

    print(f"\nAdded {added} images for character '{args.character}'")


def index_command(identifier: SAM3FursuitIdentifier, args):
    """Handle index command."""
    processed = identifier.process_unindexed(
        images_dir=args.images_dir,
        batch_size=args.batch_size
    )

    print(f"\nProcessed {processed} images")


def stats_command(identifier: SAM3FursuitIdentifier):
    """Handle stats command."""
    stats = identifier.get_stats()

    print("\nSAM3 Fursuit Recognition System Statistics")
    print("=" * 50)
    print(f"Total detections: {stats['total_detections']}")
    print(f"Unique characters: {stats['unique_characters']}")
    print(f"Unique posts: {stats['unique_posts']}")
    print(f"Index size: {stats['index_size']}")

    # Show segmentor breakdown
    if stats.get('segmentor_breakdown'):
        print("\nSegmentor breakdown:")
        for model, count in stats['segmentor_breakdown'].items():
            print(f"  {model}: {count} embeddings")

    if stats['top_characters']:
        print("\nTop characters:")
        for name, count in stats['top_characters']:
            print(f"  {name}: {count} images")


if __name__ == "__main__":
    main()
