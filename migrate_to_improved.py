"""
Migration script to upgrade from the current VGG16-based system to the improved CLIP-based system
"""

import os
import sqlite3
import numpy as np
from PIL import Image
from improved_pursuit import ImprovedFursuitIdentifier
from download import recall_furtrack_data_by_embedding_id
from random import sample

old_src_path = "/Users/golis001/Desktop/rutorch"
old_db_path = f"{old_src_path}/furtrack.db"
furtrack_images = f"{old_src_path}/furtrack_images"

def get_confidence_stats():
    """Test all files and see how many % of the same character are identified if they are in the database more than once"""
    identifier = ImprovedFursuitIdentifier()
    conn = sqlite3.connect(identifier.db_path)
    c = conn.cursor()
    
    c.execute("SELECT post_id, character_name FROM fursuits")
    rows = c.fetchall()
    
    char_stats = {}
    for post_id, char_name in rows:
        if char_name not in char_stats:
            char_stats[char_name] = []
        char_stats[char_name].append(post_id)
    
    total_files = 0
    correct_identifications = 0
    
    for char_name, post_ids in sample([*char_stats.items()], 300):
        print(f"Processing character: {char_name} with {len(post_ids)} images")
        if len(post_ids) > 1:  # Only consider characters with multiple images
            total_files += len(post_ids)
            for post_id in post_ids:
                try:
                    image_path = os.path.join(furtrack_images, post_id)
                    if os.path.exists(image_path):
                        test_image = Image.open(image_path)
                        results = identifier.identify_character(test_image, top_k=3)
                        if any(result['character_name'] == char_name for result in results if result['post_id'] != post_id):
                            correct_identifications += 1
                except Exception as e:
                    print(f"Error processing {post_id}: {e}")
    
    if total_files > 0:
        confidence = correct_identifications / total_files
        print(f"Confidence of identifying characters with multiple images: {confidence:.2%}")
    else:
        print("No characters with multiple images found.")

def test_migration():
    """Test the migrated system with a sample image"""
    identifier = ImprovedFursuitIdentifier()
    
    # Find a test image
    image_dir = furtrack_images
    if os.path.exists(image_dir):
        test_files = sample([f for f in os.listdir(image_dir) if f.endswith('.jpg')], 5)
        
        for test_file in test_files:
            print(f"\nTesting with {test_file}:")
            try:
                test_image = Image.open(os.path.join(image_dir, test_file))
                results = identifier.identify_character(test_image, top_k=3)
                
                print("Results:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['character_name']} "
                          f"(confidence: {result['confidence']:.3f})")
                post = identifier.get_post_by_id(test_file)
                ground_truth = post.get('character_name') if post else None
                # TODO: this is only for one character, need to handle multiple
                if ground_truth:
                    print(f"Ground truth character: {ground_truth}")
            except Exception as e:
                print(f"Error testing {test_file}: {e}")

def compare_systems():
    """Compare results between old and new systems"""
    from pursuit import detect_characters
    
    identifier = ImprovedFursuitIdentifier()
    
    # Find test images
    image_dir = furtrack_images
    if os.path.exists(image_dir):
        test_files =  sample([f for f in os.listdir(image_dir) if f.endswith('.jpg')], 3)
        
        for test_file in test_files:
            print(f"\n=== Comparing results for {test_file} ===")
            image_path = os.path.join(image_dir, test_file)
            
            try:
                # New system results
                print("New system (CLIP):")
                test_image = Image.open(image_path)
                new_results = identifier.identify_character(test_image, top_k=3)
                for i, result in enumerate(new_results, 1):
                    print(f"  {i}. {result['character_name']} "
                          f"(confidence: {result['confidence']:.3f})")
                    
                # Old system results
                print("Old system (VGG16):")
                old_results = detect_characters(image_path, 3)
                for i, result in enumerate(old_results, 1):
                    print(f"  {i}. {result['char']}")
            except Exception as e:
                print(f"Error comparing {test_file}: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "migrate":
            # migrate_existing_data()
            pass
        elif command == "test":
            test_migration()
        elif command == "confidence":
            get_confidence_stats()
        elif command == "compare":
            compare_systems()
        else:
            print("Usage: python migrate_to_improved.py [migrate|test|compare]")
    else:
        print("Available commands:")
        print("  migrate  - Migrate data from old system to new system")
        print("  test     - Test the new system with sample images")
        print("  compare  - Compare old vs new system results")
