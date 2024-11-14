import argparse
import logging
from pathlib import Path
from vit_image_retrieval.core import ImageRetrievalSystem  # Changed import

def main():
    parser = argparse.ArgumentParser(description='ViT Image Retrieval System CLI')  # Updated description
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Index command
    index_parser = subparsers.add_parser('index', help='Index a directory of images')
    index_parser.add_argument('directory', type=str, help='Directory containing images')
    index_parser.add_argument('--output', '-o', type=str, default='index',
                            help='Base name for output files (default: index)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for similar images')
    search_parser.add_argument('query', type=str, help='Query image path')
    search_parser.add_argument('--index', '-i', type=str, required=True,
                             help='Path to FAISS index file')
    search_parser.add_argument('--metadata', '-m', type=str, required=True,
                             help='Path to metadata file')
    search_parser.add_argument('--num-results', '-k', type=int, default=5,
                             help='Number of results to return (default: 5)')

    args = parser.parse_args()

    if args.command == 'index':
        system = ImageRetrievalSystem()
        system.index_images(args.directory)
        system.save(f"{args.output}.faiss", f"{args.output}.json")
        print(f"Index saved as {args.output}.faiss and {args.output}.json")

    elif args.command == 'search':
        system = ImageRetrievalSystem(
            index_path=args.index,
            metadata_path=args.metadata
        )
        results = system.search(args.query, k=args.num_results)
        
        print(f"\nTop {len(results)} similar images:")
        for rank, (path, similarity, metadata) in enumerate(results, 1):
            print(f"\n{rank}. {Path(path).name}")
            print(f"   Similarity: {similarity:.3f}")
            print(f"   Distance: {metadata['distance']:.3f}")

if __name__ == '__main__':
    main()