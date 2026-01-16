

import argparse
from src.run_experiments import run

def main():
    parser = argparse.ArgumentParser(description='Run DGAD Experiments.')
    parser.add_argument('--root_dir', type=str, default='./datasets/', help="dataset directory")
    parser.add_argument("--ratio", type=float, default=0.7)
    args = parser.parse_args()
    print(args.root_dir)
    
    run(args.root_dir)

if __name__ == '__main__':
    main()
   