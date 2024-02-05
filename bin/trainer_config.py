import argparse
from source.config.config_gen import ConfigGenerator


def main(argv=None):
    parser = argparse.ArgumentParser(description="TRAINER")
    parser.add_argument("--json-file", type=str, required=True, help="config path")
    args, _ = parser.parse_known_args(argv)
    config = ConfigGenerator(json_file=args.json_file)
    config.run()


if __name__ == "__main__":
    main()
