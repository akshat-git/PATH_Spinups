"""Print the `compute:` spec from a pipeline config, space-separated, so
jobs/final_setup.sh can read it into the GPU run's env instead of hardcoding values:

    batch_size  num_workers  amp_dtype  prefetch_factor

Every field is printed as a NON-EMPTY token so the reader (`read -r a b c d`) keeps
its columns aligned -- an empty token would be swallowed by shell word-splitting and
shift every later field left. Unset numeric knobs print the sentinel "-" (the job
script treats "-"/"auto" as "derive/fall back at runtime"), so the config spec stays
the single source of truth for the compute knobs.
Usage:  python tcga/print_compute_spec.py <config.yaml>
"""
import sys

from omegaconf import OmegaConf


def main():
    cfg = OmegaConf.load(sys.argv[1])
    c = cfg.get("compute", {}) or {}

    def g(key, default):
        v = c.get(key, default)
        return default if v is None else v

    # "-" = unset (derive/fall back); "auto" = explicit auto-derive. Never empty.
    print(f"{g('batch_size', '-')} {g('num_workers', 'auto')} "
          f"{g('amp_dtype', 'auto')} {g('prefetch_factor', '-')}")


if __name__ == "__main__":
    main()
