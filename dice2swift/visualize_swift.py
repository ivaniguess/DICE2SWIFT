import argparse
import os
import yt


def parse_args():

    parser = argparse.ArgumentParser(
        description="Visualize a SWIFT HDF5 snapshot with yt."
    )

    parser.add_argument(
        "input",
        help="SWIFT HDF5 snapshot (e.g. disk_halo.hdf5)",
    )

    parser.add_argument(
        "--axis",
        default="z",
        choices=["x", "y", "z"],
        help="Projection axis (default: z)",
    )

    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="Width of the plotted region in kpc.",
    )

    parser.add_argument(
        "--outdir",
        default=".",
        help="Directory where plots are saved.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ds = yt.load(args.input)

    print(f"Loaded dataset: {ds}")
    print(f"Domain dimensions: {ds.domain_width}")

    # --------------------------------------------------------
    # Width
    # --------------------------------------------------------

    if args.width is not None:
        width = (args.width, "kpc")
    else:
        width = None

    # --------------------------------------------------------
    # Particle types
    # --------------------------------------------------------

    particle_types = {
        "PartType0": "gas",
        "PartType1": "dark_matter",
        "PartType4": "stars",
    }

    # --------------------------------------------------------
    # Projection plots
    # --------------------------------------------------------

    for ptype, label in particle_types.items():

        mass_field = (ptype, "Masses")

        if mass_field not in ds.field_list:
            print(f"{ptype} not found. Skipping.")
            continue

        print(f"Creating {label} projection...")

        p = yt.ParticleProjectionPlot(
            ds,
            args.axis,
            mass_field,
            width=width,
        )

        p.set_cmap(mass_field, "magma")

        p.set_xlabel("Position [kpc]")
        p.set_ylabel("Position [kpc]")

        filename = os.path.join(
            args.outdir,
            f"{label}_projection.png"
        )

        p.save(filename)

        print(f"Saved: {filename}")

    print("\nDone.")


if __name__ == "__main__":
    main()
