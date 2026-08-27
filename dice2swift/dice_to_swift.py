import argparse
import numpy as np
import h5py as h5

from pygadgetreader import readsnap, readheader
import write_gadget as wg


GADGET_TYPE = [
    "gas",
    "dm",
    "disk",
    "bulge",
    "star",
    "bndry",
]


SWIFT_PARTTYPE = {
    "gas": 0,
    "dm": 1,
    "disk": 4,
    "bulge": 4,
    "bndry": 1,
    "star": 4,
}


UNIT_PRESETS = {

    "kpc-1e10msun-kms": {
        "length": 3.08567758e21,
        "mass": 1.9891e43,
        "velocity": 1.0e5,
    },

    "mpc-1e10msun-kms": {
        "length": 3.08567758e24,
        "mass": 1.9891e43,
        "velocity": 1.0e5,
    },

    "cgs": {
        "length": 1.0,
        "mass": 1.0,
        "velocity": 1.0,
    },
}


def positive_float(value):
    """argparse type for positive floating-point values."""

    value = float(value)

    if value <= 0:
        raise argparse.ArgumentTypeError(
            "Value must be greater than zero."
        )

    return value


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Convert a DICE/Gadget2 snapshot "
            "to a SWIFT-compatible HDF5 file."
        )
    )

    parser.add_argument(
        "input",
        help="Input Gadget2 snapshot (e.g. dice_disk_halo.g2)",
    )

    parser.add_argument(
        "output",
        help="Output HDF5 file (e.g. dice_disk_halo.hdf5)",
    )

    parser.add_argument(
        "--units",
        default="kpc-1e10msun-kms",
        choices=list(UNIT_PRESETS.keys()),
        help=(
            "Unit preset to use "
            "(default: kpc-1e10msun-kms)"
        ),
    )

    parser.add_argument(
        "--margin",
        type=positive_float,
        default=1.5,
        help=(
            "Factor used to determine the SWIFT "
            "BoxSize (default: 1.5)"
        ),
    )

    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        choices=GADGET_TYPE,
        help=(
            "Subset of particle types to convert "
            "(e.g. --types disk gas). "
            "If omitted, all present types are converted."
        ),
    )

    return parser.parse_args()


def dice_data(
    snap,
    header,
    types=None,
    verbose=True,
):
    """
    Load particle data from a DICE/Gadget2 snapshot.

    Parameters
    ----------
    snap : str
        Input snapshot.
    header : dict
        Already-loaded DICE header.
    types : list or None
        Particle types to load.
    verbose : bool
        Print information.

    Returns
    -------
    data : dict
        Particle data grouped by DICE particle type.
    """

    if types is None:
        types = GADGET_TYPE

    n_particles = header["npartTotal"]

    data = {}

    for i, ptype in enumerate(GADGET_TYPE):

        if ptype not in types:
            continue

        if n_particles[i] == 0:

            if verbose:
                print(
                    f"Skipping {ptype}: 0 particles"
                )

            continue


        entry = {
            "pos": readsnap(
                snap,
                "pos",
                ptype,
            ),

            "vel": readsnap(
                snap,
                "vel",
                ptype,
            ),

            "mass": readsnap(
                snap,
                "mass",
                ptype,
            ),

            "pid": readsnap(
                snap,
                "pid",
                ptype,
            ),
        }


        if ptype == "gas":

            try:

                entry["u"] = readsnap(
                    snap,
                    "u",
                    ptype,
                )

            except Exception as exc:

                raise RuntimeError(
                    "Gas particles are present, but "
                    "internal energy ('u') could not be read."
                ) from exc

            try:

                entry["hsml"] = readsnap(
                    snap,
                    "hsml",
                    ptype,
                )

            except Exception as exc:

                raise RuntimeError(
                    "Gas particles are present, but "
                    "smoothing length ('hsml') could not be read."
                ) from exc

        data[ptype] = entry

        if verbose:

            print(
                f"Loaded {n_particles[i]} "
                f"particles of type {ptype}"
            )

    return data


def verify_data(
    data,
    verbose=True,
):
    """
    Validate particle data before conversion.

    Checks:
        - NaN / Inf
        - positive masses
        - positive particle IDs
        - unique IDs within each DICE component
        - finite gas quantities
        - basic center-of-mass calculation
        - global ID uniqueness after SWIFT mapping
    """

    ok = True


    for ptype, d in data.items():

        n = d["pos"].shape[0]


        if (
            d["vel"].shape[0] != n
            or d["mass"].shape[0] != n
            or d["pid"].shape[0] != n
        ):

            print(
                f"ERROR: {ptype} arrays have "
                "inconsistent particle counts."
            )

            ok = False


        has_invalid = (

            not np.all(
                np.isfinite(d["pos"])
            )

            or not np.all(
                np.isfinite(d["vel"])
            )

            or not np.all(
                np.isfinite(d["mass"])
            )
        )


        nonpositive_mass = (
            np.any(d["mass"] <= 0)
        )


        if ptype == "gas":

            if "u" not in d:

                has_invalid = True

            else:

                if not np.all(
                    np.isfinite(d["u"])
                ):

                    has_invalid = True

                if np.any(d["u"] < 0):

                    print(
                        "ERROR: Gas internal energy "
                        "contains negative values."
                    )

                    has_invalid = True

            if "hsml" not in d:

                has_invalid = True

            else:

                if not np.all(
                    np.isfinite(d["hsml"])
                ):

                    has_invalid = True

                if np.any(d["hsml"] <= 0):

                    print(
                        "ERROR: Gas smoothing length "
                        "contains non-positive values."
                    )

                    has_invalid = True


        pid = np.asarray(
            d["pid"],
            dtype=np.int64,
        )

        nonpositive_ids = np.any(
            pid <= 0
        )

        unique_ids = (
            len(np.unique(pid))
            == len(pid)
        )


        total_mass = d["mass"].sum()

        if total_mass > 0:

            com = np.average(
                d["pos"],
                axis=0,
                weights=d["mass"],
            )

        else:

            com = np.array(
                [np.nan, np.nan, np.nan]
            )


        if verbose:

            print(f"\n--- {ptype} ---")

            print(
                f"N particles       : {n}"
            )

            print(
                f"NaN/Inf found     : {has_invalid}"
            )

            print(
                "Position range    : "
                f"min={d['pos'].min(axis=0)} "
                f"max={d['pos'].max(axis=0)}"
            )

            print(
                f"Total mass        : "
                f"{total_mass:.6e}"
            )

            print(
                f"Positive masses   : "
                f"{not nonpositive_mass}"
            )

            print(
                f"Unique IDs        : "
                f"{unique_ids}"
            )

            print(
                f"Non-positive IDs  : "
                f"{nonpositive_ids}"
            )

            print(
                f"Center of mass    : "
                f"{com}"
            )


        if (
            has_invalid
            or nonpositive_mass
            or not unique_ids
            or nonpositive_ids
        ):

            ok = False


    swift_groups = {}

    for ptype, d in data.items():

        swift_pt = SWIFT_PARTTYPE[ptype]

        swift_groups.setdefault(
            swift_pt,
            []
        ).append(
            np.asarray(
                d["pid"],
                dtype=np.int64,
            )
        )

    for swift_pt, id_arrays in swift_groups.items():

        if len(id_arrays) == 1:

            ids = id_arrays[0]

        else:

            ids = np.concatenate(
                id_arrays
            )

        unique_ids = (
            len(np.unique(ids))
            == len(ids)
        )

        if verbose:

            print(
                f"\nSWIFT PartType{swift_pt}:"
            )

            print(
                f"  Total particles : {len(ids)}"
            )

            print(
                f"  Global IDs unique: {unique_ids}"
            )

        if not unique_ids:

            print(
                f"ERROR: Duplicate ParticleIDs "
                f"inside SWIFT PartType{swift_pt}."
            )

            ok = False

    return ok


def create_header(
    f,
    header,
    data,
    margin=1.5,
    units=None,
    verbose=True,
):
    """
    Create the SWIFT HDF5 header and runtime/unit information.
    """

    def _n(ptype):

        if ptype in data:
            return len(
                data[ptype]["pid"]
            )

        return 0


    n_dark_matter = sum(
        _n(t)
        for t in ("dm", "bndry")
    )

    n_stars = sum(
        _n(t)
        for t in ("disk", "bulge", "star")
    )

    np_total_swift = [

        _n("gas"),

        n_dark_matter,

        0,

        0,

        n_stars,

        0,
    ]


    if not data:

        raise RuntimeError(
            "No particle data were loaded."
        )


    all_pos = np.concatenate(
        [
            data[t]["pos"]
            for t in data
        ],
        axis=0,
    )

    if all_pos.size == 0:

        raise RuntimeError(
            "No particle positions were found."
        )

    box_min = all_pos.min(
        axis=0
    )

    box_max = all_pos.max(
        axis=0
    )

    extent = (
        box_max - box_min
    ).max()

    if extent <= 0:

        raise RuntimeError(
            "Particle distribution has zero spatial extent."
        )

    boxsize = extent * margin


    center_offset = (
        boxsize / 2.0
        - (box_min + box_max) / 2.0
    )

    for t in data:

        data[t]["pos"] = (
            data[t]["pos"]
            + center_offset
        )


    other = {

        "Time": header["time"],

        "Redshift": header["redshift"],

        "MassTable": [0.0] * 6,
    }

    wg.write_header(
        f,

        boxsize=boxsize,

        flag_entropy=0,

        np_total=np_total_swift,

        np_total_hw=[0] * 6,

        other=other,
    )


    wg.write_runtime_pars(
        f,
        periodic_boundary=0,
    )


    if units is None:

        units = UNIT_PRESETS[
            "kpc-1e10msun-kms"
        ]

    time_unit = (
        units["length"]
        / units["velocity"]
    )

    wg.write_units(
        f,

        current=1.0,

        length=units["length"],

        mass=units["mass"],

        temperature=1.0,

        time=time_unit,
    )


    if verbose:

        print(
            f"\nBoxSize calculated: "
            f"{boxsize:.6f} "
            "(snapshot length units)"
        )

        print(
            "\nParticles:"
        )

        print(
            f"  Gas        : {_n('gas')}"
        )

        print(
            f"  Dark Matter: {n_dark_matter}"
        )

        print(
            f"  Stars      : {n_stars}"
        )

        print(
            "\nUnits:"
        )

        print(
            f"  Length   : "
            f"{units['length']:.8e} cm"
        )

        print(
            f"  Mass     : "
            f"{units['mass']:.8e} g"
        )

        print(
            f"  Velocity : "
            f"{units['velocity']:.8e} cm/s"
        )

        print(
            f"  Time     : "
            f"{time_unit:.8e} s"
        )

    return boxsize


def create_particle_block(
    f,
    data,
    verbose=True,
):
    """
    Group DICE components into SWIFT PartType groups
    and write them to the HDF5 file.
    """

    groups = {}


    for ptype in data:

        swift_pt = SWIFT_PARTTYPE[ptype]

        groups.setdefault(
            swift_pt,
            []
        ).append(ptype)


    for swift_pt, components in sorted(
        groups.items()
    ):


        pos = np.concatenate(
            [
                data[t]["pos"]
                for t in components
            ],
            axis=0,
        )

        vel = np.concatenate(
            [
                data[t]["vel"]
                for t in components
            ],
            axis=0,
        )

        mass = np.concatenate(
            [
                data[t]["mass"]
                for t in components
            ],
            axis=0,
        )

        ids_list = []

        for t in components:

            ids_original = np.asarray(
                data[t]["pid"],
                dtype=np.int64,
            )

            ids_list.append(
                ids_original
            )

        ids = np.concatenate(
            ids_list,
            axis=0,
        )

        n = pos.shape[0]


        if np.any(ids <= 0):

            raise ValueError(
                f"SWIFT PartType{swift_pt}: "
                "ParticleIDs <= 0."
            )

        if (
            len(np.unique(ids))
            != len(ids)
        ):

            raise ValueError(
                f"SWIFT PartType{swift_pt}: "
                "duplicate ParticleIDs."
            )


        kwargs = {

            "pos": pos,

            "vel": vel,

            "ids": ids,

            "mass": mass,
        }


        if swift_pt == 0:

            kwargs["int_energy"] = (
                np.concatenate(
                    [
                        data[t]["u"]
                        for t in components
                    ],
                    axis=0,
                )
            )

            kwargs["smoothing"] = (
                np.concatenate(
                    [
                        data[t]["hsml"]
                        for t in components
                    ],
                    axis=0,
                )
            )


        else:

            kwargs["int_energy"] = np.zeros(
                n,
                dtype=np.float64,
            )

            kwargs["smoothing"] = np.ones(
                n,
                dtype=np.float64,
            )


        wg.write_block(
            f,

            part_type=swift_pt,

            **kwargs,
        )

        if verbose:

            print(
                f"PartType{swift_pt} written "
                f"with {n} particles "
                f"({' + '.join(components)})"
            )


        del pos
        del vel
        del mass
        del ids
        del ids_list
        del kwargs


def sanity_check_output(
    hdf5_path,
    verbose=True,
):
    """
    Check the generated SWIFT HDF5 file.
    """

    ok = True

    with h5.File(
        hdf5_path,
        "r",
    ) as f:


        required = [

            "BoxSize",

            "NumPart_Total",

            "NumPart_Total_HighWord",

            "Flag_Entropy_ICs",
        ]

        if "Header" not in f:

            print(
                "ERROR: HDF5 file has no Header group."
            )

            return False

        for r in required:

            present = (
                r in f["Header"].attrs
            )

            ok &= present

            if verbose:

                print(
                    f"Header/{r}: "
                    f"{'OK' if present else 'MISSING'}"
                )

        if not ok:

            return False


        boxsize = np.asarray(
            f["Header"].attrs["BoxSize"],
            dtype=float,
        )

        boxsize = np.ravel(boxsize)

        if boxsize.size > 1 and not np.allclose(
            boxsize,
            boxsize[0],
        ):

            print(
                "WARNING: BoxSize is not isotropic."
            )

            ok = False

        boxsize_scalar = float(
            boxsize.max()
        )

        if (
            not np.all(
                np.isfinite(boxsize)
            )
            or np.any(boxsize <= 0)
        ):

            print(
                "ERROR: Invalid BoxSize."
            )

            ok = False


        num_part_total = np.asarray(
            f["Header"].attrs[
                "NumPart_Total"
            ]
        )

        for i, n_expected in enumerate(
            num_part_total
        ):

            group_name = (
                f"PartType{i}"
            )

            if n_expected == 0:
                continue


            if group_name not in f:

                ok = False

                print(
                    f"ERROR: NumPart_Total specifies "
                    f"{n_expected} particles "
                    f"in {group_name}, "
                    f"but the group does not exist."
                )

                continue

            group = f[group_name]


            required_datasets = [
                "Coordinates",
                "Velocities",
                "ParticleIDs",
                "Masses",
            ]

            for dataset_name in required_datasets:

                if dataset_name not in group:

                    print(
                        f"ERROR: {group_name}/"
                        f"{dataset_name} is missing."
                    )

                    ok = False

            if not ok:
                continue


            n_real = (
                group["Coordinates"].shape[0]
            )

            if n_real != n_expected:

                ok = False

                print(
                    f"ERROR: {group_name}: "
                    f"NumPart_Total="
                    f"{n_expected}, "
                    f"but {n_real} "
                    f"particles were written."
                )


            pos = group[
                "Coordinates"
            ][:]

            if not np.all(
                np.isfinite(pos)
            ):

                print(
                    f"ERROR: {group_name}: "
                    "Coordinates contain NaN/Inf."
                )

                ok = False

            if (
                pos.min() < 0
                or pos.max() > boxsize_scalar
            ):

                print(
                    f"ERROR: {group_name}: "
                    "particles outside "
                    "[0, BoxSize]."
                )

                ok = False


            vel = group[
                "Velocities"
            ][:]

            if not np.all(
                np.isfinite(vel)
            ):

                print(
                    f"ERROR: {group_name}: "
                    "Velocities contain NaN/Inf."
                )

                ok = False


            mass = group[
                "Masses"
            ][:]

            if not np.all(
                np.isfinite(mass)
            ):

                print(
                    f"ERROR: {group_name}: "
                    "Masses contain NaN/Inf."
                )

                ok = False

            if np.any(mass <= 0):

                print(
                    f"ERROR: {group_name}: "
                    "non-positive particle masses."
                )

                ok = False


            ids = group[
                "ParticleIDs"
            ][:]

            if np.any(ids <= 0):

                print(
                    f"ERROR: {group_name}: "
                    "ParticleIDs <= 0."
                )

                ok = False

            if (
                len(np.unique(ids))
                != len(ids)
            ):

                print(
                    f"ERROR: {group_name}: "
                    "duplicate ParticleIDs."
                )

                ok = False


            if i == 0:

                gas_datasets = [
                    "InternalEnergy",
                    "SmoothingLength",
                ]

                for dataset_name in gas_datasets:

                    if dataset_name not in group:

                        print(
                            f"ERROR: {group_name}/"
                            f"{dataset_name} is missing."
                        )

                        ok = False

                    else:

                        values = group[
                            dataset_name
                        ][:]

                        if not np.all(
                            np.isfinite(values)
                        ):

                            print(
                                f"ERROR: {group_name}/"
                                f"{dataset_name} "
                                "contains NaN/Inf."
                            )

                            ok = False

                        if (
                            dataset_name
                            == "InternalEnergy"
                            and np.any(values < 0)
                        ):

                            print(
                                f"ERROR: {group_name}/"
                                f"{dataset_name} "
                                "contains negative values."
                            )

                            ok = False

                        if (
                            dataset_name
                            == "SmoothingLength"
                            and np.any(values <= 0)
                        ):

                            print(
                                f"ERROR: {group_name}/"
                                f"{dataset_name} "
                                "contains non-positive values."
                            )

                            ok = False


    if verbose:

        if ok:

            print(
                "\nPost-write sanity check passed."
            )

        else:

            print(
                "\nPost-write sanity check "
                "found problems."
            )

    return bool(ok)


def main():


    args = parse_args()

    SNAP_IN = args.input
    HDF5_OUT = args.output

    UNIT_PRESET = args.units
    MARGIN = args.margin
    TYPES = args.types

    UNITS = UNIT_PRESETS[UNIT_PRESET]

    print(
        f"Input  : {SNAP_IN}"
    )

    print(
        f"Output : {HDF5_OUT}"
    )

    print(
        f"Units  : {UNIT_PRESET}"
    )

    print(
        f"Margin : {MARGIN}"
    )

    print(
        f"Types  : "
        f"{'all' if TYPES is None else TYPES}\n"
    )


    header = readheader(
        SNAP_IN,
        "header",
    )

    print(
        "Particle counts in DICE snapshot:"
    )

    for i, ptype in enumerate(
        GADGET_TYPE
    ):

        n = header[
            "npartTotal"
        ][i]

        if n > 0:

            print(
                f"  {ptype:6s}: "
                f"{n} particles"
            )

    print(
        f"\nOriginal BoxSize: "
        f"{header['boxsize']}"
    )


    print(
        "\nLoading DICE particle data..."
    )

    data = dice_data(
        SNAP_IN,
        header,
        types=TYPES,
        verbose=True,
    )

    print(
        "\nLoaded particle types:"
    )

    print(
        list(data.keys())
    )


    print(
        "\nChecking input data..."
    )

    input_ok = verify_data(
        data,
        verbose=True,
    )

    if not input_ok:

        raise RuntimeError(
            "\nInput validation failed. "
            "Fix the problems above before conversion."
        )

    print(
        "\nInput validation passed."
    )


    print(
        "\nCreating SWIFT HDF5 file..."
    )

    with h5.File(
        HDF5_OUT,
        "w",
    ) as f:

        create_header(
            f,

            header,

            data,

            margin=MARGIN,

            units=UNITS,

            verbose=True,
        )

        create_particle_block(
            f,

            data,

            verbose=True,
        )

    print(
        f"\nWritten file: "
        f"{HDF5_OUT}"
    )


    output_ok = sanity_check_output(
        HDF5_OUT,
        verbose=True,
    )

    if not output_ok:

        raise RuntimeError(
            "\nOutput sanity check failed."
        )

    print(
        "\nConversion completed successfully."
    )


if __name__ == "__main__":
    main()
