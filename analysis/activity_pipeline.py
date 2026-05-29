import pandas as pd
from actimotus import Features, Activities, Exposures
import actipy
import numpy as np
from pathlib import Path
import inspect
import openpyxl
from joblib import Parallel, delayed
import shutil
import gc
import logging
from actimotus.exposures import PLOT
import copy
import calendar

ACTIVITY_PLOT_COLORS = {
    "non-wear": "#BDBDBD",
    "lie": "#90CAF9",
    "sit": "#0D47A1",
    "stand": "#00695C",
    "shuffle": "#FF1493",
    "walk": "#81C784",
    "fast-walk": "#EF6C00",
    "row": "#AB47BC",
    "run": "#A00707",
    "stairs": "#FDD835",
    "bicycle": "#8D6E63",
}


def make_activity_plot_lang(activities_df: pd.DataFrame) -> dict:
    plot_lang = copy.deepcopy(PLOT)
    present = set(activities_df["activity"].dropna().astype(str).unique())

    for activity, color in ACTIVITY_PLOT_COLORS.items():
        if activity in plot_lang["activities"]:
            plot_lang["activities"][activity]["color"] = color

    plot_lang["activities"] = {
        activity: cfg
        for activity, cfg in plot_lang["activities"].items()
        if activity in present
    }
    return plot_lang


def load_raw_file(cwa_file: Path):
    df, info = actipy.read_device(
        str(cwa_file),
        calibrate_gravity=False,
        detect_nonwear=False,
        lowpass_hz=None,
        resample_hz=None
    )

    df.index.name = 'datetime'
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def _remove_dst_spring_forward(activities: pd.DataFrame) -> pd.DataFrame:
    """Fjern aktiviteter i sommertidshullet 02:00-03:00 på sidste søndag i marts."""
    years = activities.index.year.unique()
    mask_remove = pd.Series(False, index=activities.index)
    for year in years:
        last_day = calendar.monthrange(year, 3)[1]
        d = pd.Timestamp(year, 3, last_day)
        dst_sunday = d if d.dayofweek == 6 else d - pd.Timedelta(days=d.dayofweek + 1)
        gap_start = pd.Timestamp(year, 3, dst_sunday.day, 2, 0, 0)
        gap_end   = pd.Timestamp(year, 3, dst_sunday.day, 3, 0, 0)
        mask_remove |= (activities.index >= gap_start) & (activities.index < gap_end)

        # Fjern evt. syntetisk kant-række ved 03:00 (shuffle uden vinkeldata)
        boundary = (activities.index == gap_end) & (activities["activity"] == "shuffle")
        angle_cols = [c for c in ["thigh_inclination", "thigh_direction", "thigh_side_tilt"] if c in activities.columns]
        if angle_cols:
            boundary &= activities[angle_cols].isna().all(axis=1)
        mask_remove |= boundary

    return activities[~mask_remove]

def compute_activities_from_raw(df_subset: pd.DataFrame) -> pd.DataFrame:
    features = Features().compute(df_subset)
    activities, references = Activities(vendor="Other", orientation=False).compute(features)
    del features; gc.collect()
    activities = _remove_dst_spring_forward(activities) 
    activities_marked = activities.copy()
    fast_walk_mask = (
        activities_marked.loc[activities_marked['activity'] == 'walk', 'steps']
        .groupby(pd.Grouper(freq='15s'), observed=False)
        .transform('sum')
    ) > 27
    fast_walk_mask = fast_walk_mask[fast_walk_mask]
    activities_marked.loc[fast_walk_mask.index, 'activity'] = 'fast-walk'

    return activities_marked


PLOT_CHUNK_DAYS = 8


def _save_activity_plots(activities_marked: pd.DataFrame, file_stem: str, file_path_output: Path):
    if activities_marked.empty:
        return

    max_chunk = pd.Timedelta(days=PLOT_CHUNK_DAYS)
    start = activities_marked.index.min()
    end = activities_marked.index.max()

    if end - start <= max_chunk:
        chart = Exposures(fused=False).plot(
            activities_marked,
            language=make_activity_plot_lang(activities_marked),
        )
        chart.save(str(file_path_output / f"{file_stem}.png"))
        del chart
        gc.collect()
        return

    chunk_start = start
    part = 1
    while chunk_start <= end:
        chunk_end = chunk_start + max_chunk
        chunk = activities_marked.loc[(activities_marked.index >= chunk_start) & (activities_marked.index < chunk_end)]

        if not chunk.empty:
            chart = Exposures(fused=False).plot(
                chunk,
                language=make_activity_plot_lang(chunk),
            )
            chart.save(str(file_path_output / f"{file_stem}_part{part}.png"))
            del chart
            gc.collect()
            part += 1

        chunk_start = chunk_end


def process_file(
    df_subset: pd.DataFrame,
    file_stem: str,
    file_path_output: Path,
    exposures_dir: Path,
    axis_rotation_code: str,
    error_log: Path,):

    try:
        print(f"Processing: {file_stem}")

        activities_marked = compute_activities_from_raw(df_subset)
        del df_subset; gc.collect()

        # Exposures
        exposures_obj = Exposures(fused=True)
        exposure_result = exposures_obj.compute(activities_marked)
        exposure_result['file'] = file_stem

        _save_activity_plots(activities_marked, file_stem, file_path_output)

        exposure_csv = exposures_dir / f"{file_stem}.csv"
        exposure_result.to_csv(exposure_csv, index=False, encoding="utf-8")

        out_csv = file_path_output / f"{file_stem}.csv"
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            f.write(f"# axis_rotation_code:\n")
            for line in axis_rotation_code.split('\n'):
                f.write(f"# {line}\n")
            f.write(f"#\n")
            activities_marked.index.name = 'datetime'
            activities_marked.to_csv(f, index=True, encoding="utf-8")

        del activities_marked, exposure_result, exposures_obj
        gc.collect()

        return (file_stem, "SUCCESS")

    except Exception as e:
        import traceback
        error_msg = f"{file_stem}: {str(e)}\n{traceback.format_exc()}\n{'='*60}\n"
        with open(error_log, "a", encoding="utf-8") as f:
            f.write(error_msg)
        print(f"FEJL i {file_stem}: {e}")
        return (file_stem, "FAILED")



#funktion til at indlæse eksisterende kandidater
def load_existing_candidates(output_path: Path) -> pd.DataFrame:
    parquet_file = output_path / "rotation_candidates.parquet"
    csv_file = output_path / "rotation_candidates.csv"

    if parquet_file.exists():
        candidates = pd.read_parquet(parquet_file)
    else:
        candidates = pd.read_csv(csv_file)

    candidates["start"] = pd.to_datetime(candidates["start"])
    candidates["end"] = pd.to_datetime(candidates["end"])

    return candidates[["episode_id", "file_name", "file_path", "start", "end", "original_rotation_id"]].copy()


# ROTATIONS - Definition af de 8 enhedsorientaringer
ROTATIONS = {
    1: {
        "name": "Lodret, forside ud, pil nedad",
        "transform": lambda df: (df['x'] * -1, df['y'] * -1, df['z'])
    },
    2: {
        "name": "Lodret, forside ud, pil opad",
        "transform": lambda df: (df['x'], df['y'], df['z'])
    },
    3: {
        "name": "Lodret, forside ind, pil nedad",
        "transform": lambda df: (df['x'] * -1, df['y'], df['z'] * -1)
    },
    4: {
        "name": "Lodret, forside ind, pil opad",
        "transform": lambda df: (df['x'], df['y'] * -1, df['z'] * -1)
    },
    5: {
        "name": "Vandret, forside ud, pil til højre side for personen",
        "transform": lambda df: (df['y'], df['x'] * -1, df['z'])
    },
    6: {
        "name": "Vandret, forside ud, pil til venstre side for personen",
        "transform": lambda df: (df['y'] * -1, df['x'], df['z'])
    },
    7: {
        "name": "Vandret, forside ind, pil til højre side for personen",
        "transform": lambda df: (df['y'] * -1, df['x'] * -1, df['z'] * -1)
    },
    8: {
        "name": "Vandret, forside ind, pil til venstre side for personen",
        "transform": lambda df: (df['y'], df['x'], df['z'] * -1)
    },
}

# Hjælpefunktion til at anvende rotation
def apply_rotation(df, rotation_id):
    """
    Ret accelerometerakslerne efter enhedens monteringstype
    """
    # Check at rotation_id er tilladt
    if rotation_id not in ROTATIONS:
        raise ValueError(f"Ukendt monteringstype: {rotation_id}")
    
    # Lav en arbejdskopi så vi ikke ændrer originaldata
    result = df.copy()
    
    # Hent reglerne for denne monteringstype - pga lambda-funktionerne bliver det til en funktion der kan anvendes på DataFrame
    monteringstype = ROTATIONS[rotation_id]
    transform = monteringstype["transform"]
    
    # Anvend monteringsreglerne på akslerne
    # (Dette vender og bytter akslerne baseret på hvordan enheden sidder)
    new_x, new_y, new_z = transform(df)
    
    # Sæt de nye aksler ind i resultatet
    result['acc_x'] = new_x
    result['acc_y'] = new_y
    result['acc_z'] = new_z
    
    return result


